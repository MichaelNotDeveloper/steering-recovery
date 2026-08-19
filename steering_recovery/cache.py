from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from steering_recovery.layers import first_tensor, resolve_layer
from steering_recovery.runtime import (
    config_to_dict,
    ensure_output_dir,
    resolve_device,
    resolve_dtype,
    seed_everything,
)

LOGGER = logging.getLogger(__name__)


def select_token_activations(
    hidden: torch.Tensor, attention_mask: torch.Tensor, selection: str
) -> torch.Tensor:
    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError(
            "hidden/mask must have shapes [batch, seq, hidden] and [batch, seq]"
        )
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError("hidden and attention mask batch/sequence dimensions differ")
    mask = attention_mask.bool()
    if selection == "all":
        return hidden[mask]
    if selection == "last":
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None].expand_as(
            mask
        )
        last = positions.masked_fill(~mask, -1).max(dim=1).values
        if torch.any(last < 0):
            raise ValueError("cannot select a last token from an empty sequence")
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    raise ValueError("selection must be 'last' or 'all'")


class NpyShardWriter:
    def __init__(self, output_dir: str | Path, shard_size: int, dtype: str = "float32"):
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.output_dir = ensure_output_dir(output_dir)
        self.shard_size = int(shard_size)
        self.dtype = np.dtype(dtype)
        self.buffer: list[np.ndarray] = []
        self.buffered = 0
        self.count = 0
        self.shards: list[dict[str, Any]] = []

    def add(self, values: torch.Tensor | np.ndarray) -> None:
        array = (
            values.detach().cpu().numpy()
            if torch.is_tensor(values)
            else np.asarray(values)
        )
        array = array.astype(self.dtype, copy=False)
        if array.ndim != 2:
            raise ValueError("activation chunks must have shape [examples, hidden]")
        offset = 0
        while offset < len(array):
            take = min(self.shard_size - self.buffered, len(array) - offset)
            self.buffer.append(array[offset : offset + take])
            self.buffered += take
            offset += take
            if self.buffered == self.shard_size:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        array = np.concatenate(self.buffer, axis=0)
        filename = f"activations_{len(self.shards):05d}.npy"
        np.save(self.output_dir / filename, array)
        self.shards.append({"file": filename, "examples": len(array)})
        self.count += len(array)
        self.buffer.clear()
        self.buffered = 0

    def close(self) -> list[dict[str, Any]]:
        self.flush()
        return self.shards


class _Capture:
    def __init__(self):
        self.value: torch.Tensor | None = None

    def hook(self, _module, _inputs, output):
        value = first_tensor(output)
        if value is None:
            raise TypeError("selected layer did not return a tensor")
        self.value = value.detach()


def _load_text_dataset(config: DictConfig):
    from datasets import load_dataset

    path = Path(str(config.path))
    if path.is_file():
        extension = "json" if path.suffix in {".json", ".jsonl"} else "text"
        return load_dataset(extension, data_files=str(path), split=str(config.split))
    kwargs = {"split": str(config.split)}
    if config.name:
        kwargs["name"] = str(config.name)
    return load_dataset(str(config.path), streaming=True, **kwargs)


@torch.inference_mode()
def cache_activations(config: DictConfig, output_dir: str | Path) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_everything(int(config.seed))
    output_dir = ensure_output_dir(output_dir)
    device = resolve_device(str(config.model.device))
    dtype = resolve_dtype(str(config.model.dtype), device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name, trust_remote_code=bool(config.model.trust_remote_code)
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = str(config.tokenization.padding_side)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=dtype,
        trust_remote_code=bool(config.model.trust_remote_code),
    ).to(device)
    model.eval()
    dataset = _load_text_dataset(config.dataset)
    if config.dataset.max_samples is not None:
        dataset = dataset.select(
            range(min(int(config.dataset.max_samples), len(dataset)))
        )

    layer = resolve_layer(
        model, int(config.capture.layer_index), config.capture.layer_path
    )
    capture = _Capture()
    handle = layer.register_forward_hook(capture.hook)
    writer = NpyShardWriter(
        output_dir,
        shard_size=int(config.capture.shard_size),
        dtype=str(config.capture.output_dtype),
    )
    count = 0
    running_sum: torch.Tensor | None = None
    running_square_sum: torch.Tensor | None = None
    try:
        batch_size = int(config.tokenization.batch_size)
        for start in tqdm(range(0, len(dataset), batch_size), desc="caching"):
            batch = dataset[start : start + batch_size]
            texts = batch[str(config.dataset.text_column)]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(config.tokenization.max_length),
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            capture.value = None
            model(**encoded, use_cache=False)
            if capture.value is None:
                raise RuntimeError("activation hook was not called")
            selected = (
                select_token_activations(
                    capture.value,
                    encoded["attention_mask"],
                    str(config.capture.token_selection),
                )
                .float()
                .cpu()
            )
            writer.add(selected)
            values = selected.double()
            batch_sum = values.sum(dim=0)
            batch_square_sum = values.square().sum(dim=0)
            running_sum = batch_sum if running_sum is None else running_sum + batch_sum
            running_square_sum = (
                batch_square_sum
                if running_square_sum is None
                else running_square_sum + batch_square_sum
            )
            count += len(values)
    finally:
        handle.remove()
    shards = writer.close()
    if count == 0 or running_sum is None or running_square_sum is None:
        raise ValueError("no activations were captured")
    mean = running_sum / count
    variance = (running_square_sum / count - mean.square()).clamp_min(0)
    torch.save(
        {"mean": mean.float(), "std": variance.sqrt().float()},
        output_dir / "statistics.pt",
    )
    manifest = {
        "format_version": 1,
        "examples": count,
        "hidden_size": int(mean.numel()),
        "shards": shards,
        "config": config_to_dict(config),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OmegaConf.save(config, output_dir / "config.yaml")
    return manifest
