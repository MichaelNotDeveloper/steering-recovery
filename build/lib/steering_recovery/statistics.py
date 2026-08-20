from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from steering_recovery.runtime import resolve_device, resolve_dtype, seed_everything
from steering_recovery.streaming_data import (
    HuggingFaceTextStream,
    TeacherForcedActivationIterableDataset,
    load_teacher_forced_source,
)


class RunningHiddenStatistics:
    """Numerically stable feature-wise moments with O(hidden_size) state.

    Each batch is reduced in float64 and merged using the parallel Chan/Welford
    equations. A potentially large raw sum is never accumulated step by step.
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean: torch.Tensor | None = None
        self.m2: torch.Tensor | None = None

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.ndim < 2:
            raise ValueError("hidden values must have shape [..., hidden_size]")
        flattened = (
            values.detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(-1, values.shape[-1])
        )
        batch_count = int(flattened.shape[0])
        if batch_count == 0:
            return
        batch_variance, batch_mean = torch.var_mean(flattened, dim=0, correction=0)
        batch_m2 = batch_variance * batch_count
        if not torch.isfinite(batch_mean).all() or not torch.isfinite(batch_m2).all():
            raise ValueError("hidden values produced non-finite statistics")

        if self.mean is None:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        if batch_mean.shape != self.mean.shape:
            raise ValueError("hidden size changed while collecting statistics")

        assert self.m2 is not None
        combined_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / combined_count)
        self.m2 = (
            self.m2
            + batch_m2
            + delta.square() * self.count * batch_count / combined_count
        )
        self.count = combined_count

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return float64 ``sum``, population ``variance``, and sample count."""

        if self.count == 0 or self.mean is None or self.m2 is None:
            raise ValueError("cannot finalize empty hidden statistics")
        total = self.mean * self.count
        variance = (self.m2 / self.count).clamp_min(0)
        return total, variance, self.count


def save_hidden_statistics(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically save a statistics payload."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    return path


def load_normalization_statistics(
    path: str | Path,
    *,
    expected_hidden_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load ``sum``/``variance`` and derive float32 mean/std for a denoiser."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"hidden statistics file does not exist: {path}. "
            "Run collect_hidden_statistics.py before training."
        )
    try:
        raw_payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0
        raw_payload = torch.load(path, map_location="cpu")
    if not isinstance(raw_payload, dict):
        raise TypeError("hidden statistics must be stored as a mapping")
    missing = {"sum", "variance", "count"} - raw_payload.keys()
    if missing:
        raise ValueError(
            "hidden statistics are incomplete; missing keys: "
            + ", ".join(sorted(missing))
        )

    total = torch.as_tensor(raw_payload["sum"], dtype=torch.float64)
    variance = torch.as_tensor(raw_payload["variance"], dtype=torch.float64)
    count = int(raw_payload["count"])
    if count <= 0:
        raise ValueError("hidden statistics count must be positive")
    if total.ndim != 1 or variance.ndim != 1:
        raise ValueError("sum and variance must be one-dimensional vectors")
    if total.numel() == 0 or total.shape != variance.shape:
        raise ValueError("sum and variance must be non-empty vectors of equal shape")
    if expected_hidden_size is not None and total.numel() != expected_hidden_size:
        raise ValueError(
            f"statistics hidden size {total.numel()} differs from data "
            f"{expected_hidden_size}"
        )
    if not torch.isfinite(total).all() or not torch.isfinite(variance).all():
        raise ValueError("hidden statistics must contain only finite values")
    if torch.any(variance < 0):
        raise ValueError("hidden variance must be non-negative")

    mean = (total / count).float()
    std = variance.sqrt().float()
    return mean, std, dict(raw_payload)


@torch.inference_mode()
def collect_hidden_statistics(config: DictConfig) -> dict[str, Any]:
    """Collect exact-token GPT hidden statistics from a streaming text source."""

    seed_everything(int(config.seed))
    device = resolve_device(str(config.device))
    max_tokens = int(config.collection.max_tokens)
    batch_tokens = int(config.collection.batch_tokens)
    if max_tokens <= 0:
        raise ValueError("collection.max_tokens must be positive")
    if batch_tokens <= 0:
        raise ValueError("collection.batch_tokens must be positive")

    source = config.source
    extractor = load_teacher_forced_source(
        model_name=str(source.model_name),
        tokenizer_name=source.tokenizer_name,
        layer_index=int(source.layer_index),
        layer_path=source.layer_path,
        max_length=int(source.max_length),
        device=device,
        dtype=resolve_dtype(str(source.model_dtype), device),
        trust_remote_code=bool(source.trust_remote_code),
    )
    dataset_config = config.dataset
    text_stream = HuggingFaceTextStream(
        dataset_name=str(dataset_config.name),
        dataset_config=dataset_config.config,
        split=str(dataset_config.split),
        text_column=str(dataset_config.text_column),
        skip_texts=int(dataset_config.skip_texts),
        shuffle_buffer_size=int(dataset_config.shuffle_buffer_size),
        seed=int(config.seed),
    )
    emitted_batch_size = min(batch_tokens, max_tokens)
    batches = TeacherForcedActivationIterableDataset(
        text_stream,
        extractor,
        batch_size=emitted_batch_size,
        text_batch_size=int(source.text_batch_size),
        max_batches=math.ceil(max_tokens / emitted_batch_size),
    )

    accumulator = RunningHiddenStatistics()
    progress = tqdm(
        total=max_tokens,
        desc=f"GPT hidden statistics (layer {int(source.layer_index)})",
        unit="token",
        dynamic_ncols=True,
    )
    try:
        for batch in batches:
            remaining = max_tokens - accumulator.count
            selected = batch[:remaining]
            accumulator.update(selected)
            progress.update(len(selected))
            if accumulator.count == max_tokens:
                break
    finally:
        progress.close()
    if accumulator.count != max_tokens:
        raise RuntimeError(
            f"text stream ended after {accumulator.count} hidden tokens; "
            f"expected {max_tokens}"
        )

    total, variance, count = accumulator.finalize()
    payload = {
        "format_version": 1,
        "sum": total,
        "variance": variance,
        "count": count,
        "source": {
            "model_name": str(source.model_name),
            "tokenizer_name": str(source.tokenizer_name or source.model_name),
            "layer_path": source.layer_path,
            "layer_index": int(source.layer_index),
            "max_length": int(source.max_length),
        },
        "dataset": {
            "name": str(dataset_config.name),
            "config": dataset_config.config,
            "split": str(dataset_config.split),
            "text_column": str(dataset_config.text_column),
            "skip_texts": int(dataset_config.skip_texts),
        },
    }
    output_path = save_hidden_statistics(config.output_path, payload)
    return {
        "path": str(output_path),
        "count": count,
        "hidden_size": total.numel(),
        "layer_index": int(source.layer_index),
    }
