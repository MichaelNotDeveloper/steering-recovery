from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


LOGGER = logging.getLogger(__name__)

_GPT2_SMALL_MODEL_IDS = frozenset({"gpt2", "openai-community/gpt2"})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if value not in aliases:
        raise ValueError(f"unknown dtype {value!r}")
    if device.type == "cpu" and aliases[value] == torch.float16:
        LOGGER.warning("float16 on CPU is unsupported by many kernels; using float32")
        return torch.float32
    return aliases[value]


def is_gpt2_small_model(model_name: str) -> bool:
    """Return whether ``model_name`` is a known GPT-2 Small model ID."""

    normalized = str(model_name).strip().rstrip("/").lower()
    return normalized in _GPT2_SMALL_MODEL_IDS


def resolve_model_dtype(
    model_name: str, value: str, device: torch.device
) -> torch.dtype:
    """Resolve model precision and enforce FP32 for GPT-2 Small.

    ``auto`` intentionally means FP32 for GPT-2 Small on every device. Explicit
    reduced-precision requests fail instead of silently producing artifacts whose
    recorded configuration disagrees with the actual computation.
    """

    if is_gpt2_small_model(model_name):
        if value == "auto":
            return torch.float32
        dtype = resolve_dtype(value, device)
        if dtype != torch.float32:
            raise ValueError(
                "GPT-2 Small experiments require float32; "
                f"got model dtype {value!r}"
            )
        return dtype
    return resolve_dtype(value, device)


def dtype_name(dtype: torch.dtype) -> str:
    """Return the stable config spelling for a torch dtype."""

    return str(dtype).removeprefix("torch.")


def config_to_dict(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


def ensure_output_dir(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output
