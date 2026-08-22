from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from steering_recovery.denoiser import ActivationDenoiser, DenoiserBundle
from steering_recovery.normalization import ActivationNormalizer
from steering_recovery.runtime import is_gpt2_small_model


def save_checkpoint(
    path: str | Path,
    bundle: DenoiserBundle,
    *,
    step: int,
    epoch: int,
    config: Mapping[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 2,
        "model_config": bundle.model_config,
        "model_state": bundle.model.state_dict(),
        "normalizer_state": bundle.normalizer.state_dict(),
        "normalizer_eps": bundle.normalizer.eps,
        "step": int(step),
        "epoch": int(epoch),
        "config": dict(config or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[DenoiserBundle, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("format_version") != 2:
        raise ValueError(
            "unsupported denoiser checkpoint format; retrain the residual MLP grid"
        )
    model = ActivationDenoiser(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    normalizer_state = payload["normalizer_state"]
    normalizer = ActivationNormalizer(
        normalizer_state["mean"],
        normalizer_state["std"],
        eps=float(payload.get("normalizer_eps", 1e-6)),
    )
    bundle = DenoiserBundle(model, normalizer).to(device=device)
    if dtype is not None:
        bundle = bundle.to(dtype=dtype)
    bundle.eval()
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state", "normalizer_state"}
    }
    return bundle, metadata


def validate_gpt2_small_denoiser_precision(
    metadata: Mapping[str, Any], *, source_model_name: str
) -> None:
    """Reject denoisers trained from reduced-precision GPT-2 Small states."""

    if not is_gpt2_small_model(source_model_name):
        return
    config = metadata.get("config")
    experiment = config.get("experiment") if isinstance(config, Mapping) else None
    data = experiment.get("data") if isinstance(experiment, Mapping) else None
    streaming = data.get("streaming") if isinstance(data, Mapping) else None
    training = experiment.get("training") if isinstance(experiment, Mapping) else None
    source_dtype = (
        streaming.get("model_dtype") if isinstance(streaming, Mapping) else None
    )
    training_precision = (
        training.get("precision") if isinstance(training, Mapping) else None
    )
    if source_dtype not in {"float32", "fp32"} or training_precision != "fp32":
        raise ValueError(
            "GPT-2 Small experiments require a denoiser trained entirely in "
            "float32; retrain this checkpoint with source model_dtype=float32 "
            "and training.precision=fp32"
        )
