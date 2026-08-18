from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from steering_recovery.checkpoint import save_checkpoint
from steering_recovery.corruption import OnlineCorruptor
from steering_recovery.data import (
    ActivationDataset,
    compute_statistics,
    load_tensor,
    split_dataset,
)
from steering_recovery.denoiser import ActivationDenoiser, DenoiserBundle
from steering_recovery.metrics import denoising_metrics
from steering_recovery.normalization import ActivationNormalizer
from steering_recovery.runtime import (
    config_to_dict,
    ensure_output_dir,
    resolve_device,
    seed_everything,
)
from steering_recovery.tracking import Tracker

LOGGER = logging.getLogger(__name__)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)


def _load_steering_vectors(
    path: str | None,
    key: str,
    normalizer: ActivationNormalizer,
) -> torch.Tensor | None:
    if not path:
        return None
    vectors = load_tensor(path, key=key).float()
    if vectors.ndim == 1:
        vectors = vectors[None]
    if vectors.ndim != 2 or vectors.shape[-1] != normalizer.hidden_size:
        raise ValueError(
            f"steering vectors must have shape [n, {normalizer.hidden_size}], got {vectors.shape}"
        )
    return normalizer.normalize_delta(vectors)


def _make_corruptor(
    config: DictConfig, vectors: torch.Tensor | None
) -> OnlineCorruptor:
    return OnlineCorruptor(
        gaussian_std_min=config.gaussian_std_min,
        gaussian_std_max=config.gaussian_std_max,
        steering_probability=config.steering_probability,
        steering_scale_min=config.steering_scale_min,
        steering_scale_max=config.steering_scale_max,
        identity_probability=config.identity_probability,
        bidirectional=config.bidirectional,
        steering_vectors=vectors,
    )


def _cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_factor: float,
):
    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_factor + (1 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def evaluate_denoiser(
    bundle: DenoiserBundle,
    loader: DataLoader[torch.Tensor],
    corruptor: OnlineCorruptor,
    device: torch.device,
    seed: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    bundle.eval()
    generator = _make_generator(device, seed)  # cool
    totals: defaultdict[str, float] = defaultdict(float)  # cool
    total_examples = 0
    steering_examples = 0
    for batch_index, clean_raw in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        clean_raw = clean_raw.to(device)
        clean = bundle.normalizer.normalize(clean_raw)
        corruption = corruptor(clean, generator=generator)
        predicted = bundle.model(corruption.noisy, corruption.noise_level)
        denoised = corruption.noisy - predicted
        metrics = denoising_metrics(clean, corruption.noisy, denoised)
        batch_size = clean.shape[0]
        for key, value in metrics.items():
            totals[key] += value * batch_size
        totals["loss"] += (
            torch.nn.functional.mse_loss(
                predicted.float(), corruption.corruption.float()
            ).item()
            * batch_size
        )
        total_examples += batch_size
        steering_examples += int(corruption.used_steering.sum().item())
    if total_examples == 0:
        raise ValueError("validation loader produced no batches")
    result = {key: value / total_examples for key, value in totals.items()}
    result["steering_fraction"] = steering_examples / total_examples
    return result


def train_denoiser(config: DictConfig, output_dir: str | Path) -> dict[str, Any]:
    seed_everything(int(config.seed))
    output_dir = ensure_output_dir(output_dir)
    device = resolve_device(str(config.device))
    dataset = ActivationDataset(config.data.path, key=config.data.key)
    train_dataset, val_dataset = split_dataset(
        dataset, float(config.data.val_fraction), int(config.seed)
    )
    expected_hidden = config.model.hidden_size
    if expected_hidden is not None and int(expected_hidden) != dataset.hidden_size:
        raise ValueError(
            f"configured hidden size {expected_hidden} differs from data {dataset.hidden_size}"
        )

    statistics_path = config.data.statistics_path
    if statistics_path:
        statistics = torch.load(statistics_path, map_location="cpu", weights_only=True)
        mean, std = statistics["mean"], statistics["std"]
    else:
        LOGGER.info("Computing normalization statistics over the training split")
        mean, std = compute_statistics(
            train_dataset,
            batch_size=int(config.data.statistics_batch_size),
            num_workers=int(config.data.num_workers),
        )
    torch.save({"mean": mean, "std": std}, output_dir / "statistics.pt")
    normalizer = ActivationNormalizer(mean, std)
    model = ActivationDenoiser(
        hidden_size=dataset.hidden_size,
        width=int(config.model.width),
        depth=int(config.model.depth),
        expansion=int(config.model.expansion),
        dropout=float(config.model.dropout),
    )
    bundle = DenoiserBundle(model, normalizer).to(device)
    vectors = _load_steering_vectors(
        config.corruption.steering_vectors_path,
        config.corruption.steering_vectors_key,
        normalizer,
    )
    if float(config.corruption.steering_probability) > 0 and vectors is None:
        LOGGER.warning(
            "steering_probability is non-zero, but no steering vectors were provided; "
            "training will use Gaussian and identity corruptions only"
        )
    corruptor = _make_corruptor(config.corruption, vectors)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.training.batch_size),
        shuffle=True,
        num_workers=int(config.data.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=int(config.training.batch_size),
            shuffle=False,
            num_workers=int(config.data.num_workers),
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        if val_dataset is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        bundle.model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
        betas=tuple(config.training.betas),
    )
    accumulation = int(config.training.gradient_accumulation_steps)
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(config.training.log_every_steps) <= 0:
        raise ValueError("log_every_steps must be positive")
    if int(config.training.save_every_steps) <= 0:
        raise ValueError("save_every_steps must be positive")
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    planned_steps = steps_per_epoch * int(config.training.epochs)
    max_steps = config.training.max_steps
    total_steps = min(planned_steps, int(max_steps)) if max_steps else planned_steps
    warmup_steps = round(total_steps * float(config.training.warmup_ratio))
    scheduler = _cosine_schedule(
        optimizer, total_steps, warmup_steps, float(config.training.min_lr_factor)
    )

    resolved_config = config_to_dict(config)
    OmegaConf.save(config, output_dir / "config.yaml")
    tracker = Tracker.create(
        enabled=bool(config.wandb.enabled),
        project=str(config.wandb.project),
        name=config.wandb.name,
        entity=config.wandb.entity,
        mode=str(config.wandb.mode),
        config=resolved_config,
        tags=list(config.wandb.tags),
    )
    generator = _make_generator(device, int(config.seed))
    precision = str(config.training.precision)
    use_autocast = precision == "bf16" and device.type == "cuda"
    global_step = 0
    best_loss = float("inf")
    final_metrics: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(int(config.training.epochs)):
            bundle.train()
            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}", dynamic_ncols=True)
            pending = 0
            for batch_index, clean_raw in enumerate(progress):
                clean_raw = clean_raw.to(device, non_blocking=True)
                clean = bundle.normalizer.normalize(clean_raw)
                corruption = corruptor(clean, generator=generator)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_autocast,
                ):
                    predicted = bundle.model(corruption.noisy, corruption.noise_level)
                    raw_loss = torch.nn.functional.mse_loss(
                        predicted.float(), corruption.corruption.float()
                    )
                    group_start = (batch_index // accumulation) * accumulation
                    group_size = min(accumulation, len(train_loader) - group_start)
                    loss = raw_loss / group_size
                loss.backward()
                pending += 1
                is_last_batch = batch_index + 1 == len(train_loader)
                if pending < accumulation and not is_last_batch:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    bundle.model.parameters(), float(config.training.max_grad_norm)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                global_step += 1
                with torch.no_grad():
                    denoised = corruption.noisy - predicted
                    batch_metrics = denoising_metrics(clean, corruption.noisy, denoised)
                train_metrics = {
                    "train/loss": float(raw_loss.item()),
                    "train/learning_rate": float(scheduler.get_last_lr()[0]),
                    "train/grad_norm": float(grad_norm),
                    "train/steering_fraction": float(
                        corruption.used_steering.float().mean().item()
                    ),
                    **{f"train/{key}": value for key, value in batch_metrics.items()},
                }
                if global_step % int(config.training.log_every_steps) == 0:
                    tracker.log(train_metrics, step=global_step)
                progress.set_postfix(loss=f"{raw_loss.item():.4f}")
                if global_step % int(config.training.save_every_steps) == 0:
                    save_checkpoint(
                        output_dir / f"step_{global_step}.pt",
                        bundle,
                        step=global_step,
                        epoch=epoch,
                        config=resolved_config,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                if max_steps and global_step >= int(max_steps):
                    break

            if val_loader is not None:
                final_metrics = evaluate_denoiser(
                    bundle,
                    val_loader,
                    corruptor,
                    device,
                    seed=int(config.seed) + 10_000,
                    max_batches=config.training.validation_batches,
                )
                tracker.log(
                    {f"val/{key}": value for key, value in final_metrics.items()},
                    step=global_step,
                )
                if final_metrics["loss"] < best_loss:
                    best_loss = final_metrics["loss"]
                    save_checkpoint(
                        output_dir / "best.pt",
                        bundle,
                        step=global_step,
                        epoch=epoch,
                        config=resolved_config,
                    )
            save_checkpoint(
                output_dir / "last.pt",
                bundle,
                step=global_step,
                epoch=epoch,
                config=resolved_config,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            if max_steps and global_step >= int(max_steps):
                break
        tracker.save(output_dir / "last.pt")
    finally:
        tracker.finish()
    return {
        "output_dir": str(output_dir),
        "steps": global_step,
        "best_validation_loss": best_loss,
        "validation": final_metrics,
    }
