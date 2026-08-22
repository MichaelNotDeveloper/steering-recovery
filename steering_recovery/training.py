from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from steering_recovery.checkpoint import save_checkpoint
from steering_recovery.data import ActivationDataset, split_dataset
from steering_recovery.denoiser import ActivationDenoiser, DenoiserBundle
from steering_recovery.metrics import denoising_metrics
from steering_recovery.normalization import ActivationNormalizer
from steering_recovery.runtime import (
    config_to_dict,
    dtype_name,
    ensure_output_dir,
    is_gpt2_small_model,
    resolve_device,
    resolve_model_dtype,
    seed_everything,
)
from steering_recovery.statistics import load_normalization_statistics
from steering_recovery.streaming_data import (
    HuggingFaceTextStream,
    TeacherForcedActivationIterableDataset,
    load_teacher_forced_source,
)
from steering_recovery.tracking import Tracker

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelRun:
    name: str
    latent_dim: int
    num_layers: int
    sigma: float
    dropout: float
    bundle: DenoiserBundle
    optimizer: torch.optim.Optimizer
    directory: Path
    best_l2: float = float("inf")
    best_step: int = 0
    best_validation: dict[str, float] = field(default_factory=dict)
    last_validation: dict[str, float] = field(default_factory=dict)

    @property
    def parameters(self) -> dict[str, int | float]:
        return {
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
            "sigma": self.sigma,
            "dropout": self.dropout,
        }


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)


def _validate_streaming_statistics_source(
    statistics: Mapping[str, Any],
    stream_config: DictConfig,
    *,
    device: torch.device,
) -> None:
    source = statistics.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(
            "streaming statistics must include source metadata; regenerate them "
            "with collect_hidden_statistics.py"
        )
    expected = {
        "model_name": str(stream_config.model_name),
        "tokenizer_name": str(stream_config.tokenizer_name or stream_config.model_name),
        "layer_path": stream_config.layer_path,
        "layer_index": int(stream_config.layer_index),
        "max_length": int(stream_config.max_length),
        "model_dtype": dtype_name(
            resolve_model_dtype(
                str(stream_config.model_name),
                str(stream_config.model_dtype),
                device,
            )
        ),
    }
    mismatches = [
        f"{key}: statistics={source.get(key)!r}, training={value!r}"
        for key, value in expected.items()
        if source.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "statistics source does not match the training source ("
            + "; ".join(mismatches)
            + ")"
        )


def build_streaming_activation_datasets(
    data_config: DictConfig,
    training_config: DictConfig,
    *,
    device: torch.device,
    seed: int,
) -> tuple[
    TeacherForcedActivationIterableDataset,
    TeacherForcedActivationIterableDataset | None,
]:
    """Build restartable and non-overlapping train/validation streams."""

    if training_config.max_steps is None:
        raise ValueError("training.max_steps is required for streaming data")
    stream_config = data_config.streaming
    model_dtype = resolve_model_dtype(
        str(stream_config.model_name), str(stream_config.model_dtype), device
    )
    extractor = load_teacher_forced_source(
        model_name=str(stream_config.model_name),
        tokenizer_name=stream_config.tokenizer_name,
        layer_index=int(stream_config.layer_index),
        layer_path=stream_config.layer_path,
        max_length=int(stream_config.max_length),
        device=device,
        dtype=model_dtype,
        trust_remote_code=bool(stream_config.trust_remote_code),
    )
    validation_texts = int(stream_config.validation_texts)
    if validation_texts <= 0:
        raise ValueError("data.streaming.validation_texts must be positive")
    train_texts = HuggingFaceTextStream(
        dataset_name=str(stream_config.dataset_name),
        dataset_config=stream_config.dataset_config,
        split=str(stream_config.split),
        text_column=str(stream_config.text_column),
        skip_texts=validation_texts,
        shuffle_buffer_size=int(stream_config.shuffle_buffer_size),
        seed=seed,
    )
    train_dataset = TeacherForcedActivationIterableDataset(
        train_texts,
        extractor,
        batch_size=int(training_config.batch_size),
        text_batch_size=int(stream_config.text_batch_size),
        max_batches=int(training_config.max_steps),
    )
    validation_text_stream = HuggingFaceTextStream(
        dataset_name=str(stream_config.dataset_name),
        dataset_config=stream_config.dataset_config,
        split=str(stream_config.split),
        text_column=str(stream_config.text_column),
        limit_texts=validation_texts,
        seed=seed,
    )
    validation_dataset = TeacherForcedActivationIterableDataset(
        validation_text_stream,
        extractor,
        batch_size=int(training_config.batch_size),
        text_batch_size=int(stream_config.text_batch_size),
        max_batches=int(training_config.validation_batches),
    )
    return train_dataset, validation_dataset


def _sigma_slug(sigma: float) -> str:
    return format(sigma, ".8g").replace("-", "m").replace(".", "p")


def _create_model_runs(
    config: DictConfig,
    *,
    hidden_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    output_dir: Path,
) -> list[ModelRun]:
    latent_dims = [int(value) for value in config.model.latent_dims]
    layer_counts = [int(value) for value in config.model.num_layers]
    sigmas = [float(value) for value in config.model.sigmas]
    dropout = float(config.model.get("dropout", 0.0))
    if not latent_dims or not layer_counts or not sigmas:
        raise ValueError(
            "model.latent_dims, model.num_layers and model.sigmas are required"
        )
    if (
        len(set(latent_dims)) != len(latent_dims)
        or len(set(layer_counts)) != len(layer_counts)
        or len(set(sigmas)) != len(sigmas)
    ):
        raise ValueError("model grid values must be unique")
    if any(value <= 0 for value in latent_dims + layer_counts):
        raise ValueError("latent dimensions and layer counts must be positive")
    if any(value <= 0 for value in sigmas):
        raise ValueError("all model.sigmas must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")

    runs: list[ModelRun] = []
    for sigma in sigmas:
        for num_layers in layer_counts:
            for latent_dim in latent_dims:
                name = (
                    f"latent_{latent_dim}_layers_{num_layers}_"
                    f"sigma_{_sigma_slug(sigma)}"
                )
                if dropout > 0:
                    name += f"_dropout_{_sigma_slug(dropout)}"
                directory = ensure_output_dir(output_dir / "models" / name)
                architecture_seed = (
                    int(config.seed) + latent_dim * 101 + num_layers * 1_000_003
                )
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(architecture_seed)
                    model = ActivationDenoiser(
                        hidden_size=hidden_size,
                        latent_dim=latent_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                    )
                bundle = DenoiserBundle(
                    model,
                    ActivationNormalizer(mean, std),
                ).to(device)
                optimizer = torch.optim.AdamW(
                    bundle.model.parameters(),
                    lr=float(config.training.learning_rate),
                    weight_decay=float(config.training.weight_decay),
                    betas=tuple(config.training.betas),
                )
                run = ModelRun(
                    name=name,
                    latent_dim=latent_dim,
                    num_layers=num_layers,
                    sigma=sigma,
                    dropout=dropout,
                    bundle=bundle,
                    optimizer=optimizer,
                    directory=directory,
                )
                _write_json(
                    directory / "model_config.json",
                    {"name": name, "parameters": run.parameters},
                )
                runs.append(run)
    return runs


def _autocast_parameters(
    precision: str, device: torch.device
) -> tuple[bool, torch.dtype]:
    if precision == "bf16":
        return device.type == "cuda", torch.bfloat16
    if precision == "fp16":
        return device.type == "cuda", torch.float16
    if precision != "fp32":
        raise ValueError("training.precision must be fp32, fp16 or bf16")
    return False, torch.float32


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _append_metrics(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def _summary_payload(run: ModelRun) -> dict[str, Any]:
    return {
        "name": run.name,
        "parameters": run.parameters,
        "best_step": run.best_step,
        "best_validation": run.best_validation,
        "last_validation": run.last_validation,
        "best_checkpoint": "best.pt",
        "last_checkpoint": "last.pt",
    }


@torch.no_grad()
def evaluate_model_grid(
    runs: list[ModelRun],
    loader: Iterable[torch.Tensor],
    normalizer: ActivationNormalizer,
    *,
    device: torch.device,
    seed: int,
    max_batches: int,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> dict[str, dict[str, float]]:
    """Evaluate every model on identical normalized batches and Gaussian noise."""

    if max_batches <= 0:
        raise ValueError("validation_batches must be positive")
    for run in runs:
        run.bundle.eval()
    totals: dict[str, defaultdict[str, float]] = {
        run.name: defaultdict(float) for run in runs
    }
    total_examples = 0
    generator = _make_generator(device, seed)
    for batch_index, clean_raw in enumerate(loader):
        if batch_index >= max_batches:
            break
        clean = normalizer.normalize(clean_raw.to(device, non_blocking=True))
        epsilon = torch.randn(
            clean.shape, device=device, dtype=clean.dtype, generator=generator
        )
        batch_size = int(clean.shape[0])
        for run in runs:
            noisy = clean + run.sigma * epsilon
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                recovered = run.bundle.model(noisy)
            metrics = denoising_metrics(
                clean,
                noisy,
                recovered,
                noise_std=run.sigma,
            )
            for key, value in metrics.items():
                totals[run.name][key] += value * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError("validation loader produced no batches")
    for run in runs:
        run.bundle.train()
    results: dict[str, dict[str, float]] = {}
    for run in runs:
        metrics = {
            key: value / total_examples for key, value in totals[run.name].items()
        }
        metrics["rmse"] = metrics["l2"] ** 0.5
        metrics["noisy_rmse"] = metrics["noisy_l2"] ** 0.5
        metrics["score_rms"] = metrics["score_mse"] ** 0.5
        results[run.name] = metrics
    return results


def _save_validation_results(
    runs: list[ModelRun],
    results: Mapping[str, Mapping[str, float]],
    *,
    step: int,
    epoch: int,
    resolved_config: Mapping[str, Any],
    tracker: Tracker,
) -> None:
    wandb_metrics: dict[str, float] = {}
    for run in runs:
        metrics = dict(results[run.name])
        run.last_validation = metrics
        record = {"split": "validation", "step": step, **metrics}
        _append_metrics(run.directory / "metrics.jsonl", record)
        for key, value in metrics.items():
            wandb_metrics[f"models/{run.name}/val/{key}"] = value
        if metrics["l2"] < run.best_l2:
            run.best_l2 = metrics["l2"]
            run.best_step = step
            run.best_validation = metrics
            save_checkpoint(
                run.directory / "best.pt",
                run.bundle,
                step=step,
                epoch=epoch,
                config={"experiment": dict(resolved_config), "variant": run.parameters},
            )
        _write_json(run.directory / "summary.json", _summary_payload(run))
    tracker.log(wandb_metrics, step=step)


def _save_last_models(
    runs: list[ModelRun],
    *,
    step: int,
    epoch: int,
    resolved_config: Mapping[str, Any],
) -> None:
    for run in runs:
        save_checkpoint(
            run.directory / "last.pt",
            run.bundle,
            step=step,
            epoch=epoch,
            config={"experiment": dict(resolved_config), "variant": run.parameters},
            optimizer=run.optimizer,
        )
        _write_json(run.directory / "summary.json", _summary_payload(run))


def train_denoiser(config: DictConfig, output_dir: str | Path) -> dict[str, Any]:
    """Train the complete latent_dim/layer_count/sigma grid in lockstep."""

    seed_everything(int(config.seed))
    output_dir = ensure_output_dir(output_dir)
    device = resolve_device(str(config.device))
    statistics_path = config.data.statistics_path
    if not statistics_path:
        raise ValueError(
            "data.statistics_path is required; run collect_hidden_statistics.py "
            "before training"
        )
    mean, std, statistics = load_normalization_statistics(statistics_path)
    data_mode = str(config.data.get("mode", "static"))
    if data_mode == "streaming":
        _validate_streaming_statistics_source(
            statistics, config.data.streaming, device=device
        )
        if (
            is_gpt2_small_model(str(config.data.streaming.model_name))
            and str(config.training.precision) != "fp32"
        ):
            raise ValueError(
                "GPT-2 Small denoiser experiments require training.precision=fp32"
            )
        train_loader, val_loader = build_streaming_activation_datasets(
            config.data,
            config.training,
            device=device,
            seed=int(config.seed),
        )
        hidden_size = train_loader.hidden_size
    elif data_mode == "static":
        dataset = ActivationDataset(config.data.path, key=config.data.key)
        train_dataset, val_dataset = split_dataset(
            dataset, float(config.data.val_fraction), int(config.seed)
        )
        if val_dataset is None:
            raise ValueError("a non-empty validation split is required")
        hidden_size = dataset.hidden_size
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config.training.batch_size),
            shuffle=True,
            num_workers=int(config.data.num_workers),
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config.training.batch_size),
            shuffle=False,
            num_workers=int(config.data.num_workers),
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
    else:
        raise ValueError("data.mode must be 'streaming' or 'static'")

    expected_hidden = config.model.hidden_size
    if expected_hidden is not None and int(expected_hidden) != hidden_size:
        raise ValueError(
            f"configured hidden size {expected_hidden} differs from data {hidden_size}"
        )
    if mean.numel() != hidden_size:
        raise ValueError(
            f"statistics hidden size {mean.numel()} differs from data {hidden_size}"
        )

    validation_every = int(config.training.validation_every_batches)
    validation_batches = int(config.training.validation_batches)
    log_every = int(config.training.log_every_batches)
    if min(validation_every, validation_batches, log_every) <= 0:
        raise ValueError("validation and logging batch intervals must be positive")
    max_steps = (
        int(config.training.max_steps)
        if config.training.max_steps is not None
        else None
    )
    normalizer = ActivationNormalizer(mean, std).to(device)
    runs = _create_model_runs(
        config,
        hidden_size=hidden_size,
        mean=mean,
        std=std,
        device=device,
        output_dir=output_dir,
    )
    torch.save(statistics, output_dir / "statistics.pt")
    OmegaConf.save(config, output_dir / "config.yaml")
    resolved_config = config_to_dict(config)
    tracker = Tracker.create(
        enabled=bool(config.wandb.enabled),
        project=str(config.wandb.project),
        name=config.wandb.name,
        entity=config.wandb.entity,
        mode=str(config.wandb.mode),
        config=resolved_config,
        tags=list(config.wandb.tags),
    )
    autocast_enabled, autocast_dtype = _autocast_parameters(
        str(config.training.precision), device
    )
    train_generator = _make_generator(device, int(config.seed))
    global_step = 0
    last_validation_step = -1
    last_epoch = 0

    try:
        for epoch in range(int(config.training.epochs)):
            last_epoch = epoch
            epoch_setter = getattr(train_loader, "set_epoch", None)
            if epoch_setter is not None:
                epoch_setter(epoch)
            for run in runs:
                run.bundle.train()
            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}", dynamic_ncols=True)
            for clean_raw in progress:
                global_step += 1
                clean = normalizer.normalize(clean_raw.to(device, non_blocking=True))
                epsilon = torch.randn(
                    clean.shape,
                    device=device,
                    dtype=clean.dtype,
                    generator=train_generator,
                )
                should_log = global_step % log_every == 0
                train_log: dict[str, float] = {}
                mean_l2 = 0.0
                for run in runs:
                    run.optimizer.zero_grad(set_to_none=True)
                    noisy = clean + run.sigma * epsilon
                    with torch.autocast(
                        device_type=device.type,
                        dtype=autocast_dtype,
                        enabled=autocast_enabled,
                    ):
                        recovered = run.bundle.model(noisy)
                        loss = torch.nn.functional.mse_loss(
                            recovered.float(), clean.float()
                        )
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        run.bundle.model.parameters(),
                        float(config.training.max_grad_norm),
                    )
                    run.optimizer.step()
                    mean_l2 += float(loss.item())
                    if should_log:
                        metrics = denoising_metrics(
                            clean,
                            noisy,
                            recovered.detach(),
                            noise_std=run.sigma,
                        )
                        record = {
                            "split": "train",
                            "step": global_step,
                            "grad_norm": float(grad_norm),
                            **metrics,
                        }
                        _append_metrics(run.directory / "metrics.jsonl", record)
                        for key, value in metrics.items():
                            train_log[f"models/{run.name}/train/{key}"] = value
                        train_log[f"models/{run.name}/train/grad_norm"] = float(
                            grad_norm
                        )
                progress.set_postfix(l2=f"{mean_l2 / len(runs):.5f}")
                if should_log:
                    tracker.log(train_log, step=global_step)

                if global_step % validation_every == 0:
                    results = evaluate_model_grid(
                        runs,
                        val_loader,
                        normalizer,
                        device=device,
                        seed=int(config.seed) + 10_000,
                        max_batches=validation_batches,
                        autocast_enabled=autocast_enabled,
                        autocast_dtype=autocast_dtype,
                    )
                    _save_validation_results(
                        runs,
                        results,
                        step=global_step,
                        epoch=epoch,
                        resolved_config=resolved_config,
                        tracker=tracker,
                    )
                    last_validation_step = global_step
                if max_steps is not None and global_step >= max_steps:
                    break
            if max_steps is not None and global_step >= max_steps:
                break

        if global_step == 0:
            raise ValueError("training loader produced no batches")
        if last_validation_step != global_step:
            results = evaluate_model_grid(
                runs,
                val_loader,
                normalizer,
                device=device,
                seed=int(config.seed) + 10_000,
                max_batches=validation_batches,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
            )
            _save_validation_results(
                runs,
                results,
                step=global_step,
                epoch=last_epoch,
                resolved_config=resolved_config,
                tracker=tracker,
            )
        _save_last_models(
            runs,
            step=global_step,
            epoch=last_epoch,
            resolved_config=resolved_config,
        )
        grid_summary = {
            "steps": global_step,
            "models": [_summary_payload(run) for run in runs],
        }
        _write_json(output_dir / "grid_summary.json", grid_summary)
        for run in runs:
            tracker.save(run.directory / "best.pt")
    finally:
        tracker.finish()

    best_run = min(runs, key=lambda run: run.best_l2)
    return {
        "output_dir": str(output_dir),
        "steps": global_step,
        "models": len(runs),
        "best_model": best_run.name,
        "best_validation_l2": best_run.best_l2,
    }
