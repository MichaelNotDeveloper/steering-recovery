from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from steering_recovery.checkpoint import load_checkpoint
from steering_recovery.data import load_tensor
from steering_recovery.generation import generate_with_intervention
from steering_recovery.intervention import (
    ActivationIntervention,
    InterventionController,
)
from steering_recovery.metrics import generation_metrics
from steering_recovery.reporting import (
    load_prompt_records,
    write_html_report,
    write_jsonl,
)
from steering_recovery.runtime import (
    config_to_dict,
    ensure_output_dir,
    resolve_device,
    resolve_dtype,
    seed_everything,
)
from steering_recovery.tracking import Tracker


LOGGER = logging.getLogger(__name__)


def _load_model_and_tokenizer(config: DictConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(str(config.device))
    dtype = resolve_dtype(str(config.dtype), device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.name, trust_remote_code=bool(config.trust_remote_code)
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(config.trust_remote_code),
        "torch_dtype": dtype,
    }
    if config.device_map:
        kwargs["device_map"] = config.device_map
    model = AutoModelForCausalLM.from_pretrained(config.name, **kwargs)
    if not config.device_map:
        model.to(device)
    model.eval()
    return model, tokenizer, device, dtype


def _steering_vector(config: DictConfig, hidden_size: int) -> torch.Tensor:
    if not config.path:
        if float(config.scale) == 0:
            return torch.zeros(hidden_size)
        raise ValueError("steering.vector_path is required when scale is non-zero")
    vector = load_tensor(config.path, key=config.key).float().squeeze()
    if vector.ndim != 1 or vector.numel() != hidden_size:
        raise ValueError(
            f"steering vector must have shape [{hidden_size}], got {tuple(vector.shape)}"
        )
    return vector


def run_generation_records(
    *,
    model: torch.nn.Module,
    tokenizer: object,
    records: list[dict[str, Any]],
    steering_vector: torch.Tensor,
    config: DictConfig,
    denoiser=None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(tqdm(records, desc="prompts")):
        for sample_index in range(int(config.generation.num_samples)):
            seed = int(config.generation.seed) + sample_index
            controller = InterventionController(
                mode=str(config.steering.mode),
                scale=float(config.steering.scale),
                entropy_threshold=float(config.steering.entropy_threshold),
            )
            intervention = ActivationIntervention(
                model,
                steering_vector,
                layer_index=int(config.steering.layer_index),
                layer_path=config.steering.layer_path,
                controller=controller,
                denoiser=denoiser,
            )
            trace = generate_with_intervention(
                model,
                tokenizer,
                record["prompt"],
                intervention,
                controller,
                max_new_tokens=int(config.generation.max_new_tokens),
                temperature=float(config.generation.temperature),
                top_p=float(config.generation.top_p),
                seed=seed,
            )
            rows.append(
                {
                    "prompt_id": record["id"],
                    "prompt": record["prompt"],
                    "sample_index": sample_index,
                    "seed": seed,
                    "generated_text": trace.text,
                    "token_ids": trace.token_ids,
                    "normalized_entropies": trace.normalized_entropies,
                    "intervention_steps": trace.intervention_steps,
                    "forward_calls": trace.forward_calls,
                    "steering": {
                        "mode": str(config.steering.mode),
                        "scale": float(config.steering.scale),
                        "entropy_threshold": float(config.steering.entropy_threshold),
                        "layer_index": int(config.steering.layer_index),
                    },
                    "denoiser_enabled": denoiser is not None,
                }
            )
    return rows


def run_baseline(config: DictConfig, output_dir: str | Path) -> dict[str, Any]:
    seed_everything(int(config.generation.seed))
    output_dir = ensure_output_dir(output_dir)
    model, tokenizer, device, dtype = _load_model_and_tokenizer(config.model)
    hidden_size = int(getattr(model.config, "hidden_size"))
    vector = _steering_vector(config.steering.vector, hidden_size)
    records = load_prompt_records(config.data.prompts_path)
    if config.data.max_prompts is not None:
        records = records[: int(config.data.max_prompts)]
    denoiser = None
    if config.denoiser.enabled:
        denoiser, _ = load_checkpoint(
            config.denoiser.checkpoint, device=device, dtype=dtype
        )

    resolved = config_to_dict(config)
    OmegaConf.save(config, output_dir / "config.yaml")
    tracker = Tracker.create(
        enabled=bool(config.wandb.enabled),
        project=str(config.wandb.project),
        name=config.wandb.name,
        entity=config.wandb.entity,
        mode=str(config.wandb.mode),
        config=resolved,
        tags=list(config.wandb.tags),
    )
    try:
        rows = run_generation_records(
            model=model,
            tokenizer=tokenizer,
            records=records,
            steering_vector=vector,
            config=config,
            denoiser=denoiser,
        )
        metrics = generation_metrics(rows)
        write_jsonl(rows, output_dir / "generations.jsonl")
        write_html_report(
            rows,
            output_dir / "report.html",
            settings={
                "mode": config.steering.mode,
                "scale": config.steering.scale,
                "entropy_threshold": config.steering.entropy_threshold,
                "denoiser": bool(config.denoiser.enabled),
            },
        )
        write_jsonl([metrics], output_dir / "metrics.jsonl")
        tracker.log({f"baseline/{key}": value for key, value in metrics.items()})
        tracker.save(output_dir / "generations.jsonl")
        tracker.save(output_dir / "report.html")
    finally:
        tracker.finish()
    return {"output_dir": str(output_dir), "metrics": metrics}
