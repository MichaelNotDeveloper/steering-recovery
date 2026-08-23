from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from steering_recovery.checkpoint import (
    load_checkpoint,
    validate_gpt2_small_denoiser_precision,
)
from steering_recovery.runtime import (
    config_to_dict,
    ensure_output_dir,
    resolve_device,
    resolve_model_dtype,
    seed_everything,
)
from steering_recovery.steering.benchmarking.data import (
    BenchmarkPrompt,
    load_ag_news_prompts,
)
from steering_recovery.steering.benchmarking.runner import (
    validate_gpt2_small_vector_precision,
)
from steering_recovery.steering.epistemic.generation import (
    generate_epistemic_continuation,
)
from steering_recovery.steering.epistemic.plotting import (
    plot_epistemic_summaries,
)
from steering_recovery.steering.epistemic.reporting import (
    write_epistemic_examples_html,
)
from steering_recovery.steering.epistemic.statistics import (
    METRIC_LABELS,
    summarize_token_metrics,
)


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_DIAGNOSTICS_VERSION = 2


@dataclass(frozen=True)
class EpistemicVector:
    name: str
    slug: str
    dataset_label: int
    vector: torch.Tensor


@dataclass(frozen=True)
class EpistemicDenoiser:
    sigma: float
    model_directory: str
    checkpoint: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class EpistemicCondition:
    denoiser: EpistemicDenoiser
    vector: EpistemicVector
    alpha: float

    @property
    def condition_id(self) -> str:
        return (
            f"sigma_{_number_slug(self.denoiser.sigma)}__{self.vector.slug}__"
            f"alpha_{_number_slug(self.alpha)}"
        )


def _number_slug(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"artifact must contain a mapping: {path}")
    return payload


def _load_vectors(path: str | Path) -> tuple[list[EpistemicVector], dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _safe_torch_load(path)
    required = {"steering_vectors", "vector_names", "vector_slugs", "group_labels"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"steering artifact is missing keys: {sorted(missing)}")
    matrix = torch.as_tensor(payload["steering_vectors"]).float()
    names = list(payload["vector_names"])
    slugs = list(payload["vector_slugs"])
    labels = list(payload["group_labels"])
    if matrix.ndim != 2 or not (len(matrix) == len(names) == len(slugs) == len(labels)):
        raise ValueError("steering vector matrix and metadata lengths differ")
    vectors = []
    for index, (name, slug, label) in enumerate(zip(names, slugs, labels)):
        slug = str(slug)
        if not _SAFE_NAME.fullmatch(slug):
            raise ValueError(f"unsafe steering vector slug {slug!r}")
        vectors.append(
            EpistemicVector(
                name=str(name),
                slug=slug,
                dataset_label=int(label),
                vector=matrix[index],
            )
        )
    if not vectors or not torch.isfinite(matrix).all():
        raise ValueError("steering vectors must be non-empty and finite")
    return vectors, payload


def _resolve_denoisers(config: DictConfig) -> list[EpistemicDenoiser]:
    if config.denoiser_run_dir is None:
        raise ValueError(
            "denoiser_run_dir is required; point it to the output of "
            "train_denoiser.py experiment=epistemic_dropout"
        )
    run_dir = Path(str(config.denoiser_run_dir)).expanduser()
    denoisers: list[EpistemicDenoiser] = []
    for item in config.denoisers:
        sigma = float(item.sigma)
        model_directory = str(item.model_directory)
        if not _SAFE_NAME.fullmatch(model_directory):
            raise ValueError(f"unsafe denoiser model directory {model_directory!r}")
        checkpoint = (
            run_dir / "models" / model_directory / str(config.checkpoint_name)
        ).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        denoisers.append(
            EpistemicDenoiser(
                sigma=sigma,
                model_directory=model_directory,
                checkpoint=checkpoint,
                checkpoint_sha256=_file_sha256(checkpoint),
            )
        )
    if not denoisers or len({item.sigma for item in denoisers}) != len(denoisers):
        raise ValueError("denoiser sigma values must be non-empty and unique")
    return denoisers


def _validate_denoiser(
    denoiser: Any,
    metadata: Mapping[str, Any],
    spec: EpistemicDenoiser,
    *,
    expected_dropout: float,
    source_model_name: str,
    source_layer_index: int,
) -> None:
    validate_gpt2_small_denoiser_precision(
        metadata, source_model_name=source_model_name
    )
    model_config = denoiser.model.config
    if model_config.latent_dim != 3072 or model_config.num_layers != 3:
        raise ValueError("epistemic denoisers must use latent_dim=3072 and 3 blocks")
    if not math.isclose(model_config.dropout, expected_dropout):
        raise ValueError(
            f"checkpoint dropout={model_config.dropout} differs from "
            f"configured {expected_dropout}"
        )
    config = metadata.get("config")
    experiment = config.get("experiment") if isinstance(config, Mapping) else None
    data = experiment.get("data") if isinstance(experiment, Mapping) else None
    streaming = data.get("streaming") if isinstance(data, Mapping) else None
    checkpoint_model = (
        streaming.get("model_name") if isinstance(streaming, Mapping) else None
    )
    checkpoint_layer = (
        streaming.get("layer_index") if isinstance(streaming, Mapping) else None
    )
    if checkpoint_model != source_model_name or checkpoint_layer != source_layer_index:
        raise ValueError(
            "denoiser source model/layer differs from epistemic generation: "
            f"checkpoint={checkpoint_model!r}/h[{checkpoint_layer!r}], "
            f"generation={source_model_name!r}/h[{source_layer_index}]"
        )
    variant = config.get("variant") if isinstance(config, Mapping) else None
    checkpoint_sigma = variant.get("sigma") if isinstance(variant, Mapping) else None
    if checkpoint_sigma is None or not math.isclose(
        float(checkpoint_sigma), spec.sigma
    ):
        raise ValueError(
            f"checkpoint sigma={checkpoint_sigma!r} differs from {spec.sigma}"
        )


def _load_generation_model(config: Any, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.tokenizer_name or config.name),
        trust_remote_code=bool(config.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(config.name),
        torch_dtype=dtype,
        trust_remote_code=bool(config.trust_remote_code),
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer


def _release_model(value: Any) -> None:
    module = getattr(value, "model", value)
    if module is not None and hasattr(module, "to"):
        module.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _atomic_write_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _condition_path(output_dir: Path, condition: EpistemicCondition) -> Path:
    return (
        output_dir
        / "conditions"
        / f"sigma_{_number_slug(condition.denoiser.sigma)}"
        / condition.vector.slug
        / f"alpha_{_number_slug(condition.alpha)}.jsonl"
    )


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode([token_id], skip_special_tokens=False))


def _select_examples(
    rows: Sequence[dict[str, Any]], *, examples_per_condition: int
) -> list[dict[str, Any]]:
    if examples_per_condition <= 0:
        raise ValueError("examples_per_condition must be positive")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["condition_id"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for condition_rows in groups.values():
        source_labels = sorted({int(row["source_label"]) for row in condition_rows})
        ordered = sorted(condition_rows, key=lambda row: int(row["sample_index"]))
        condition_selected: list[dict[str, Any]] = []
        while len(condition_selected) < examples_per_condition:
            added = False
            for label in source_labels:
                already = {int(row["sample_index"]) for row in condition_selected}
                candidate = next(
                    (
                        row
                        for row in ordered
                        if int(row["source_label"]) == label
                        and int(row["sample_index"]) not in already
                    ),
                    None,
                )
                if candidate is not None:
                    condition_selected.append(candidate)
                    added = True
                if len(condition_selected) >= examples_per_condition:
                    break
            if not added:
                raise RuntimeError("not enough examples for an epistemic condition")
        selected.extend(condition_selected)
    return selected


def _write_summary_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_epistemic_steering(
    config: DictConfig, output_dir: str | Path
) -> dict[str, Any]:
    """Run vector × alpha × denoiser MC-dropout steering diagnostics."""

    seed = int(config.seed)
    seed_everything(seed)
    output_dir = ensure_output_dir(output_dir)
    OmegaConf.save(config, output_dir / "config.yaml")
    device = resolve_device(str(config.device))
    dtype = resolve_model_dtype(str(config.model.name), str(config.model.dtype), device)
    if dtype != torch.float32:
        raise ValueError("epistemic GPT-2 steering must run in float32")
    vectors, vector_payload = _load_vectors(config.steering_vectors_path)
    validate_gpt2_small_vector_precision(
        vector_payload, source_model_name=str(config.model.name)
    )
    denoisers = _resolve_denoisers(config)
    alphas = [float(value) for value in config.alphas]
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("alphas must be non-empty and unique")
    if any(not math.isfinite(value) for value in alphas):
        raise ValueError("alphas must be finite")
    mc_samples = int(config.mc_dropout.samples)
    if mc_samples < 2:
        raise ValueError("mc_dropout.samples must be at least two")

    model, tokenizer = _load_generation_model(config.model, device, dtype)
    hidden_size = int(
        getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0))
    )
    if hidden_size <= 0 or any(
        vector.vector.numel() != hidden_size for vector in vectors
    ):
        raise ValueError("generation model and steering vector hidden sizes differ")
    samples_per_condition = int(config.generation.samples_per_condition)
    prompt_tokens = int(config.generation.prompt_tokens)
    new_tokens = int(config.generation.new_tokens)
    topics = {vector.dataset_label: vector.name for vector in vectors}
    prompt_path = output_dir / "prompts.jsonl"
    prompt_signature = _signature(
        {
            "dataset": config_to_dict(config.dataset),
            "tokenizer": str(config.model.tokenizer_name or config.model.name),
            "samples_per_condition": samples_per_condition,
            "prompt_tokens": prompt_tokens,
            "seed": seed,
        }
    )
    prompt_metadata_path = output_dir / "prompts_metadata.json"
    prompt_metadata = (
        json.loads(prompt_metadata_path.read_text(encoding="utf-8"))
        if prompt_metadata_path.is_file()
        else {}
    )
    if (
        bool(config.resume)
        and prompt_path.is_file()
        and prompt_metadata.get("signature") == prompt_signature
    ):
        prompts = [BenchmarkPrompt.from_dict(row) for row in _load_jsonl(prompt_path)]
    else:
        prompts = load_ag_news_prompts(
            config.dataset,
            tokenizer=tokenizer,
            topics=topics,
            total_samples=samples_per_condition,
            prompt_tokens=prompt_tokens,
            seed=seed,
        )
        _atomic_write_jsonl([prompt.to_dict() for prompt in prompts], prompt_path)
        prompt_metadata_path.write_text(
            json.dumps({"signature": prompt_signature}, indent=2), encoding="utf-8"
        )
    if len(prompts) != samples_per_condition:
        raise ValueError("saved prompt count differs from samples_per_condition")

    conditions = [
        EpistemicCondition(denoiser=denoiser, vector=vector, alpha=alpha)
        for denoiser in denoisers
        for vector in vectors
        for alpha in alphas
    ]
    base_signature = {
        "diagnostics_version": _DIAGNOSTICS_VERSION,
        "model": config_to_dict(config.model),
        "generation": config_to_dict(config.generation),
        "mc_dropout": config_to_dict(config.mc_dropout),
        "prompts": [prompt.to_dict() for prompt in prompts],
        "steering_vectors_sha256": _file_sha256(Path(config.steering_vectors_path)),
    }
    condition_files: list[Path] = []
    progress = tqdm(
        total=len(conditions) * len(prompts),
        desc="epistemic steered generations",
        unit="generation",
        dynamic_ncols=True,
    )
    try:
        for denoiser_spec in denoisers:
            denoiser, denoiser_metadata = load_checkpoint(
                denoiser_spec.checkpoint, device=device, dtype=dtype
            )
            _validate_denoiser(
                denoiser,
                denoiser_metadata,
                denoiser_spec,
                expected_dropout=float(config.mc_dropout.expected_dropout),
                source_model_name=str(config.model.name),
                source_layer_index=int(config.model.layer_index),
            )
            for condition in [
                item for item in conditions if item.denoiser == denoiser_spec
            ]:
                path = _condition_path(output_dir, condition)
                condition_files.append(path)
                signature = _signature(
                    {
                        **base_signature,
                        "checkpoint_sha256": denoiser_spec.checkpoint_sha256,
                        "sigma": denoiser_spec.sigma,
                        "vector": condition.vector.slug,
                        "alpha": condition.alpha,
                    }
                )
                existing = _load_jsonl(path) if bool(config.resume) else []
                if len(existing) == len(prompts) and all(
                    row.get("signature") == signature for row in existing
                ):
                    progress.update(len(existing))
                    continue
                partial = path.with_suffix(".partial.jsonl")
                partial_rows = _load_jsonl(partial) if bool(config.resume) else []
                if len(partial_rows) > len(prompts) or not all(
                    row.get("signature") == signature
                    and int(row.get("sample_index", -1)) == index
                    for index, row in enumerate(partial_rows)
                ):
                    partial_rows = []
                rows = list(partial_rows)
                progress.update(len(rows))
                partial.parent.mkdir(parents=True, exist_ok=True)
                with partial.open("a" if rows else "w", encoding="utf-8") as stream:
                    for sample_index in range(len(rows), len(prompts)):
                        prompt = prompts[sample_index]
                        generation_seed = seed + sample_index
                        dropout_seed = seed + 1_000_000 + sample_index
                        continuation = generate_epistemic_continuation(
                            model,
                            tokenizer,
                            prompt.prompt_token_ids,
                            condition.vector.vector,
                            denoiser,
                            alpha=condition.alpha,
                            layer_index=int(config.model.layer_index),
                            layer_path=config.model.layer_path,
                            mc_samples=mc_samples,
                            max_new_tokens=new_tokens,
                            temperature=float(config.generation.temperature),
                            top_p=float(config.generation.top_p),
                            generation_seed=generation_seed,
                            dropout_seed=dropout_seed,
                            stop_on_eos=bool(config.generation.stop_on_eos),
                        )
                        token_statistics = [
                            {
                                **token,
                                "token_text": _decode_token(
                                    tokenizer, int(token["token_id"])
                                ),
                            }
                            for token in continuation.token_statistics
                        ]
                        row = {
                            "signature": signature,
                            "condition_id": condition.condition_id,
                            "sigma": denoiser_spec.sigma,
                            "denoiser_checkpoint": str(denoiser_spec.checkpoint),
                            "denoiser_sha256": denoiser_spec.checkpoint_sha256,
                            "vector_name": condition.vector.name,
                            "vector_slug": condition.vector.slug,
                            "target_dataset_label": condition.vector.dataset_label,
                            "alpha": condition.alpha,
                            "mc_samples": mc_samples,
                            "sample_index": sample_index,
                            "sample_id": prompt.sample_id,
                            "source_label": prompt.source_label,
                            "source_topic": prompt.source_topic,
                            "generation_seed": generation_seed,
                            "dropout_seed": dropout_seed,
                            "prompt_text": continuation.prompt_text,
                            "prompt_token_ids": prompt.prompt_token_ids,
                            "generated_text": continuation.generated_text,
                            "generated_token_ids": continuation.generated_token_ids,
                            "full_text": continuation.full_text,
                            "forward_calls": continuation.forward_calls,
                            "token_statistics": token_statistics,
                        }
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        rows.append(row)
                        progress.update(1)
                os.replace(partial, path)
            _release_model(denoiser)
    finally:
        progress.close()
    _release_model(model)

    all_rows = [row for path in condition_files for row in _load_jsonl(path)]
    summaries = summarize_token_metrics(all_rows)
    (output_dir / "condition_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_summary_csv(summaries, output_dir / "condition_summary.csv")
    plot_paths = plot_epistemic_summaries(
        summaries,
        output_dir / "plots",
        formats=[str(value) for value in config.plot.formats],
        dpi=int(config.plot.dpi),
    )
    example_rows = _select_examples(
        all_rows, examples_per_condition=int(config.examples.per_condition)
    )
    report_metadata = {
        "source_model": str(config.model.name),
        "layer_index": int(config.model.layer_index),
        "model_dtype": str(config.model.dtype),
        "mc_samples": mc_samples,
        "score_definition": "delta_i = D(z)_i - z = sigma^2 * grad log p_sigma(z)",
        "coordinate_systems": {
            "score_and_prediction_metrics": "feature-wise normalized hidden space",
            "denoiser_error_and_projection": "raw GPT-2 hidden space",
        },
        "prediction_used_for_generation": "mean of MC-dropout predictions",
        "diagnostics_version": _DIAGNOSTICS_VERSION,
        "metrics": dict(METRIC_LABELS),
        "resolved_config": config_to_dict(config),
    }
    report_rows = [
        {
            **row,
            "metadata": {
                key: value for key, value in row.items() if key != "token_statistics"
            },
        }
        for row in example_rows
    ]
    report_path = write_epistemic_examples_html(
        report_rows,
        output_dir / "examples.html",
        metadata=report_metadata,
    )
    manifest = {
        "format_version": 2,
        "conditions": len(conditions),
        "generations": len(all_rows),
        "token_statistics": sum(len(row["token_statistics"]) for row in all_rows),
        "condition_summary_json": "condition_summary.json",
        "condition_summary_csv": "condition_summary.csv",
        "plots": [str(path.relative_to(output_dir)) for path in plot_paths],
        "examples_html": str(report_path.relative_to(output_dir)),
        "metadata": report_metadata,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "output_dir": str(output_dir),
        "conditions": len(conditions),
        "generations": len(all_rows),
        "token_statistics": manifest["token_statistics"],
        "plots": len(plot_paths),
        "examples": len(example_rows),
    }
