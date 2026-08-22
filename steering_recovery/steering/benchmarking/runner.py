from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
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
    is_gpt2_small_model,
    resolve_device,
    resolve_dtype,
    resolve_model_dtype,
    seed_everything,
)
from steering_recovery.steering.benchmarking.data import (
    BenchmarkPrompt,
    load_ag_news_prompts,
    select_examples,
)
from steering_recovery.steering.benchmarking.generation import (
    generate_steered_continuation,
)
from steering_recovery.steering.benchmarking.reporting import write_examples_html
from steering_recovery.steering.benchmarking.scoring import (
    CausalLMSLORScorer,
    FrozenAGNewsClassifier,
    distinct_n,
    estimate_token_unigram_log_probabilities,
)
from steering_recovery.steering.benchmarking.statistics import summarize_condition


LOGGER = logging.getLogger(__name__)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_SLOR_FORMULA = "mean(log_p_causal_lm - log_p_token_unigram)"
_SLOR_FORMULA_VERSION = 1


@dataclass(frozen=True)
class SteeringVectorSpec:
    name: str
    slug: str
    dataset_label: int
    classifier_index: int
    vector: torch.Tensor


@dataclass(frozen=True)
class SteeringMethod:
    name: str
    intervention_mode: str
    denoiser_checkpoint: str | None
    denoiser_sha256: str | None


@dataclass(frozen=True)
class Condition:
    method: SteeringMethod
    vector: SteeringVectorSpec
    alpha: float

    @property
    def alpha_slug(self) -> str:
        value = format(self.alpha, ".12g").replace("-", "m").replace(".", "p")
        return f"alpha_{value}"

    @property
    def condition_id(self) -> str:
        return f"{self.method.name}__{self.vector.slug}__{self.alpha_slug}"


def validate_gpt2_small_vector_precision(
    payload: Mapping[str, Any], *, source_model_name: str
) -> None:
    """Reject steering vectors extracted from reduced-precision GPT-2 states."""

    if not is_gpt2_small_model(source_model_name):
        return
    metadata = payload.get("metadata")
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    if not isinstance(source, Mapping) or source.get("model_dtype") != "float32":
        raise ValueError(
            "GPT-2 Small benchmark requires steering vectors generated in "
            "float32; regenerate the configured vector artifact"
        )


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("steering vector artifact must be a mapping")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_steering_vectors(
    path: str | Path, class_indices: Mapping[str, int]
) -> tuple[list[SteeringVectorSpec], dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _safe_torch_load(path)
    required = {"steering_vectors", "vector_names", "vector_slugs", "group_labels"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"steering artifact is missing keys: {sorted(missing)}")
    vectors = torch.as_tensor(payload["steering_vectors"]).float()
    names = list(payload["vector_names"])
    slugs = list(payload["vector_slugs"])
    labels = [int(value) for value in payload["group_labels"]]
    if vectors.ndim != 2 or not (
        len(vectors) == len(names) == len(slugs) == len(labels)
    ):
        raise ValueError("steering vector matrix and metadata lengths differ")
    if not torch.isfinite(vectors).all():
        raise ValueError("steering vectors contain non-finite values")
    specs: list[SteeringVectorSpec] = []
    for index, (name, slug, label) in enumerate(zip(names, slugs, labels)):
        slug = str(slug)
        if not _SAFE_NAME.fullmatch(slug):
            raise ValueError(f"vector slug must be filesystem-safe: {slug!r}")
        if slug not in class_indices:
            raise KeyError(f"classifier class index is not configured for {slug!r}")
        specs.append(
            SteeringVectorSpec(
                name=str(name),
                slug=slug,
                dataset_label=label,
                classifier_index=int(class_indices[slug]),
                vector=vectors[index],
            )
        )
    return specs, payload


def parse_methods(config: Sequence[Any]) -> list[SteeringMethod]:
    methods: list[SteeringMethod] = []
    for item in config:
        name = str(item.name)
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"method name must be filesystem-safe: {name!r}")
        checkpoint = None if item.denoiser_checkpoint is None else str(item.denoiser_checkpoint)
        if checkpoint is not None and not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        intervention_mode = str(item.intervention_mode)
        if intervention_mode not in {
            "none",
            "once_at_start",
            "every_step",
            "entropy_threshold",
        }:
            raise ValueError(f"unknown intervention mode {intervention_mode!r}")
        methods.append(
            SteeringMethod(
                name=name,
                intervention_mode=intervention_mode,
                denoiser_checkpoint=checkpoint,
                denoiser_sha256=(
                    _file_sha256(Path(checkpoint)) if checkpoint is not None else None
                ),
            )
        )
    if not methods or len({method.name for method in methods}) != len(methods):
        raise ValueError("benchmark methods must have unique names")
    return methods


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


def _release_model(value: Any) -> None:
    module = getattr(value, "model", value)
    if module is not None and hasattr(module, "to"):
        module.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _condition_path(output_dir: Path, condition: Condition) -> Path:
    return (
        output_dir
        / "conditions"
        / condition.method.name
        / condition.vector.slug
        / f"{condition.alpha_slug}.jsonl"
    )


def _write_summary_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_examples_markdown(rows: Sequence[dict[str, Any]], path: Path) -> None:
    parts = ["# Steering benchmark examples", ""]
    for row in rows:
        metric_parts = [
            f"Dist-{order}: `{float(row[f'distinct_{order}']):.4f}`"
            for order in (1, 2, 3)
            if f"distinct_{order}" in row
        ]
        if "slor" in row:
            metric_parts.append(f"SLOR: `{float(row['slor']):.4f}`")
        parts.extend(
            [
                f"## {row['vector_name']} · {row['method']} · α={row['alpha']}",
                "",
                f"Source topic: `{row['source_topic']}` · sample: `{row['sample_id']}`",
                "",
                " · ".join(
                    [
                        f"Target probability: `{float(row['target_probability']):.4f}`",
                        *metric_parts,
                    ]
                ),
                "",
                "```text",
                str(row["full_text"]),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(parts), encoding="utf-8")


def _load_or_estimate_slor_unigram(
    config: Any,
    scorer: CausalLMSLORScorer,
    output_dir: Path,
    *,
    resume: bool,
) -> tuple[torch.Tensor, dict[str, Any], str]:
    unigram_config = config_to_dict(config.unigram)
    signature = _signature(
        {
            "unigram": unigram_config,
            "tokenizer_name": str(config.tokenizer_name),
            "vocab_size": scorer.vocab_size,
        }
    )
    cache_path = output_dir / "metrics" / "slor_unigram.pt"
    if resume and cache_path.is_file():
        payload = _safe_torch_load(cache_path)
        cached_probabilities = payload.get("log_probabilities")
        cached_metadata = payload.get("metadata")
        if cached_probabilities is not None and isinstance(cached_metadata, Mapping):
            probabilities = torch.as_tensor(cached_probabilities).float()
            if (
                payload.get("signature") == signature
                and probabilities.shape == (scorer.vocab_size,)
                and torch.isfinite(probabilities).all()
            ):
                return probabilities, dict(cached_metadata), signature

    from datasets import load_dataset

    dataset = load_dataset(
        str(config.unigram.dataset_name),
        config.unigram.dataset_config,
        split=str(config.unigram.split),
        streaming=bool(config.unigram.streaming),
    )
    text_column = str(config.unigram.text_column)
    max_documents_value = config.unigram.max_documents
    max_documents = (
        None if max_documents_value is None else int(max_documents_value)
    )
    estimate = estimate_token_unigram_log_probabilities(
        (
            row.get(text_column)
            for row in tqdm(dataset, desc="SLOR unigram corpus", unit="document")
        ),
        tokenizer=scorer.tokenizer,
        vocab_size=scorer.vocab_size,
        batch_size=int(config.unigram.batch_size),
        smoothing=float(config.unigram.smoothing),
        max_documents=max_documents,
    )
    metadata = {
        "dataset_name": str(config.unigram.dataset_name),
        "dataset_config": config.unigram.dataset_config,
        "split": str(config.unigram.split),
        "text_column": text_column,
        "documents": estimate.documents,
        "tokens": estimate.tokens,
        "vocab_size": scorer.vocab_size,
        "smoothing": float(config.unigram.smoothing),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    torch.save(
        {
            "signature": signature,
            "log_probabilities": estimate.log_probabilities,
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, cache_path)
    return estimate.log_probabilities, metadata, signature


def run_steering_benchmark(
    config: DictConfig, output_dir: str | Path
) -> dict[str, Any]:
    seed = int(config.seed)
    seed_everything(seed)
    output_dir = ensure_output_dir(output_dir)
    OmegaConf.save(config, output_dir / "config.yaml")
    device = resolve_device(str(config.device))
    generation_dtype = resolve_model_dtype(
        str(config.model.name), str(config.model.dtype), device
    )
    class_indices = {
        str(key): int(value)
        for key, value in config_to_dict(config.classifier.class_indices).items()
    }
    vectors, vector_payload = load_steering_vectors(
        config.steering_vectors_path, class_indices
    )
    validate_gpt2_small_vector_precision(
        vector_payload, source_model_name=str(config.model.name)
    )
    methods = parse_methods(config.methods)
    alphas = [float(value) for value in config.alphas]
    if not alphas or any(not math.isfinite(alpha) for alpha in alphas):
        raise ValueError("alphas must be a non-empty list of finite values")
    if len(set(alphas)) != len(alphas):
        raise ValueError("alphas must be unique")
    samples_per_point = int(config.generation.samples_per_point)
    prompt_tokens = int(config.generation.prompt_tokens)
    new_tokens = int(config.generation.new_tokens)
    if samples_per_point != 100:
        LOGGER.warning(
            "Configured %d generations per point; canonical benchmark uses 100",
            samples_per_point,
        )

    model, tokenizer = _load_generation_model(config.model, device, generation_dtype)
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("cannot infer generation model hidden size")
    hidden_size = int(hidden_size)
    if any(spec.vector.numel() != hidden_size for spec in vectors):
        raise ValueError("steering vector hidden size differs from generation model")
    prompt_path = output_dir / "prompts.jsonl"
    prompt_metadata_path = output_dir / "prompts_metadata.json"
    prompt_signature = _signature(
        {
            "dataset": config_to_dict(config.dataset),
            "model_tokenizer": str(config.model.tokenizer_name or config.model.name),
            "samples_per_point": samples_per_point,
            "prompt_tokens": prompt_tokens,
            "seed": seed,
        }
    )
    saved_prompt_metadata = (
        json.loads(prompt_metadata_path.read_text(encoding="utf-8"))
        if prompt_metadata_path.is_file()
        else {}
    )
    if (
        bool(config.resume)
        and prompt_path.is_file()
        and saved_prompt_metadata.get("signature") == prompt_signature
    ):
        prompts = [BenchmarkPrompt.from_dict(row) for row in _load_jsonl(prompt_path)]
        if len(prompts) != samples_per_point:
            raise ValueError("saved prompt count differs from configured samples_per_point")
    else:
        topics = {spec.dataset_label: spec.name for spec in vectors}
        prompts = load_ag_news_prompts(
            config.dataset,
            tokenizer=tokenizer,
            topics=topics,
            total_samples=samples_per_point,
            prompt_tokens=prompt_tokens,
            seed=seed,
        )
        _atomic_write_jsonl([prompt.to_dict() for prompt in prompts], prompt_path)
        prompt_metadata_path.write_text(
            json.dumps({"signature": prompt_signature}, indent=2), encoding="utf-8"
        )

    conditions = [
        Condition(method=method, vector=vector, alpha=alpha)
        for method in methods
        for vector in vectors
        for alpha in alphas
    ]
    base_signature = {
        "model": config_to_dict(config.model),
        "generation": config_to_dict(config.generation),
        "prompts": [prompt.to_dict() for prompt in prompts],
        "steering_vectors_path": str(config.steering_vectors_path),
        "steering_vectors_sha256": _file_sha256(Path(config.steering_vectors_path)),
        "vector_metadata": vector_payload.get("metadata", {}),
    }
    progress = tqdm(
        total=len(conditions) * len(prompts),
        desc="steered generations",
        unit="generation",
        dynamic_ncols=True,
    )
    condition_files: list[Path] = []
    try:
        for method in methods:
            denoiser = None
            if method.denoiser_checkpoint is not None:
                denoiser, denoiser_metadata = load_checkpoint(
                    method.denoiser_checkpoint,
                    device=device,
                    dtype=generation_dtype,
                )
                validate_gpt2_small_denoiser_precision(
                    denoiser_metadata, source_model_name=str(config.model.name)
                )
            for condition in [item for item in conditions if item.method == method]:
                path = _condition_path(output_dir, condition)
                condition_files.append(path)
                condition_signature = _signature(
                    {
                        **base_signature,
                        "method": method.__dict__,
                        "vector": condition.vector.slug,
                        "alpha": condition.alpha,
                    }
                )
                existing = _load_jsonl(path) if bool(config.resume) else []
                if len(existing) == len(prompts) and all(
                    row.get("signature") == condition_signature for row in existing
                ):
                    progress.update(len(existing))
                    continue
                rows: list[dict[str, Any]] = []
                partial = path.with_suffix(".partial.jsonl")
                partial_rows = _load_jsonl(partial) if bool(config.resume) else []
                if len(partial_rows) > len(prompts) or not all(
                    row.get("signature") == condition_signature
                    and int(row.get("sample_index", -1)) == index
                    for index, row in enumerate(partial_rows)
                ):
                    partial_rows = []
                rows.extend(partial_rows)
                progress.update(len(partial_rows))
                path.parent.mkdir(parents=True, exist_ok=True)
                mode = "a" if partial_rows else "w"
                with partial.open(mode, encoding="utf-8") as stream:
                    for sample_index in range(len(rows), len(prompts)):
                        prompt = prompts[sample_index]
                        generation_seed = seed + sample_index
                        continuation = generate_steered_continuation(
                            model,
                            tokenizer,
                            prompt.prompt_token_ids,
                            condition.vector.vector,
                            alpha=condition.alpha,
                            layer_index=int(config.model.layer_index),
                            layer_path=config.model.layer_path,
                            intervention_mode=method.intervention_mode,
                            entropy_threshold=float(config.model.entropy_threshold),
                            denoiser=denoiser,
                            max_new_tokens=new_tokens,
                            temperature=float(config.generation.temperature),
                            top_p=float(config.generation.top_p),
                            seed=generation_seed,
                            stop_on_eos=bool(config.generation.stop_on_eos),
                        )
                        row = {
                            "signature": condition_signature,
                            "condition_id": condition.condition_id,
                            "method": method.name,
                            "intervention_mode": method.intervention_mode,
                            "denoiser_checkpoint": method.denoiser_checkpoint,
                            "vector_name": condition.vector.name,
                            "vector_slug": condition.vector.slug,
                            "target_dataset_label": condition.vector.dataset_label,
                            "target_classifier_index": condition.vector.classifier_index,
                            "alpha": condition.alpha,
                            "sample_index": sample_index,
                            "sample_id": prompt.sample_id,
                            "source_label": prompt.source_label,
                            "source_topic": prompt.source_topic,
                            "seed": generation_seed,
                            "prompt_text": continuation.prompt_text,
                            "prompt_token_ids": prompt.prompt_token_ids,
                            "generated_text": continuation.generated_text,
                            "generated_token_ids": continuation.generated_token_ids,
                            "full_text": continuation.full_text,
                            "intervention_steps": continuation.intervention_steps,
                            "forward_calls": continuation.forward_calls,
                        }
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        rows.append(row)
                        progress.update(1)
                os.replace(partial, path)
            _release_model(denoiser)
            denoiser = None
    finally:
        progress.close()
    _release_model(model)
    del model

    classifier_signature = _signature(config_to_dict(config.classifier))
    classifier_needed = any(
        any(
            "target_probability" not in row
            or row.get("classifier_signature") != classifier_signature
            for row in _load_jsonl(path)
        )
        for path in condition_files
    )
    classifier_metadata: dict[str, Any] = {}
    if classifier_needed:
        metric_dtype = resolve_dtype(str(config.classifier.dtype), device)
        if metric_dtype != torch.float32:
            raise ValueError("steering benchmark classifier must use float32")
        classifier = FrozenAGNewsClassifier.from_pretrained(
            str(config.classifier.model_name),
            device=device,
            dtype=metric_dtype,
            trust_remote_code=bool(config.classifier.trust_remote_code),
        )
        classifier_metadata["id2label"] = classifier.id2label
        for path in tqdm(condition_files, desc="classifier metrics", unit="condition"):
            rows = _load_jsonl(path)
            if all(
                "target_probability" in row
                and row.get("classifier_signature") == classifier_signature
                for row in rows
            ):
                continue
            scores = classifier.score(
                [str(row["generated_text"]) for row in rows],
                [int(row["target_classifier_index"]) for row in rows],
                batch_size=int(config.classifier.batch_size),
                max_length=int(config.classifier.max_length),
            )
            for row, score in zip(rows, scores):
                row["target_probability"] = score
                row["classifier_signature"] = classifier_signature
            _atomic_write_jsonl(rows, path)
        _release_model(classifier)
        classifier = None

    distinct_orders = [int(value) for value in config.metrics.distinct_orders]
    if (
        not distinct_orders
        or any(order <= 0 for order in distinct_orders)
        or len(set(distinct_orders)) != len(distinct_orders)
    ):
        raise ValueError("metrics.distinct_orders must contain unique positive values")
    distinct_signature = _signature({"distinct_orders": distinct_orders})
    distinct_fields = [f"distinct_{order}" for order in distinct_orders]
    for path in tqdm(condition_files, desc="Dist-N metrics", unit="condition"):
        rows = _load_jsonl(path)
        changed = False
        for row in rows:
            if (
                any(field not in row for field in distinct_fields)
                or row.get("distinct_n_signature") != distinct_signature
            ):
                for order, field in zip(distinct_orders, distinct_fields):
                    row[field] = distinct_n(row["generated_token_ids"], order)
                row["distinct_n_signature"] = distinct_signature
                changed = True
            if "distinct_n" in row or "distinct_n_order" in row:
                row.pop("distinct_n", None)
                row.pop("distinct_n_order", None)
                changed = True
        if changed:
            _atomic_write_jsonl(rows, path)

    slor_config_signature = _signature(
        {
            "formula_version": _SLOR_FORMULA_VERSION,
            "config": config_to_dict(config.slor),
        }
    )
    slor_needed = any(
        any(
            "slor" not in row
            or row.get("slor_config_signature") != slor_config_signature
            for row in _load_jsonl(path)
        )
        for path in condition_files
    )
    slor_metadata: dict[str, Any] = {}
    unigram_cache_path = output_dir / "metrics" / "slor_unigram.pt"
    if slor_needed:
        slor_dtype = resolve_dtype(str(config.slor.dtype), device)
        if slor_dtype != torch.float16:
            raise ValueError(
                "steering benchmark SLOR model requires float16 inference; "
                "run it on an FP16-capable accelerator"
            )
        slor_scorer = CausalLMSLORScorer.from_pretrained(
            str(config.slor.model_name),
            tokenizer_name=str(config.slor.tokenizer_name),
            device=device,
            dtype=slor_dtype,
            trust_remote_code=bool(config.slor.trust_remote_code),
        )
        if tokenizer.get_vocab() != slor_scorer.tokenizer.get_vocab():
            raise ValueError(
                "generation and SLOR tokenizers must use the same token-to-id mapping"
            )
        unigram_log_probabilities, unigram_metadata, unigram_signature = (
            _load_or_estimate_slor_unigram(
                config.slor, slor_scorer, output_dir, resume=bool(config.resume)
            )
        )
        slor_signature = _signature(
            {
                "formula_version": _SLOR_FORMULA_VERSION,
                "slor": config_to_dict(config.slor),
                "unigram_signature": unigram_signature,
            }
        )
        slor_metadata["unigram"] = unigram_metadata
        for path in tqdm(condition_files, desc="GPT-2 Large SLOR", unit="condition"):
            rows = _load_jsonl(path)
            if all(
                "slor" in row and row.get("slor_signature") == slor_signature
                for row in rows
            ):
                continue
            scores = slor_scorer.score(
                [row["prompt_token_ids"] for row in rows],
                [row["generated_token_ids"] for row in rows],
                unigram_log_probabilities=unigram_log_probabilities,
                batch_size=int(config.slor.batch_size),
            )
            for row, score in zip(rows, scores):
                row["slor"] = score
                row["slor_signature"] = slor_signature
                row["slor_config_signature"] = slor_config_signature
            _atomic_write_jsonl(rows, path)
        _release_model(slor_scorer)
        slor_scorer = None
    elif unigram_cache_path.is_file():
        unigram_payload = _safe_torch_load(unigram_cache_path)
        if isinstance(unigram_payload.get("metadata"), Mapping):
            slor_metadata["unigram"] = dict(unigram_payload["metadata"])

    all_rows = [_load_jsonl(path) for path in condition_files]
    metric_fields = [*distinct_fields, "slor"]
    summaries = [
        summarize_condition(
            rows,
            metric_fields=metric_fields,
            confidence=float(config.statistics.confidence),
            bootstrap_resamples=int(config.statistics.bootstrap_resamples),
            seed=seed + index * 1009,
        )
        for index, rows in enumerate(all_rows)
    ]
    _atomic_write_jsonl(summaries, output_dir / "summary.jsonl")
    _write_summary_csv(summaries, output_dir / "summary.csv")
    source_labels = sorted(spec.dataset_label for spec in vectors)
    example_rows = [
        row
        for rows in all_rows
        for row in select_examples(
            rows,
            source_labels=source_labels,
            examples_per_source_topic=int(config.examples.per_source_topic),
        )
    ]
    _atomic_write_jsonl(example_rows, output_dir / "examples.jsonl")
    _write_examples_markdown(example_rows, output_dir / "examples.md")
    write_examples_html(example_rows, output_dir / "examples.html")
    from steering_recovery.steering.benchmarking.plotting import (
        plot_benchmark_series,
    )

    plot_paths = plot_benchmark_series(
        summaries,
        output_dir / "plots",
        distinct_orders=distinct_orders,
        slor_model_name=str(config.slor.model_name),
        formats=[str(value) for value in config.plot.formats],
        dpi=int(config.plot.dpi),
    )
    result = {
        "format_version": 2,
        "output_dir": str(output_dir),
        "conditions": len(conditions),
        "generations_per_condition": len(prompts),
        "total_generations": len(conditions) * len(prompts),
        "vectors": [spec.slug for spec in vectors],
        "methods": [method.name for method in methods],
        "alphas": alphas,
        "classifier": {
            "model_name": str(config.classifier.model_name),
            "class_indices": class_indices,
            **classifier_metadata,
        },
        "distinct_orders": distinct_orders,
        "slor": {
            "model_name": str(config.slor.model_name),
            "tokenizer_name": str(config.slor.tokenizer_name),
            "dtype": str(config.slor.dtype),
            "formula": _SLOR_FORMULA,
            "formula_version": _SLOR_FORMULA_VERSION,
            "scope": "generated_tokens_conditioned_on_prompt",
            **slor_metadata,
        },
        "summary_files": ["summary.jsonl", "summary.csv"],
        "example_files": ["examples.jsonl", "examples.md", "examples.html"],
        "plots": [str(path.relative_to(output_dir)) for path in plot_paths],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
