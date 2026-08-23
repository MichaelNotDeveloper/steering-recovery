from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return mean and a percentile bootstrap confidence interval."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    if not np.isfinite(array).all():
        raise ValueError("bootstrap values must be finite")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(resamples, len(array)))
    means = array[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(array.mean()), float(low), float(high)


def summarize_condition(
    rows: Sequence[dict[str, object]],
    *,
    metric_fields: Sequence[str],
    confidence: float,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize an empty condition")
    first = rows[0]
    stable_fields = (
        "condition_id",
        "method",
        "intervention_mode",
        "denoiser_name",
        "denoiser_checkpoint",
        "denoiser_sigma",
        "denoiser_dropout",
        "recovery_name",
        "denoising_mode",
        "beta",
        "vector_name",
        "vector_slug",
        "target_dataset_label",
        "target_classifier_index",
        "alpha",
    )
    for field in stable_fields:
        if any(row.get(field) != first.get(field) for row in rows):
            raise ValueError(f"condition rows disagree on {field}")
    probability = bootstrap_mean_interval(
        [float(row["target_probability"]) for row in rows],
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    if not metric_fields or len(set(metric_fields)) != len(metric_fields):
        raise ValueError("metric_fields must contain unique values")
    result = {
        **{field: first.get(field) for field in stable_fields},
        "samples": len(rows),
        "confidence": confidence,
        "target_probability_mean": probability[0],
        "target_probability_ci_low": probability[1],
        "target_probability_ci_high": probability[2],
        "mean_generated_tokens": float(
            np.mean([len(row["generated_token_ids"]) for row in rows])
        ),
    }
    for offset, field in enumerate(metric_fields, start=1):
        interval = bootstrap_mean_interval(
            [float(row[field]) for row in rows],
            confidence=confidence,
            resamples=bootstrap_resamples,
            seed=seed + offset,
        )
        result[f"{field}_mean"] = interval[0]
        result[f"{field}_ci_low"] = interval[1]
        result[f"{field}_ci_high"] = interval[2]
    return result
