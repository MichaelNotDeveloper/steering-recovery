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
        "denoiser_checkpoint",
        "vector_name",
        "vector_slug",
        "target_dataset_label",
        "target_classifier_index",
        "alpha",
        "distinct_n_order",
    )
    for field in stable_fields:
        if any(row[field] != first[field] for row in rows):
            raise ValueError(f"condition rows disagree on {field}")
    probability = bootstrap_mean_interval(
        [float(row["target_probability"]) for row in rows],
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    diversity = bootstrap_mean_interval(
        [float(row["distinct_n"]) for row in rows],
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=seed + 1,
    )
    return {
        **{field: first[field] for field in stable_fields},
        "samples": len(rows),
        "confidence": confidence,
        "target_probability_mean": probability[0],
        "target_probability_ci_low": probability[1],
        "target_probability_ci_high": probability[2],
        "distinct_n_mean": diversity[0],
        "distinct_n_ci_low": diversity[1],
        "distinct_n_ci_high": diversity[2],
        "mean_generated_tokens": float(
            np.mean([len(row["generated_token_ids"]) for row in rows])
        ),
    }
