from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


METRIC_LABELS: Mapping[str, str] = {
    "score_mean_deviation": "Score dispersion: mean ||δᵢ − Eδ||₂",
    "score_length_variance": "Variance of ||δᵢ||₂",
    "score_cosine_distance_variance": "Variance of cosine distance to Eδ",
    "prediction_mean_deviation": "Prediction dispersion: mean ||Dᵢ − ED||₂",
}


def mc_dropout_statistics(
    normalized_input: torch.Tensor,
    normalized_predictions: torch.Tensor,
) -> dict[str, float]:
    """Summarize MC predictions and ``sigma² score = D(z) - z`` samples."""

    if normalized_input.ndim != 1:
        raise ValueError("normalized_input must have shape [hidden_size]")
    if (
        normalized_predictions.ndim != 2
        or normalized_predictions.shape[1] != normalized_input.numel()
    ):
        raise ValueError(
            "normalized_predictions must have shape [samples, hidden_size]"
        )
    if len(normalized_predictions) < 2:
        raise ValueError("at least two MC-dropout predictions are required")
    if not torch.isfinite(normalized_predictions).all():
        raise ValueError("MC-dropout predictions contain non-finite values")

    score_samples = normalized_predictions - normalized_input.unsqueeze(0)
    mean_score = score_samples.mean(dim=0)
    score_deviations = torch.linalg.vector_norm(
        score_samples - mean_score.unsqueeze(0), dim=-1
    )
    score_lengths = torch.linalg.vector_norm(score_samples, dim=-1)
    cosine_distances = 1 - functional.cosine_similarity(
        score_samples,
        mean_score.unsqueeze(0).expand_as(score_samples),
        dim=-1,
        eps=1e-8,
    )

    mean_prediction = normalized_predictions.mean(dim=0)
    prediction_deviations = torch.linalg.vector_norm(
        normalized_predictions - mean_prediction.unsqueeze(0), dim=-1
    )
    return {
        "score_mean_deviation": float(score_deviations.mean()),
        "score_length_variance": float(score_lengths.var(correction=0)),
        "score_cosine_distance_variance": float(cosine_distances.var(correction=0)),
        "prediction_mean_deviation": float(prediction_deviations.mean()),
    }


def summarize_token_metrics(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate token metrics for every sigma/vector/alpha condition."""

    groups: dict[tuple[float, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["sigma"]), str(row["vector_slug"]), float(row["alpha"]))
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (sigma, vector_slug, alpha), group in sorted(groups.items()):
        first = group[0]
        token_metrics = [token for row in group for token in row["token_statistics"]]
        if not token_metrics:
            raise ValueError("epistemic condition contains no token statistics")
        summary: dict[str, Any] = {
            "condition_id": first["condition_id"],
            "sigma": sigma,
            "vector_name": first["vector_name"],
            "vector_slug": vector_slug,
            "alpha": alpha,
            "generations": len(group),
            "tokens": len(token_metrics),
        }
        for metric in METRIC_LABELS:
            values = np.asarray(
                [float(token[metric]) for token in token_metrics], dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite values in epistemic metric {metric}")
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std())
            summary[f"{metric}_median"] = float(np.median(values))
            summary[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            summary[f"{metric}_q75"] = float(np.quantile(values, 0.75))
        summaries.append(summary)
    return summaries
