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
    "score_pairwise_cosine_distance": "Mean pairwise cosine distance between δᵢ",
    "score_inverse_snr": "Inverse score SNR: mean ||δᵢ − Eδ||₂ / ||Eδ||₂",
    "prediction_mean_deviation": "Prediction dispersion: mean ||Dᵢ − ED||₂",
    "denoiser_l2_error": "Denoiser error: ||ED(h + αv) − h||₂",
    "steering_projection_removal": "Steering projection change after denoising",
}

_COSINE_EPS = 1e-8
_SNR_EPS = 1e-8


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
        eps=_COSINE_EPS,
    )
    score_directions = functional.normalize(
        score_samples, p=2, dim=-1, eps=_COSINE_EPS
    )
    pair_count = len(score_samples) * (len(score_samples) - 1) / 2
    pairwise_similarity_sum = (
        torch.linalg.vector_norm(score_directions.sum(dim=0)).square()
        - score_directions.square().sum()
    ) / 2
    pairwise_cosine_distance = (
        1 - pairwise_similarity_sum / pair_count
    ).clamp(0, 2)
    score_mean_deviation = score_deviations.mean()
    inverse_snr = score_mean_deviation / torch.linalg.vector_norm(mean_score).clamp_min(
        _SNR_EPS
    )

    mean_prediction = normalized_predictions.mean(dim=0)
    prediction_deviations = torch.linalg.vector_norm(
        normalized_predictions - mean_prediction.unsqueeze(0), dim=-1
    )
    return {
        "score_mean_deviation": float(score_mean_deviation),
        "score_length_variance": float(score_lengths.var(correction=0)),
        "score_cosine_distance_variance": float(cosine_distances.var(correction=0)),
        "score_pairwise_cosine_distance": float(pairwise_cosine_distance),
        "score_inverse_snr": float(inverse_snr),
        "prediction_mean_deviation": float(prediction_deviations.mean()),
    }


def denoising_geometry_statistics(
    original_hidden: torch.Tensor,
    steered_hidden: torch.Tensor,
    recovered_hidden: torch.Tensor,
    steering_vector: torch.Tensor,
) -> dict[str, float]:
    """Measure denoising error and steering-axis change in raw GPT coordinates."""

    tensors = [
        torch.as_tensor(value).flatten()
        for value in (
            original_hidden,
            steered_hidden,
            recovered_hidden,
            steering_vector,
        )
    ]
    if not all(tensor.shape == tensors[0].shape for tensor in tensors):
        raise ValueError("hidden states and steering_vector must have equal shapes")
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("denoising geometry contains non-finite values")
    original, steered, recovered, vector = tensors
    vector_norm = torch.linalg.vector_norm(vector)
    if float(vector_norm) == 0:
        raise ValueError("steering_vector must have non-zero norm")
    direction = vector / vector_norm
    before_scalar = torch.dot(steered - original, direction)
    after_scalar = torch.dot(recovered - original, direction)
    before_projection = before_scalar * direction
    after_projection = after_scalar * direction
    return {
        "denoiser_l2_error": float(torch.linalg.vector_norm(recovered - original)),
        "steering_projection_before": float(
            torch.linalg.vector_norm(before_projection)
        ),
        "steering_projection_after": float(
            torch.linalg.vector_norm(after_projection)
        ),
        "steering_projection_removal": float(
            torch.linalg.vector_norm(before_projection - after_projection)
        ),
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
