from __future__ import annotations

import math
from collections.abc import Sequence

import torch


@torch.no_grad()
def denoising_metrics(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    *,
    noise_std: float,
) -> dict[str, float]:
    if not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be finite and positive for score metrics")
    clean = clean.float()
    noisy = noisy.float()
    denoised = denoised.float()
    noisy_l2 = torch.mean((noisy - clean) ** 2)
    denoised_l2 = torch.mean((denoised - clean) ** 2)
    denoising_displacement_l2 = torch.mean((denoised - noisy) ** 2)
    noise_variance = noise_std**2
    score_mse = denoising_displacement_l2 / noise_variance**2
    flat_clean = clean.reshape(clean.shape[0], -1)
    flat_noisy = noisy.reshape(noisy.shape[0], -1)
    flat_denoised = denoised.reshape(denoised.shape[0], -1)
    noisy_cosine_distance = (
        1 - torch.nn.functional.cosine_similarity(flat_clean, flat_noisy).mean()
    )
    denoised_cosine_distance = (
        1 - torch.nn.functional.cosine_similarity(flat_clean, flat_denoised).mean()
    )
    return {
        "l2": float(denoised_l2.item()),
        "rmse": float(denoised_l2.sqrt().item()),
        "cosine_distance": float(denoised_cosine_distance.item()),
        "noisy_l2": float(noisy_l2.item()),
        "noisy_rmse": float(noisy_l2.sqrt().item()),
        "noisy_cosine_distance": float(noisy_cosine_distance.item()),
        "score_mse": float(score_mse.item()),
        "score_rms": float(score_mse.sqrt().item()),
    }


def generation_metrics(rows: Sequence[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {
            "samples": 0.0,
            "mean_generated_tokens": 0.0,
            "mean_normalized_entropy": 0.0,
            "intervention_rate": 0.0,
        }
    token_counts = [len(row.get("token_ids", [])) for row in rows]
    all_entropies = [
        float(value)
        for row in rows
        for value in row.get("normalized_entropies", [])  # type: ignore[union-attr]
    ]
    interventions = sum(int(row.get("intervention_steps", 0)) for row in rows)
    forwards = sum(int(row.get("forward_calls", 0)) for row in rows)
    return {
        "samples": float(len(rows)),
        "mean_generated_tokens": sum(token_counts) / len(token_counts),
        "mean_normalized_entropy": (
            sum(all_entropies) / len(all_entropies) if all_entropies else 0.0
        ),
        "intervention_rate": interventions / max(forwards, 1),
    }
