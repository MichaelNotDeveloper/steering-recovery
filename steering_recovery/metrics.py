from __future__ import annotations

from collections.abc import Sequence

import torch


@torch.no_grad()
def denoising_metrics(
    clean: torch.Tensor, noisy: torch.Tensor, denoised: torch.Tensor
) -> dict[str, float]:
    clean = clean.float()
    noisy = noisy.float()
    denoised = denoised.float()
    noisy_mse = torch.mean((noisy - clean) ** 2)
    denoised_mse = torch.mean((denoised - clean) ** 2)
    flat_clean = clean.reshape(clean.shape[0], -1)
    flat_denoised = denoised.reshape(denoised.shape[0], -1)
    cosine = torch.nn.functional.cosine_similarity(flat_clean, flat_denoised).mean()
    improvement = 1 - denoised_mse / noisy_mse.clamp_min(1e-12)
    return {
        "noisy_mse": float(noisy_mse.item()),
        "denoised_mse": float(denoised_mse.item()),
        "relative_mse_improvement": float(improvement.item()),
        "cosine_similarity": float(cosine.item()),
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
