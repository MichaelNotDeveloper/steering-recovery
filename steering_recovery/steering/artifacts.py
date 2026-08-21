from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from steering_recovery.runtime import ensure_output_dir
from steering_recovery.steering.core import (
    ContrastResult,
    HiddenMoments,
    Label,
    TopicDefinition,
)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def save_steering_artifacts(
    output_dir: str | Path,
    *,
    topics: Sequence[TopicDefinition],
    moments: Mapping[Label, HiddenMoments],
    contrasts: Sequence[ContrastResult],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Save directly loadable vectors plus a combined, self-describing artifact."""

    output_dir = ensure_output_dir(output_dir)
    if not contrasts:
        raise ValueError("at least one steering contrast is required")
    hidden_size = int(contrasts[0].steering_vector.numel())
    vector_entries: list[dict[str, Any]] = []
    for result in contrasts:
        if result.steering_vector.shape != (hidden_size,):
            raise ValueError("all steering vectors must have the same hidden size")
        filename = f"{result.definition.slug}.pt"
        payload = {
            "format_version": 1,
            "steering_vector": result.steering_vector,
            "name": result.definition.name,
            "slug": result.definition.slug,
            "positive_labels": list(result.definition.positive_labels),
            "negative_labels": list(result.definition.negative_labels),
            "positive_mean": result.positive_mean,
            "negative_mean": result.negative_mean,
            "positive_count": result.positive_count,
            "negative_count": result.negative_count,
            "metadata": dict(metadata),
        }
        _atomic_torch_save(payload, output_dir / filename)
        vector_entries.append(
            {
                "name": result.definition.name,
                "slug": result.definition.slug,
                "file": filename,
                "positive_labels": list(result.definition.positive_labels),
                "negative_labels": list(result.definition.negative_labels),
                "positive_count": result.positive_count,
                "negative_count": result.negative_count,
                "l2_norm": float(torch.linalg.vector_norm(result.steering_vector)),
            }
        )

    combined_payload = {
        "format_version": 1,
        "steering_vectors": torch.stack(
            [result.steering_vector for result in contrasts]
        ),
        "vector_names": [result.definition.name for result in contrasts],
        "vector_slugs": [result.definition.slug for result in contrasts],
        "contrasts": vector_entries,
        "group_labels": [topic.label for topic in topics],
        "group_names": [topic.name for topic in topics],
        "group_means": torch.stack([moments[topic.label].mean.float() for topic in topics]),
        "group_variances": torch.stack(
            [moments[topic.label].variance.float() for topic in topics]
        ),
        "group_counts": [moments[topic.label].count for topic in topics],
        "metadata": dict(metadata),
    }
    combined_filename = "steering_vectors.pt"
    _atomic_torch_save(combined_payload, output_dir / combined_filename)
    manifest = {
        "format_version": 1,
        "method": "difference_of_means",
        "formula": "mean(positive) - mean(negative)",
        "hidden_size": hidden_size,
        "combined_file": combined_filename,
        "vectors": vector_entries,
        "metadata": dict(metadata),
    }
    _atomic_json_save(manifest, output_dir / "manifest.json")
    return manifest
