from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from steering_recovery.steering.epistemic.statistics import METRIC_LABELS


def _shared_y_limits(
    summaries: Sequence[dict[str, Any]], metric: str
) -> tuple[float, float]:
    """Return padded limits containing every mean and interquartile band."""

    values = np.asarray(
        [
            float(row[f"{metric}_{statistic}"])
            for row in summaries
            for statistic in ("mean", "q25", "q75")
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"cannot plot non-finite {metric} values")
    minimum = float(values.min())
    maximum = float(values.max())
    scale = max(maximum - minimum, abs(minimum), abs(maximum), 1e-12)
    padding = 0.06 * scale
    lower = 0.0 if minimum >= 0 else minimum - padding
    upper = maximum + padding
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def plot_epistemic_summaries(
    summaries: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Plot vector/alpha epistemic statistics in one panel per noise level."""

    if not summaries:
        raise ValueError("cannot plot empty epistemic summaries")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sigmas = sorted({float(row["sigma"]) for row in summaries})
    vector_slugs = list(dict.fromkeys(str(row["vector_slug"]) for row in summaries))
    colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, max(4, len(vector_slugs))))
    paths: list[Path] = []
    for metric, label in METRIC_LABELS.items():
        y_limits = _shared_y_limits(summaries, metric)
        figure, axes = plt.subplots(
            1,
            len(sigmas),
            figsize=(6.2 * len(sigmas), 5.2),
            squeeze=False,
            sharey=True,
        )
        for sigma_index, sigma in enumerate(sigmas):
            axis = axes[0, sigma_index]
            for vector_index, vector_slug in enumerate(vector_slugs):
                group = sorted(
                    (
                        row
                        for row in summaries
                        if float(row["sigma"]) == sigma
                        and str(row["vector_slug"]) == vector_slug
                    ),
                    key=lambda row: float(row["alpha"]),
                )
                if not group:
                    continue
                alphas = np.asarray([float(row["alpha"]) for row in group])
                means = np.asarray([float(row[f"{metric}_mean"]) for row in group])
                lower = np.asarray([float(row[f"{metric}_q25"]) for row in group])
                upper = np.asarray([float(row[f"{metric}_q75"]) for row in group])
                color = colors[vector_index]
                axis.plot(
                    alphas,
                    means,
                    marker="o",
                    linewidth=2,
                    color=color,
                    label=str(group[0]["vector_name"]),
                )
                axis.fill_between(alphas, lower, upper, color=color, alpha=0.14)
            axis.set_title(f"Denoiser σ = {sigma:g}")
            axis.set_xlabel("Steering strength α")
            axis.set_xticks(sorted({float(row["alpha"]) for row in summaries}))
            axis.grid(True, alpha=0.22)
        axes[0, 0].set_ylim(*y_limits)
        axes[0, 0].set_ylabel(label)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=max(1, len(labels)),
            frameon=False,
        )
        figure.suptitle(f"MC-dropout steering · {label}", y=0.995)
        figure.tight_layout(rect=(0, 0, 1, 0.86))
        for extension in formats:
            path = output_dir / f"{metric}.{extension}"
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            paths.append(path)
        plt.close(figure)
    return paths
