from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402


def plot_benchmark_series(
    rows: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Plot one alpha-colored probability/Dist-N series per vector and method."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["vector_slug"]))
        groups.setdefault(key, []).append(row)
    paths: list[Path] = []
    for (method, vector_slug), group in groups.items():
        ordered = sorted(group, key=lambda row: float(row["alpha"]))
        alphas = np.asarray([float(row["alpha"]) for row in ordered])
        vmin, vmax = float(alphas.min()), float(alphas.max())
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.colormaps["viridis"]
        fig, axis = plt.subplots(figsize=(8.5, 6.5))
        for row in ordered:
            alpha = float(row["alpha"])
            x = float(row["target_probability_mean"])
            y = float(row["distinct_n_mean"])
            x_low = float(row["target_probability_ci_low"])
            x_high = float(row["target_probability_ci_high"])
            y_low = float(row["distinct_n_ci_low"])
            y_high = float(row["distinct_n_ci_high"])
            color = cmap(norm(alpha))
            width, height = x_high - x_low, y_high - y_low
            if width > 0 and height > 0:
                axis.add_patch(
                    Ellipse(
                        (x, y),
                        width=width,
                        height=height,
                        facecolor=color,
                        edgecolor="none",
                        alpha=0.16,
                        zorder=1,
                    )
                )
            axis.errorbar(
                x,
                y,
                xerr=[[x - x_low], [x_high - x]],
                yerr=[[y - y_low], [y_high - y]],
                fmt="none",
                ecolor=color,
                elinewidth=1.4,
                capsize=3,
                alpha=0.8,
                zorder=2,
            )
            axis.scatter(
                [x],
                [y],
                s=85,
                color=[color],
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
            )
        vector_name = str(ordered[0]["vector_name"])
        confidence = float(ordered[0]["confidence"])
        distinct_order = int(ordered[0]["distinct_n_order"])
        axis.set_title(f"{vector_name} steering · {method}")
        axis.set_xlabel("Frozen AG News classifier: target-class probability")
        axis.set_ylabel(f"Distinct-{distinct_order}")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.22)
        scalar = ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        colorbar = fig.colorbar(scalar, ax=axis, pad=0.02)
        colorbar.set_label("Steering strength α")
        fig.text(
            0.01,
            0.01,
            f"Ellipses/error bars: {confidence:.0%} bootstrap CI of the mean",
            fontsize=9,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        for extension in formats:
            path = output_dir / f"{method}__{vector_slug}.{extension}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
    return paths
