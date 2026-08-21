from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComparisonRow:
    name: str
    latent_dim: int
    num_layers: int
    sigma: float
    best_step: int
    l2: float
    rmse: float
    cosine_distance: float
    noisy_l2: float
    noisy_rmse: float
    noisy_cosine_distance: float
    score_mse: float | None
    score_rms: float | None
    summary_path: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def collect_comparison_rows(root: str | Path) -> list[ComparisonRow]:
    """Recursively load per-model ``summary.json`` files."""

    root = Path(root).expanduser().resolve()
    rows: list[ComparisonRow] = []
    for path in sorted(root.rglob("summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        parameters = payload.get("parameters", {})
        metrics = payload.get("best_validation", {})
        required_metrics = {
            "l2",
            "rmse",
            "cosine_distance",
            "noisy_l2",
            "noisy_rmse",
            "noisy_cosine_distance",
        }
        if not required_metrics.issubset(metrics):
            continue
        rows.append(
            ComparisonRow(
                name=str(payload["name"]),
                latent_dim=int(parameters["latent_dim"]),
                num_layers=int(parameters["num_layers"]),
                sigma=float(parameters["sigma"]),
                best_step=int(payload["best_step"]),
                l2=float(metrics["l2"]),
                rmse=float(metrics["rmse"]),
                cosine_distance=float(metrics["cosine_distance"]),
                noisy_l2=float(metrics["noisy_l2"]),
                noisy_rmse=float(metrics["noisy_rmse"]),
                noisy_cosine_distance=float(metrics["noisy_cosine_distance"]),
                score_mse=(
                    float(metrics["score_mse"]) if "score_mse" in metrics else None
                ),
                score_rms=(
                    float(metrics["score_rms"]) if "score_rms" in metrics else None
                ),
                summary_path=str(path),
            )
        )
    if not rows:
        raise FileNotFoundError(f"no completed denoiser summaries found under {root}")
    return sorted(
        rows, key=lambda row: (row.l2, row.sigma, row.num_layers, row.latent_dim)
    )


def write_comparison(root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Write tables plus aggregate and per-sigma validation barplots."""

    rows = collect_comparison_rows(root)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "denoiser_comparison.csv"
    markdown_path = output_dir / "denoiser_comparison.md"
    plot_path = output_dir / "denoiser_comparison.png"

    fieldnames = list(rows[0].as_dict())
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)

    markdown_lines = [
        "| model | latent_dim | layers | sigma | best step | L2 | RMSE | cosine distance | score RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        score_rms = f"{row.score_rms:.8g}" if row.score_rms is not None else "n/a"
        markdown_lines.append(
            f"| {row.name} | {row.latent_dim} | {row.num_layers} | "
            f"{row.sigma:g} | {row.best_step} | {row.l2:.8g} | "
            f"{row.rmse:.8g} | {row.cosine_distance:.8g} | {score_rms} |"
        )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    _write_barplot(rows, plot_path, title="Best denoiser validation metrics")
    plots_by_sigma: dict[str, str] = {}
    for sigma in sorted({row.sigma for row in rows}):
        sigma_key = format(sigma, ".8g")
        sigma_plot_path = (
            output_dir / f"denoiser_comparison_sigma_{_sigma_slug(sigma)}.png"
        )
        sigma_rows = [row for row in rows if row.sigma == sigma]
        _write_barplot(
            sigma_rows,
            sigma_plot_path,
            title=f"Best denoiser validation metrics (sigma={sigma:g})",
            include_sigma_in_labels=False,
        )
        plots_by_sigma[sigma_key] = str(sigma_plot_path)
    return {
        "models": len(rows),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "plot": str(plot_path),
        "plots_by_sigma": plots_by_sigma,
        "score_models": sum(row.score_rms is not None for row in rows),
    }


def _sigma_slug(sigma: float) -> str:
    return format(sigma, ".8g").replace("-", "m").replace(".", "p")


def _write_barplot(
    rows: list[ComparisonRow],
    path: Path,
    *,
    title: str,
    include_sigma_in_labels: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    for row in rows:
        label = f"L={row.latent_dim}, blocks={row.num_layers}"
        if include_sigma_in_labels:
            label += f", σ={row.sigma:g}"
        labels.append(label)
    colormap = plt.get_cmap("tab10")
    color_by_sigma = {
        sigma: colormap(index % 10)
        for index, sigma in enumerate(sorted({row.sigma for row in rows}))
    }
    colors = [color_by_sigma[row.sigma] for row in rows]
    figure_height = max(8.0, len(rows) * 0.42)
    panels = [
        (
            "Validation L2",
            [row.l2 for row in rows],
            [row.noisy_l2 for row in rows],
            "lower is better (log scale)",
        ),
        (
            "Validation RMSE",
            [row.rmse for row in rows],
            [row.noisy_rmse for row in rows],
            "lower is better (log scale)",
        ),
        (
            "Validation cosine distance",
            [row.cosine_distance for row in rows],
            [row.noisy_cosine_distance for row in rows],
            "lower is better (log scale)",
        ),
    ]
    if all(row.score_rms is not None for row in rows):
        panels.append(
            (
                "Estimated score RMS",
                [float(row.score_rms) for row in rows],
                [0.0 for _row in rows],
                "RMS magnitude (log scale)",
            )
        )
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(6.5 * len(panels), figure_height),
        sharey=True,
    )
    positions = list(range(len(rows)))
    for axis, (panel_title, values, identity_values, axis_label) in zip(
        axes, panels, strict=True
    ):
        if any(value < 0 for value in values + identity_values):
            raise ValueError(f"{panel_title} cannot contain negative values")
        positive_values = [value for value in values + identity_values if value > 0]
        log_floor = min(positive_values) * 0.1 if positive_values else 1e-12
        plotted_values = [value if value > 0 else log_floor for value in values]
        plotted_identity = [
            value if value > 0 else log_floor for value in identity_values
        ]
        axis.barh(positions, plotted_values, color=colors)
        identity_label = "Identity f(y)=y"
        if any(value == 0 for value in identity_values):
            identity_label += ": 0 (shown at log floor)"
        axis.vlines(
            plotted_identity,
            [position - 0.36 for position in positions],
            [position + 0.36 for position in positions],
            color="black",
            linewidth=1.0,
            zorder=4,
            label=identity_label,
        )
        axis.set_title(panel_title)
        axis.set_xlabel(axis_label)
        axis.set_xscale("log")
        axis.set_axisbelow(True)
        axis.grid(axis="x", which="both", alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
