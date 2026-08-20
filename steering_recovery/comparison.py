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
                summary_path=str(path),
            )
        )
    if not rows:
        raise FileNotFoundError(f"no completed denoiser summaries found under {root}")
    return sorted(
        rows, key=lambda row: (row.l2, row.sigma, row.num_layers, row.latent_dim)
    )


def write_comparison(root: str | Path, output_dir: str | Path) -> dict[str, str | int]:
    """Write CSV, Markdown table and a three-panel validation barplot."""

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
        "| model | latent_dim | layers | sigma | best step | L2 | RMSE | cosine distance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown_lines.append(
            f"| {row.name} | {row.latent_dim} | {row.num_layers} | "
            f"{row.sigma:g} | {row.best_step} | {row.l2:.8g} | "
            f"{row.rmse:.8g} | {row.cosine_distance:.8g} |"
        )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    _write_barplot(rows, plot_path)
    return {
        "models": len(rows),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "plot": str(plot_path),
    }


def _write_barplot(rows: list[ComparisonRow], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        f"L={row.latent_dim}, blocks={row.num_layers}, σ={row.sigma:g}" for row in rows
    ]
    colormap = plt.get_cmap("tab10")
    color_by_sigma = {
        sigma: colormap(index % 10)
        for index, sigma in enumerate(sorted({row.sigma for row in rows}))
    }
    colors = [color_by_sigma[row.sigma] for row in rows]
    figure_height = max(8.0, len(rows) * 0.42)
    figure, axes = plt.subplots(1, 3, figsize=(20, figure_height), sharey=True)
    panels = [
        ("Validation L2", [row.l2 for row in rows]),
        ("Validation RMSE", [row.rmse for row in rows]),
        ("Validation cosine distance", [row.cosine_distance for row in rows]),
    ]
    positions = list(range(len(rows)))
    for axis, (title, values) in zip(axes, panels, strict=True):
        axis.barh(positions, values, color=colors)
        axis.set_title(title)
        axis.set_xlabel("lower is better")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    figure.suptitle("Best denoiser validation metrics")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
