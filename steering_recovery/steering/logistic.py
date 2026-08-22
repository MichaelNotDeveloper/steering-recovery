from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from steering_recovery.runtime import ensure_output_dir
from steering_recovery.steering.core import Label, TopicDefinition


class OneVsRestLogisticTrainer:
    """Online trainer for independent one-vs-rest L2 logistic regressions."""

    def __init__(
        self,
        *,
        hidden_size: int,
        topics: Sequence[TopicDefinition],
        learning_rate: float,
        l2_strength: float,
    ):
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not topics:
            raise ValueError("at least one logistic-regression topic is required")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2_strength < 0:
            raise ValueError("l2_strength must be non-negative")
        self.hidden_size = int(hidden_size)
        self.topics = tuple(topics)
        self.learning_rate = float(learning_rate)
        self.l2_strength = float(l2_strength)
        self.label_to_index = {
            topic.label: index for index, topic in enumerate(self.topics)
        }
        if len(self.label_to_index) != len(self.topics):
            raise ValueError("logistic-regression topic labels must be unique")
        self.model = nn.Linear(self.hidden_size, len(self.topics))
        nn.init.zeros_(self.model.weight)
        nn.init.zeros_(self.model.bias)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.history: list[dict[str, float | int]] = []

    def update(
        self, hidden_chunks: Sequence[torch.Tensor], labels: Sequence[Label]
    ) -> None:
        if len(hidden_chunks) != len(labels) or not hidden_chunks:
            raise ValueError("hidden chunks and labels must be non-empty and aligned")
        features = torch.cat([chunk.detach().float().cpu() for chunk in hidden_chunks])
        if features.ndim != 2 or features.shape[1] != self.hidden_size:
            raise ValueError("logistic-regression features must be [tokens, hidden]")
        target_indices: list[int] = []
        for chunk, label in zip(hidden_chunks, labels):
            if label not in self.label_to_index:
                raise KeyError(f"unknown logistic-regression label {label!r}")
            target_indices.extend([self.label_to_index[label]] * len(chunk))
        indices = torch.tensor(target_indices, dtype=torch.long)
        targets = torch.nn.functional.one_hot(
            indices, num_classes=len(self.topics)
        ).float()

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(features)
        per_class_data_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ).mean(dim=0)
        per_class_l2 = 0.5 * self.l2_strength * self.model.weight.square().sum(dim=1)
        per_class_loss = per_class_data_loss + per_class_l2
        loss = per_class_loss.mean()
        if not torch.isfinite(loss):
            raise ValueError("logistic-regression loss became non-finite")
        loss.backward()
        self.optimizer.step()

        record: dict[str, float | int] = {
            "step": len(self.history) + 1,
            "loss": float(loss.detach()),
            "data_loss": float(per_class_data_loss.mean().detach()),
            "l2_penalty": float(per_class_l2.mean().detach()),
            "tokens": len(features),
        }
        for index, topic in enumerate(self.topics):
            record[f"{topic.slug}_loss"] = float(per_class_loss[index].detach())
        self.history.append(record)

    def save(
        self,
        output_dir: str | Path,
        *,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.history:
            raise ValueError("cannot save logistic regressions before training")
        output_dir = ensure_output_dir(output_dir)
        weights = self.model.weight.detach().float().cpu()
        bias = self.model.bias.detach().float().cpu()
        training = {
            "optimizer": "Adam",
            "learning_rate": self.learning_rate,
            "l2_strength": self.l2_strength,
            "epochs": 1,
            "validation": False,
            "steps": len(self.history),
        }
        combined_file = "logistic_regressions.pt"
        self._atomic_torch_save(
            {
                "format_version": 1,
                "method": "one_vs_rest_logistic_regression",
                "weights": weights,
                "bias": bias,
                "steering_vectors": weights,
                "vector_names": [topic.name for topic in self.topics],
                "vector_slugs": [topic.slug for topic in self.topics],
                "group_labels": [topic.label for topic in self.topics],
                "topic_labels": [topic.label for topic in self.topics],
                "topic_names": [topic.name for topic in self.topics],
                "topic_slugs": [topic.slug for topic in self.topics],
                "training": training,
                "loss_history": self.history,
                "metadata": dict(metadata),
            },
            output_dir / combined_file,
        )
        topic_files: list[str] = []
        for index, topic in enumerate(self.topics):
            filename = f"logistic_{topic.slug}.pt"
            self._atomic_torch_save(
                {
                    "format_version": 1,
                    "method": "one_vs_rest_logistic_regression",
                    "topic_label": topic.label,
                    "topic_name": topic.name,
                    "topic_slug": topic.slug,
                    "weight": weights[index],
                    "steering_vector": weights[index],
                    "bias": bias[index],
                    "training": training,
                    "metadata": dict(metadata),
                },
                output_dir / filename,
            )
            topic_files.append(filename)
        history_file = "logistic_regression_loss.json"
        self._atomic_json_save(self.history, output_dir / history_file)
        plot_file = "logistic_regression_loss.png"
        self._write_loss_plot(output_dir / plot_file)
        return {
            "enabled": True,
            "combined_file": combined_file,
            "topic_files": topic_files,
            "loss_history_file": history_file,
            "loss_plot_file": plot_file,
            **training,
            "final_loss": float(self.history[-1]["loss"]),
        }

    @staticmethod
    def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)

    @staticmethod
    def _atomic_json_save(payload: Any, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)

    def _write_loss_plot(self, path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [int(record["step"]) for record in self.history]
        figure, axis = plt.subplots(figsize=(9, 5.5))
        axis.plot(
            steps,
            [float(record["loss"]) for record in self.history],
            label="mean loss",
            linewidth=2.2,
            color="black",
        )
        for topic in self.topics:
            axis.plot(
                steps,
                [
                    float(record[f"{topic.slug}_loss"])
                    for record in self.history
                ],
                label=topic.name,
                alpha=0.82,
            )
        axis.set_title("One-vs-rest logistic regression training loss")
        axis.set_xlabel("Optimization step")
        axis.set_ylabel("BCE + L2 penalty")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
