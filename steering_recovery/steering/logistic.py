from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from steering_recovery.runtime import ensure_output_dir
from steering_recovery.steering.core import Label, TopicDefinition


class BalancedHiddenReservoir:
    """Uniform fixed-size token-hidden sample for every class."""

    def __init__(
        self,
        *,
        hidden_size: int,
        topics: Sequence[TopicDefinition],
        samples_per_class: int,
        seed: int,
    ):
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be positive")
        if not topics:
            raise ValueError("at least one topic is required")
        self.hidden_size = int(hidden_size)
        self.topics = tuple(topics)
        self.samples_per_class = int(samples_per_class)
        self.label_to_topic = {topic.label: topic for topic in self.topics}
        if len(self.label_to_topic) != len(self.topics):
            raise ValueError("topic labels must be unique")
        self._features: dict[Label, torch.Tensor] = {}
        self._priorities: dict[Label, torch.Tensor] = {}
        self._seen = {topic.label: 0 for topic in self.topics}
        self._generator = torch.Generator().manual_seed(int(seed))
        self._trim_threshold = max(
            self.samples_per_class + 1,
            int(self.samples_per_class * 1.2),
        )

    def update(
        self, hidden_chunks: Sequence[torch.Tensor], labels: Sequence[Label]
    ) -> None:
        if len(hidden_chunks) != len(labels) or not hidden_chunks:
            raise ValueError("hidden chunks and labels must be non-empty and aligned")
        grouped: dict[Label, list[torch.Tensor]] = defaultdict(list)
        for hidden, label in zip(hidden_chunks, labels):
            if label not in self.label_to_topic:
                raise KeyError(f"unknown hidden-state label {label!r}")
            hidden = hidden.detach().float().cpu()
            if hidden.ndim != 2 or hidden.shape[1] != self.hidden_size:
                raise ValueError("hidden features must be [tokens, hidden_size]")
            if len(hidden):
                grouped[label].append(hidden)

        for label, chunks in grouped.items():
            incoming = torch.cat(chunks)
            priorities = torch.rand(len(incoming), generator=self._generator)
            self._seen[label] += len(incoming)
            if label in self._features:
                self._features[label] = torch.cat((self._features[label], incoming))
                self._priorities[label] = torch.cat(
                    (self._priorities[label], priorities)
                )
            else:
                self._features[label] = incoming
                self._priorities[label] = priorities
            if len(self._features[label]) >= self._trim_threshold:
                self._trim(label)

    def _trim(self, label: Label) -> None:
        features = self._features[label]
        if len(features) <= self.samples_per_class:
            return
        indices = torch.topk(
            self._priorities[label], self.samples_per_class, sorted=False
        ).indices
        self._features[label] = features.index_select(0, indices)
        self._priorities[label] = self._priorities[label].index_select(0, indices)

    def finalize(self) -> dict[Label, torch.Tensor]:
        missing = {
            topic.slug: self.samples_per_class - self._seen[topic.label]
            for topic in self.topics
            if self._seen[topic.label] < self.samples_per_class
        }
        if missing:
            raise RuntimeError(
                f"not enough hidden states for the balanced sample: {missing}"
            )
        result: dict[Label, torch.Tensor] = {}
        for topic in self.topics:
            self._trim(topic.label)
            result[topic.label] = self._features[topic.label]
            if len(result[topic.label]) != self.samples_per_class:
                raise RuntimeError("balanced reservoir returned an invalid sample size")
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "strategy": "uniform_random_priority_reservoir",
            "samples_per_class": self.samples_per_class,
            "seen_hidden_states": {
                topic.slug: self._seen[topic.label] for topic in self.topics
            },
        }


def binary_auc_metrics(targets: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Compute threshold-grouped ROC-AUC and trapezoidal PR-AUC."""

    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if targets.ndim != 1 or scores.ndim != 1 or len(targets) != len(scores):
        raise ValueError("targets and scores must be aligned one-dimensional arrays")
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("AUC metrics require both positive and negative examples")

    order = np.argsort(-scores, kind="stable")
    ordered_targets = targets[order]
    ordered_scores = scores[order]
    threshold_indices = np.r_[
        np.flatnonzero(np.diff(ordered_scores)), len(ordered_scores) - 1
    ]
    true_positives = np.cumsum(ordered_targets)[threshold_indices].astype(np.float64)
    false_positives = (threshold_indices + 1).astype(np.float64) - true_positives

    true_positive_rate = np.r_[0.0, true_positives / positives]
    false_positive_rate = np.r_[0.0, false_positives / negatives]
    roc_auc = float(
        np.sum(
            np.diff(false_positive_rate)
            * (true_positive_rate[:-1] + true_positive_rate[1:])
            * 0.5
        )
    )

    recall = np.r_[0.0, true_positives / positives]
    precision = np.r_[1.0, true_positives / (true_positives + false_positives)]
    auc_prc = float(np.sum(np.diff(recall) * (precision[:-1] + precision[1:]) * 0.5))
    return roc_auc, auc_prc


class EpochLogisticTrainer:
    """Train independent one-vs-rest L2 logistic regressions by epochs."""

    def __init__(
        self,
        *,
        hidden_size: int,
        topics: Sequence[TopicDefinition],
        learning_rate: float,
        l2_strength: float,
        device: torch.device,
        seed: int,
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
        self.device = device
        self.seed = int(seed)
        self.label_to_index = {
            topic.label: index for index, topic in enumerate(self.topics)
        }
        if len(self.label_to_index) != len(self.topics):
            raise ValueError("logistic-regression topic labels must be unique")
        self.model = nn.Linear(self.hidden_size, len(self.topics)).to(device)
        nn.init.zeros_(self.model.weight)
        nn.init.zeros_(self.model.bias)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.history: list[dict[str, float | int]] = []

    def fit(
        self,
        features_by_label: Mapping[Label, torch.Tensor],
        *,
        epochs: int,
        batch_size_per_class: int,
        evaluation_batch_size: int,
    ) -> list[dict[str, float | int]]:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size_per_class <= 0 or evaluation_batch_size <= 0:
            raise ValueError("training and evaluation batch sizes must be positive")
        features = self._validate_features(features_by_label)
        sample_count = len(next(iter(features.values())))

        for epoch in range(1, epochs + 1):
            generator = torch.Generator().manual_seed(self.seed + epoch)
            permutations = {
                label: torch.randperm(sample_count, generator=generator)
                for label in features
            }
            self.model.train()
            for start in range(0, sample_count, batch_size_per_class):
                end = min(start + batch_size_per_class, sample_count)
                batch_parts: list[torch.Tensor] = []
                target_parts: list[torch.Tensor] = []
                for topic in self.topics:
                    indices = permutations[topic.label][start:end]
                    batch_parts.append(features[topic.label].index_select(0, indices))
                    target_parts.append(
                        torch.full(
                            (end - start,),
                            self.label_to_index[topic.label],
                            dtype=torch.long,
                        )
                    )
                batch = torch.cat(batch_parts).to(self.device)
                target_indices = torch.cat(target_parts).to(self.device)
                targets = torch.nn.functional.one_hot(
                    target_indices, num_classes=len(self.topics)
                ).float()

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch)
                per_class_data_loss = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, targets, reduction="none"
                    ).mean(dim=0)
                )
                per_class_l2 = (
                    0.5 * self.l2_strength * self.model.weight.square().sum(dim=1)
                )
                loss = (per_class_data_loss + per_class_l2).mean()
                if not torch.isfinite(loss):
                    raise ValueError("logistic-regression loss became non-finite")
                loss.backward()
                self.optimizer.step()

            record = self._evaluate(
                features, evaluation_batch_size=evaluation_batch_size
            )
            record["epoch"] = epoch
            record["hidden_states"] = sample_count * len(self.topics)
            self.history.append(record)
        return self.history

    def _validate_features(
        self, features_by_label: Mapping[Label, torch.Tensor]
    ) -> dict[Label, torch.Tensor]:
        result: dict[Label, torch.Tensor] = {}
        counts: set[int] = set()
        for topic in self.topics:
            if topic.label not in features_by_label:
                raise KeyError(f"missing hidden sample for label {topic.label!r}")
            feature = features_by_label[topic.label].detach().float().cpu()
            if feature.ndim != 2 or feature.shape[1] != self.hidden_size:
                raise ValueError("training features must be [tokens, hidden_size]")
            if not len(feature):
                raise ValueError("training features must not be empty")
            result[topic.label] = feature
            counts.add(len(feature))
        if len(counts) != 1:
            raise ValueError("every class must contain the same number of hiddens")
        return result

    @torch.inference_mode()
    def _evaluate(
        self,
        features: Mapping[Label, torch.Tensor],
        *,
        evaluation_batch_size: int,
    ) -> dict[str, float]:
        self.model.eval()
        score_chunks: list[np.ndarray] = []
        target_chunks: list[np.ndarray] = []
        data_loss_sum = torch.zeros(len(self.topics), dtype=torch.float64)
        total = 0
        for actual_topic in self.topics:
            topic_features = features[actual_topic.label]
            actual_index = self.label_to_index[actual_topic.label]
            for start in range(0, len(topic_features), evaluation_batch_size):
                batch = topic_features[start : start + evaluation_batch_size].to(
                    self.device
                )
                logits = self.model(batch)
                target_indices = torch.full(
                    (len(batch),), actual_index, dtype=torch.long, device=self.device
                )
                targets = torch.nn.functional.one_hot(
                    target_indices, num_classes=len(self.topics)
                ).float()
                data_loss_sum += (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, targets, reduction="none"
                    )
                    .sum(dim=0)
                    .double()
                    .cpu()
                )
                score_chunks.append(torch.sigmoid(logits).float().cpu().numpy())
                target_chunks.append(targets.byte().cpu().numpy())
                total += len(batch)

        scores = np.concatenate(score_chunks)
        targets = np.concatenate(target_chunks)
        per_class_data_loss = data_loss_sum / total
        per_class_l2 = (
            0.5
            * self.l2_strength
            * self.model.weight.detach().square().sum(dim=1).double().cpu()
        )
        per_class_loss = per_class_data_loss + per_class_l2
        record: dict[str, float] = {
            "loss": float(per_class_loss.mean()),
            "data_loss": float(per_class_data_loss.mean()),
            "l2_penalty": float(per_class_l2.mean()),
        }
        roc_values: list[float] = []
        pr_values: list[float] = []
        for index, topic in enumerate(self.topics):
            roc_auc, auc_prc = binary_auc_metrics(targets[:, index], scores[:, index])
            record[f"{topic.slug}_loss"] = float(per_class_loss[index])
            record[f"{topic.slug}_roc_auc"] = roc_auc
            record[f"{topic.slug}_auc_prc"] = auc_prc
            roc_values.append(roc_auc)
            pr_values.append(auc_prc)
        record["macro_roc_auc"] = float(np.mean(roc_values))
        record["macro_auc_prc"] = float(np.mean(pr_values))
        return record

    @torch.inference_mode()
    def predict_proba(
        self, features: torch.Tensor, *, batch_size: int = 4096
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.hidden_size:
            raise ValueError("prediction features must be [tokens, hidden_size]")
        self.model.eval()
        probabilities = []
        for start in range(0, len(features), batch_size):
            logits = self.model(features[start : start + batch_size].to(self.device))
            probabilities.append(torch.sigmoid(logits).float().cpu())
        return (
            torch.cat(probabilities)
            if probabilities
            else torch.empty((0, len(self.topics)))
        )

    def save(
        self,
        output_dir: str | Path,
        *,
        metadata: Mapping[str, Any],
        sampling: Mapping[str, Any],
        training: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.history:
            raise ValueError("cannot save logistic regressions before training")
        output_dir = ensure_output_dir(output_dir)
        weights = self.model.weight.detach().float().cpu()
        bias = self.model.bias.detach().float().cpu()
        training_metadata = {
            "optimizer": "Adam",
            "learning_rate": self.learning_rate,
            "l2_strength": self.l2_strength,
            "validation": False,
            **dict(training),
        }
        combined_file = "logistic_regressions.pt"
        self._atomic_torch_save(
            {
                "format_version": 2,
                "method": "epoch_one_vs_rest_logistic_regression",
                "weights": weights,
                "bias": bias,
                "steering_vectors": weights,
                "vector_names": [topic.name for topic in self.topics],
                "vector_slugs": [topic.slug for topic in self.topics],
                "group_labels": [topic.label for topic in self.topics],
                "topic_labels": [topic.label for topic in self.topics],
                "topic_names": [topic.name for topic in self.topics],
                "topic_slugs": [topic.slug for topic in self.topics],
                "sampling": dict(sampling),
                "training": training_metadata,
                "training_history": self.history,
                "metadata": dict(metadata),
            },
            output_dir / combined_file,
        )
        topic_files: list[str] = []
        for index, topic in enumerate(self.topics):
            filename = f"logistic_{topic.slug}.pt"
            self._atomic_torch_save(
                {
                    "format_version": 2,
                    "method": "epoch_one_vs_rest_logistic_regression",
                    "topic_label": topic.label,
                    "topic_name": topic.name,
                    "topic_slug": topic.slug,
                    "weight": weights[index],
                    "steering_vector": weights[index],
                    "bias": bias[index],
                    "sampling": dict(sampling),
                    "training": training_metadata,
                    "metadata": dict(metadata),
                },
                output_dir / filename,
            )
            topic_files.append(filename)
        history_file = "training_history.json"
        self._atomic_json_save(self.history, output_dir / history_file)
        plot_file = "training_curves.png"
        self._write_training_plot(output_dir / plot_file)
        final = self.history[-1]
        return {
            "combined_file": combined_file,
            "topic_files": topic_files,
            "training_history_file": history_file,
            "training_plot_file": plot_file,
            "epochs": int(training_metadata["epochs"]),
            "validation": False,
            "final_loss": float(final["loss"]),
            "final_macro_roc_auc": float(final["macro_roc_auc"]),
            "final_macro_auc_prc": float(final["macro_auc_prc"]),
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

    def _write_training_plot(self, path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [int(record["epoch"]) for record in self.history]
        figure, axes = plt.subplots(1, 3, figsize=(17, 5.4))
        axes[0].plot(
            epochs,
            [float(record["loss"]) for record in self.history],
            label="macro",
            linewidth=2.4,
            color="black",
        )
        for topic in self.topics:
            axes[0].plot(
                epochs,
                [float(record[f"{topic.slug}_loss"]) for record in self.history],
                label=topic.name,
                alpha=0.82,
            )
            axes[1].plot(
                epochs,
                [float(record[f"{topic.slug}_roc_auc"]) for record in self.history],
                label=topic.name,
                alpha=0.82,
            )
            axes[2].plot(
                epochs,
                [float(record[f"{topic.slug}_auc_prc"]) for record in self.history],
                label=topic.name,
                alpha=0.82,
            )
        axes[1].plot(
            epochs,
            [float(record["macro_roc_auc"]) for record in self.history],
            label="macro",
            linewidth=2.4,
            color="black",
        )
        axes[2].plot(
            epochs,
            [float(record["macro_auc_prc"]) for record in self.history],
            label="macro",
            linewidth=2.4,
            color="black",
        )
        titles = ("BCE + L2", "ROC-AUC", "AUC-PRC")
        for axis, title in zip(axes, titles):
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8)
        axes[0].set_ylabel("Loss")
        axes[1].set_ylim(0.0, 1.0)
        axes[2].set_ylim(0.0, 1.0)
        figure.suptitle("AG News one-vs-rest logistic regressions")
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
