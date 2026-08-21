from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def distinct_n(token_ids: Sequence[int], n: int) -> float:
    """Return unique token n-grams divided by all available token n-grams."""

    if n <= 0:
        raise ValueError("n must be positive")
    total = len(token_ids) - n + 1
    if total <= 0:
        return 0.0
    ngrams = {
        tuple(int(token_id) for token_id in token_ids[start : start + n])
        for start in range(total)
    }
    return len(ngrams) / total


class FrozenAGNewsClassifier:
    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: torch.device):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.device = device
        self.id2label = {
            int(key): str(value)
            for key, value in getattr(model.config, "id2label", {}).items()
        }

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool,
    ) -> "FrozenAGNewsClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)
        return cls(model, tokenizer, device)

    @torch.inference_mode()
    def score(
        self,
        texts: Sequence[str],
        target_indices: Sequence[int],
        *,
        batch_size: int,
        max_length: int,
    ) -> list[float]:
        if len(texts) != len(target_indices):
            raise ValueError("texts and target_indices must have the same length")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        scores: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            indices = torch.tensor(
                target_indices[start : start + batch_size],
                dtype=torch.long,
                device=self.device,
            )
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits.float()
            if torch.any(indices < 0) or torch.any(indices >= logits.shape[-1]):
                raise ValueError("target classifier index is outside model logits")
            probabilities = torch.softmax(logits, dim=-1)
            batch_scores = probabilities[
                torch.arange(len(indices), device=self.device), indices
            ]
            scores.extend(float(value) for value in batch_scores.cpu())
        return scores
