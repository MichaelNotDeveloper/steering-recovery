from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as functional


def conditional_perplexities_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute per-row PPL while masking every target token in the prompt."""

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits/input_ids must be [batch, seq, vocab] and [batch, seq]")
    if logits.shape[:2] != input_ids.shape or attention_mask.shape != input_ids.shape:
        raise ValueError("logits, input IDs and attention mask shapes are inconsistent")
    if prompt_lengths.shape != (input_ids.shape[0],):
        raise ValueError("prompt_lengths must have shape [batch]")
    labels = input_ids.clone()
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
    labels[(positions < prompt_lengths[:, None]) | ~attention_mask.bool()] = -100
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    losses = functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shifted_labels)
    valid = shifted_labels.ne(-100)
    counts = valid.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("each sequence must contain at least one generated token")
    mean_nll = (losses * valid).sum(dim=1) / counts
    perplexities = mean_nll.double().exp()
    if not torch.isfinite(perplexities).all():
        raise ValueError("perplexity model produced non-finite values")
    return perplexities.cpu()


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


class ConditionalPerplexityScorer:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        pad_token_id: int,
        device: torch.device,
    ):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.pad_token_id = int(pad_token_id)
        self.device = device
        self.vocab_size = int(model.config.vocab_size)
        self.context_size = int(
            getattr(
                model.config,
                "max_position_embeddings",
                getattr(model.config, "n_positions", 0),
            )
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        tokenizer_name: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool,
    ) -> "ConditionalPerplexityScorer":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=trust_remote_code
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)
        return cls(model, pad_token_id=int(tokenizer.pad_token_id), device=device)

    @torch.inference_mode()
    def score(
        self,
        prompt_token_ids: Sequence[Sequence[int]],
        generated_token_ids: Sequence[Sequence[int]],
        *,
        batch_size: int,
    ) -> list[float]:
        if len(prompt_token_ids) != len(generated_token_ids):
            raise ValueError("prompt and generated sequence counts differ")
        scores: list[float] = []
        for start in range(0, len(prompt_token_ids), batch_size):
            prompts = prompt_token_ids[start : start + batch_size]
            generated = generated_token_ids[start : start + batch_size]
            sequences = [list(prompt) + list(new) for prompt, new in zip(prompts, generated)]
            if any(not new for new in generated):
                raise ValueError("generated token sequences must be non-empty")
            max_length = max(map(len, sequences))
            if self.context_size and max_length > self.context_size:
                raise ValueError("sequence exceeds perplexity model context size")
            if max(max(sequence) for sequence in sequences) >= self.vocab_size:
                raise ValueError(
                    "generation token IDs are incompatible with perplexity model vocabulary"
                )
            input_ids = torch.full(
                (len(sequences), max_length),
                self.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, sequence in enumerate(sequences):
                values = torch.tensor(sequence, dtype=torch.long, device=self.device)
                input_ids[row_index, : len(values)] = values
                attention_mask[row_index, : len(values)] = 1
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            batch_scores = conditional_perplexities_from_logits(
                output.logits,
                input_ids,
                attention_mask,
                torch.tensor([len(prompt) for prompt in prompts], device=self.device),
            )
            scores.extend(float(value) for value in batch_scores)
        return scores
