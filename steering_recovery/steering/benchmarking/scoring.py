from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TokenUnigramEstimate:
    log_probabilities: torch.Tensor
    documents: int
    tokens: int


def estimate_token_unigram_log_probabilities(
    texts: Iterable[str],
    *,
    tokenizer: Any,
    vocab_size: int,
    batch_size: int,
    smoothing: float,
    max_documents: int | None,
) -> TokenUnigramEstimate:
    """Estimate a smoothed token unigram model from a text corpus."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive when configured")

    counts = torch.zeros(vocab_size, dtype=torch.int64)
    documents = 0
    batch: list[str] = []

    def consume(batch_texts: Sequence[str]) -> None:
        nonlocal documents
        encoded = tokenizer(
            list(batch_texts),
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        encoded_ids = encoded["input_ids"]
        ids = torch.tensor(
            [token_id for sequence in encoded_ids for token_id in sequence],
            dtype=torch.long,
        )
        if ids.numel():
            if torch.any(ids < 0) or torch.any(ids >= vocab_size):
                raise ValueError("unigram corpus contains a token outside model vocab")
            counts.add_(torch.bincount(ids, minlength=vocab_size))
        documents += len(encoded_ids)

    for text in texts:
        if max_documents is not None and documents + len(batch) >= max_documents:
            break
        if not isinstance(text, str) or not text.strip():
            continue
        batch.append(text)
        if len(batch) == batch_size:
            consume(batch)
            batch = []
    if batch:
        consume(batch)
    tokens = int(counts.sum())
    if documents == 0 or tokens == 0:
        raise ValueError("unigram corpus did not produce any tokens")
    probabilities = (counts.double() + smoothing) / (
        tokens + smoothing * vocab_size
    )
    return TokenUnigramEstimate(
        log_probabilities=probabilities.log().float(),
        documents=documents,
        tokens=tokens,
    )


class CausalLMSLORScorer:
    """Compute token-level SLOR for continuations under a causal language model."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: torch.device):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = int(model.config.vocab_size)
        self.max_positions = int(
            getattr(
                model.config,
                "max_position_embeddings",
                getattr(model.config, "n_positions", 0),
            )
        )
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("SLOR tokenizer must define pad_token_id")
        self.pad_token_id = int(pad_token_id)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        tokenizer_name: str,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool,
    ) -> "CausalLMSLORScorer":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=trust_remote_code
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("SLOR tokenizer has neither pad nor EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)
        return cls(model, tokenizer, device)

    @torch.inference_mode()
    def score(
        self,
        prompt_token_ids: Sequence[Sequence[int]],
        generated_token_ids: Sequence[Sequence[int]],
        *,
        unigram_log_probabilities: torch.Tensor,
        batch_size: int,
    ) -> list[float]:
        if len(prompt_token_ids) != len(generated_token_ids):
            raise ValueError(
                "prompt_token_ids and generated_token_ids must have the same length"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        unigram = torch.as_tensor(unigram_log_probabilities).float()
        if unigram.shape != (self.vocab_size,) or not torch.isfinite(unigram).all():
            raise ValueError("unigram log probabilities do not match model vocab")
        unigram = unigram.to(self.device)

        scores: list[float] = []
        for start in range(0, len(prompt_token_ids), batch_size):
            batch_prompts = prompt_token_ids[start : start + batch_size]
            batch_generated = generated_token_ids[start : start + batch_size]
            full_sequences: list[list[int]] = []
            prompt_lengths: list[int] = []
            generated_lengths: list[int] = []
            for prompt, generated in zip(batch_prompts, batch_generated):
                prompt = [int(value) for value in prompt]
                generated = [int(value) for value in generated]
                if not prompt:
                    raise ValueError("SLOR requires a non-empty conditioning prompt")
                if not generated:
                    raise ValueError("SLOR requires a non-empty continuation")
                full = prompt + generated
                if self.max_positions > 0 and len(full) > self.max_positions:
                    raise ValueError("SLOR input exceeds model context window")
                if min(full) < 0 or max(full) >= self.vocab_size:
                    raise ValueError("SLOR input contains a token outside model vocab")
                full_sequences.append(full)
                prompt_lengths.append(len(prompt))
                generated_lengths.append(len(generated))

            max_length = max(map(len, full_sequences))
            input_ids = torch.full(
                (len(full_sequences), max_length),
                self.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, sequence in enumerate(full_sequences):
                length = len(sequence)
                input_ids[row_index, :length] = torch.tensor(
                    sequence, dtype=torch.long, device=self.device
                )
                attention_mask[row_index, :length] = 1

            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            for row_index, (prompt_length, generated_length) in enumerate(
                zip(prompt_lengths, generated_lengths)
            ):
                step_logits = logits[
                    row_index,
                    prompt_length - 1 : prompt_length + generated_length - 1,
                ].float()
                targets = input_ids[
                    row_index, prompt_length : prompt_length + generated_length
                ]
                conditional_log_probabilities = step_logits.gather(
                    dim=-1, index=targets[:, None]
                ).squeeze(-1) - torch.logsumexp(step_logits, dim=-1)
                token_log_odds = (
                    conditional_log_probabilities - unigram[targets]
                )
                scores.append(float(token_log_odds.mean().cpu()))
        return scores


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
