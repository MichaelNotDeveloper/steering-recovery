from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import torch
from torch import nn

from steering_recovery.layers import first_tensor, resolve_layer
from steering_recovery.statistics import RunningHiddenStatistics


Label: TypeAlias = int | str


@dataclass(frozen=True)
class TopicDefinition:
    """A labeled group whose hidden-state moments are collected together."""

    label: Label
    name: str
    slug: str


@dataclass(frozen=True)
class ContrastDefinition:
    """A steering direction defined by positive and negative topic groups."""

    name: str
    slug: str
    positive_labels: tuple[Label, ...]
    negative_labels: tuple[Label, ...]


@dataclass(frozen=True)
class LabeledText:
    label: Label
    text: str


@dataclass(frozen=True)
class HiddenMoments:
    total: torch.Tensor
    variance: torch.Tensor
    count: int

    @property
    def mean(self) -> torch.Tensor:
        return self.total / self.count


@dataclass(frozen=True)
class ContrastResult:
    definition: ContrastDefinition
    steering_vector: torch.Tensor
    positive_mean: torch.Tensor
    negative_mean: torch.Tensor
    positive_count: int
    negative_count: int


class TokenSequenceHiddenExtractor(Protocol):
    hidden_size: int

    def __call__(self, token_sequences: Sequence[Sequence[int]]) -> torch.Tensor: ...


class TokenSequenceAllHiddenExtractor(Protocol):
    hidden_size: int

    def __call__(
        self, token_sequences: Sequence[Sequence[int]]
    ) -> Sequence[torch.Tensor]: ...


HiddenBatchObserver = Callable[[Sequence[torch.Tensor], Sequence[Label]], None]


class FullTextTokenBuilder:
    """Tokenize an article without a prompt and keep the complete model context."""

    def __init__(self, tokenizer: Any, *, max_length: int):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("article text must be non-empty")
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        selected = tuple(int(token_id) for token_id in token_ids[: self.max_length])
        if not selected:
            raise ValueError("article text produced no tokens")
        return selected

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "full_text_all_tokens",
            "template": "{article}",
            "max_length": self.max_length,
            "truncation": "right_at_model_context_limit",
            "capture_position": "all_non_padding_tokens",
            "pooling": "token_weighted_mean",
        }


class PromptTokenBuilder:
    """Build prompts while preserving the exact article-token budget."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        prefix: str,
        suffix: str,
        article_token_limit: int,
    ):
        if article_token_limit <= 0:
            raise ValueError("article_token_limit must be positive")
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.suffix = suffix
        self.article_token_limit = int(article_token_limit)
        self.prefix_ids = tuple(self._encode(prefix))
        self.suffix_ids = tuple(self._encode(suffix))
        if not self.suffix_ids:
            raise ValueError("prompt suffix must contain at least one token")

    def _encode(self, text: str) -> list[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [int(token_id) for token_id in token_ids]

    def __call__(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("article text must be non-empty")
        article_ids = self._encode(text)[: self.article_token_limit]
        if not article_ids:
            raise ValueError("article text produced no tokens")
        return self.prefix_ids + tuple(article_ids) + self.suffix_ids

    def metadata(self) -> dict[str, Any]:
        final_token = self.tokenizer.decode([self.suffix_ids[-1]])
        return {
            "template": f"{self.prefix}{{article}}{self.suffix}",
            "prefix": self.prefix,
            "suffix": self.suffix,
            "article_token_limit": self.article_token_limit,
            "article_truncation": "token_ids_before_prompt_concatenation",
            "capture_position": "last_prompt_token",
            "capture_token": final_token,
        }


class LastTokenHiddenExtractor:
    """Capture the selected block output at each prompt's final real token."""

    def __init__(
        self,
        model: nn.Module,
        *,
        layer_index: int,
        layer_path: str | None,
        pad_token_id: int,
        device: torch.device,
    ):
        self.model = model.eval()
        self.layer = resolve_layer(model, layer_index, layer_path)
        self.layer_index = int(layer_index)
        self.layer_path = layer_path
        self.pad_token_id = int(pad_token_id)
        self.device = device
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("cannot infer hidden size from source model config")
        self.hidden_size = int(hidden_size)
        context_size = getattr(model.config, "max_position_embeddings", None)
        if context_size is None:
            context_size = getattr(model.config, "n_positions", None)
        self.context_size = int(context_size) if context_size is not None else None

    @torch.inference_mode()
    def __call__(self, token_sequences: Sequence[Sequence[int]]) -> torch.Tensor:
        if not token_sequences:
            return torch.empty((0, self.hidden_size), dtype=torch.float32)
        lengths = torch.tensor([len(tokens) for tokens in token_sequences])
        if torch.any(lengths <= 0):
            raise ValueError("prompt token sequences must be non-empty")
        max_length = int(lengths.max())
        if self.context_size is not None and max_length > self.context_size:
            raise ValueError(
                f"prompt has {max_length} tokens but model context is "
                f"{self.context_size}"
            )

        input_ids = torch.full(
            (len(token_sequences), max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, tokens in enumerate(token_sequences):
            row = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
            input_ids[row_index, : len(row)] = row
            attention_mask[row_index, : len(row)] = 1

        captured: list[torch.Tensor] = []

        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object):
            hidden = first_tensor(output)
            if hidden is None:
                raise TypeError("selected source layer did not return hidden states")
            captured.append(hidden.detach())

        handle = self.layer.register_forward_hook(hook)
        try:
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(
                f"selected source layer was called {len(captured)} times; expected once"
            )
        hidden = captured[0]
        if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
            raise ValueError(
                "source layer output must have shape [batch, sequence, hidden]"
            )
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError("source layer output differs from configured hidden size")
        row_indices = torch.arange(len(token_sequences), device=hidden.device)
        positions = lengths.to(hidden.device) - 1
        return hidden[row_indices, positions].float().cpu()


class AllTokenHiddenExtractor:
    """Capture selected-block hidden states for every non-padding input token."""

    def __init__(
        self,
        model: nn.Module,
        *,
        layer_index: int,
        layer_path: str | None,
        pad_token_id: int,
        device: torch.device,
    ):
        self.model = model.eval()
        self.layer = resolve_layer(model, layer_index, layer_path)
        self.pad_token_id = int(pad_token_id)
        self.device = device
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("cannot infer hidden size from source model config")
        self.hidden_size = int(hidden_size)
        context_size = getattr(model.config, "max_position_embeddings", None)
        if context_size is None:
            context_size = getattr(model.config, "n_positions", None)
        self.context_size = int(context_size) if context_size is not None else None

    @torch.inference_mode()
    def __call__(self, token_sequences: Sequence[Sequence[int]]) -> list[torch.Tensor]:
        if not token_sequences:
            return []
        lengths = torch.tensor([len(tokens) for tokens in token_sequences])
        if torch.any(lengths <= 0):
            raise ValueError("article token sequences must be non-empty")
        max_length = int(lengths.max())
        if self.context_size is not None and max_length > self.context_size:
            raise ValueError(
                f"article has {max_length} tokens but model context is "
                f"{self.context_size}"
            )
        input_ids = torch.full(
            (len(token_sequences), max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, tokens in enumerate(token_sequences):
            row = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
            input_ids[row_index, : len(row)] = row
            attention_mask[row_index, : len(row)] = 1

        captured: list[torch.Tensor] = []

        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object):
            hidden = first_tensor(output)
            if hidden is None:
                raise TypeError("selected source layer did not return hidden states")
            captured.append(hidden.detach())

        handle = self.layer.register_forward_hook(hook)
        try:
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(
                f"selected source layer was called {len(captured)} times; expected once"
            )
        hidden = captured[0]
        if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
            raise ValueError(
                "source layer output must have shape [batch, sequence, hidden]"
            )
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError("source layer output differs from configured hidden size")
        return [
            hidden[index, : int(length)].float().cpu()
            for index, length in enumerate(lengths)
        ]


def collect_group_moments(
    examples: Iterable[LabeledText],
    *,
    prompt_builder: PromptTokenBuilder,
    extractor: TokenSequenceHiddenExtractor,
    target_counts: Mapping[Label, int],
    batch_size: int,
    progress: Any | None = None,
) -> dict[Label, HiddenMoments]:
    """Collect exact per-group quotas in one pass over a labeled text source."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not target_counts or any(count <= 0 for count in target_counts.values()):
        raise ValueError("all target counts must be positive")
    accumulators = {label: RunningHiddenStatistics() for label in target_counts}
    pending: Counter[Label] = Counter()
    batch_labels: list[Label] = []
    batch_tokens: list[tuple[int, ...]] = []

    def complete() -> bool:
        return all(
            accumulators[label].count >= target
            for label, target in target_counts.items()
        )

    def flush() -> None:
        if not batch_tokens:
            return
        hidden = extractor(batch_tokens)
        if hidden.ndim != 2 or hidden.shape != (
            len(batch_tokens),
            extractor.hidden_size,
        ):
            raise ValueError("extractor must return [number_of_prompts, hidden_size]")
        for label in dict.fromkeys(batch_labels):
            indices = [
                index
                for index, batch_label in enumerate(batch_labels)
                if batch_label == label
            ]
            accumulators[label].update(hidden[indices])
        if progress is not None:
            progress.update(len(batch_tokens))
        batch_labels.clear()
        batch_tokens.clear()
        pending.clear()

    for example in examples:
        if example.label not in accumulators:
            continue
        collected = accumulators[example.label].count + pending[example.label]
        if collected >= target_counts[example.label]:
            continue
        try:
            tokens = prompt_builder(example.text)
        except ValueError:
            continue
        batch_labels.append(example.label)
        batch_tokens.append(tokens)
        pending[example.label] += 1
        if len(batch_tokens) >= batch_size:
            flush()
            if complete():
                break
    flush()

    counts = {label: accumulator.count for label, accumulator in accumulators.items()}
    missing = {
        label: target_counts[label] - count
        for label, count in counts.items()
        if count < target_counts[label]
    }
    if missing:
        raise RuntimeError(
            f"dataset ended before all hidden-state quotas were met: {missing}"
        )
    result: dict[Label, HiddenMoments] = {}
    for label, accumulator in accumulators.items():
        total, variance, count = accumulator.finalize()
        result[label] = HiddenMoments(total=total, variance=variance, count=count)
    return result


def collect_group_token_moments(
    examples: Iterable[LabeledText],
    *,
    token_builder: FullTextTokenBuilder,
    extractor: TokenSequenceAllHiddenExtractor,
    target_articles: Mapping[Label, int],
    batch_size: int,
    batch_observer: HiddenBatchObserver | None = None,
    progress: Any | None = None,
) -> tuple[dict[Label, HiddenMoments], dict[Label, int]]:
    """Collect every token hidden from exact per-group article quotas in one pass."""

    accumulators = {label: RunningHiddenStatistics() for label in target_articles}

    def update_moments(
        hidden_chunks: Sequence[torch.Tensor], labels: Sequence[Label]
    ) -> None:
        for hidden, label in zip(hidden_chunks, labels):
            accumulators[label].update(hidden)
        if batch_observer is not None:
            batch_observer(hidden_chunks, labels)

    article_counts = collect_group_token_hiddens(
        examples,
        token_builder=token_builder,
        extractor=extractor,
        target_articles=target_articles,
        batch_size=batch_size,
        batch_observer=update_moments,
        progress=progress,
    )
    moments: dict[Label, HiddenMoments] = {}
    for label, accumulator in accumulators.items():
        total, variance, count = accumulator.finalize()
        moments[label] = HiddenMoments(total=total, variance=variance, count=count)
    return moments, article_counts


def collect_group_token_hiddens(
    examples: Iterable[LabeledText],
    *,
    token_builder: FullTextTokenBuilder,
    extractor: TokenSequenceAllHiddenExtractor,
    target_articles: Mapping[Label, int],
    batch_size: int,
    batch_observer: HiddenBatchObserver,
    progress: Any | None = None,
) -> dict[Label, int]:
    """Pass token hiddens to an observer using exact per-group article quotas."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not target_articles or any(count <= 0 for count in target_articles.values()):
        raise ValueError("all target article counts must be positive")
    article_counts: Counter[Label] = Counter()
    batch_labels: list[Label] = []
    batch_tokens: list[tuple[int, ...]] = []

    def complete() -> bool:
        return all(
            article_counts[label] >= target for label, target in target_articles.items()
        )

    def flush() -> None:
        if not batch_tokens:
            return
        hidden_chunks = list(extractor(batch_tokens))
        if len(hidden_chunks) != len(batch_tokens):
            raise ValueError("extractor must return one hidden matrix per article")
        for hidden, label in zip(hidden_chunks, batch_labels):
            if hidden.ndim != 2 or hidden.shape[1] != extractor.hidden_size:
                raise ValueError("each article hidden matrix must be [tokens, hidden]")
            if len(hidden) == 0:
                raise ValueError(
                    "article hidden matrix must contain at least one token"
                )
        batch_observer(hidden_chunks, tuple(batch_labels))
        if progress is not None:
            progress.update(len(batch_tokens))
        batch_labels.clear()
        batch_tokens.clear()

    for example in examples:
        if example.label not in target_articles:
            continue
        if article_counts[example.label] >= target_articles[example.label]:
            continue
        try:
            tokens = token_builder(example.text)
        except ValueError:
            continue
        article_counts[example.label] += 1
        batch_labels.append(example.label)
        batch_tokens.append(tokens)
        if len(batch_tokens) >= batch_size:
            flush()
            if complete():
                break
    flush()

    missing = {
        label: target_articles[label] - article_counts[label]
        for label in target_articles
        if article_counts[label] < target_articles[label]
    }
    if missing:
        raise RuntimeError(f"dataset ended before article quotas were met: {missing}")
    return dict(article_counts)


def compute_contrasts(
    moments: Mapping[Label, HiddenMoments],
    definitions: Sequence[ContrastDefinition],
) -> list[ContrastResult]:
    """Compute weighted mean differences for arbitrary group contrasts."""

    def pooled_mean(labels: tuple[Label, ...]) -> tuple[torch.Tensor, int]:
        if not labels:
            raise ValueError("contrast sides must contain at least one label")
        missing = set(labels) - moments.keys()
        if missing:
            raise KeyError(f"contrast references unknown labels: {sorted(missing)!r}")
        count = sum(moments[label].count for label in labels)
        total = torch.zeros_like(moments[labels[0]].total)
        for label in labels:
            total = total + moments[label].total
        return total / count, count

    results: list[ContrastResult] = []
    for definition in definitions:
        overlap = set(definition.positive_labels) & set(definition.negative_labels)
        if overlap:
            raise ValueError(f"positive and negative labels overlap: {overlap}")
        positive_mean, positive_count = pooled_mean(definition.positive_labels)
        negative_mean, negative_count = pooled_mean(definition.negative_labels)
        steering_vector = (positive_mean - negative_mean).float()
        if not torch.isfinite(steering_vector).all():
            raise ValueError(f"contrast {definition.name!r} produced non-finite values")
        results.append(
            ContrastResult(
                definition=definition,
                steering_vector=steering_vector,
                positive_mean=positive_mean.float(),
                negative_mean=negative_mean.float(),
                positive_count=positive_count,
                negative_count=negative_count,
            )
        )
    return results
