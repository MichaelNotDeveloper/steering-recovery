from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn
from torch.utils.data import IterableDataset, get_worker_info

from steering_recovery.layers import first_tensor, resolve_layer


class HiddenExtractor(Protocol):
    hidden_size: int

    def __call__(self, texts: Sequence[str]) -> list[torch.Tensor]: ...


@dataclass
class HuggingFaceTextStream:
    """Restartable streaming text source backed by Hugging Face Datasets."""

    dataset_name: str = "Skylion007/openwebtext"
    dataset_config: str | None = None
    split: str = "train"
    text_column: str = "text"
    skip_texts: int = 0
    limit_texts: int | None = None
    shuffle_buffer_size: int = 0
    seed: int = 42
    epoch: int = 0

    def __post_init__(self) -> None:
        if self.skip_texts < 0:
            raise ValueError("skip_texts must be non-negative")
        if self.limit_texts is not None and self.limit_texts <= 0:
            raise ValueError("limit_texts must be positive when set")
        if self.shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be non-negative")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[str]:
        from datasets import load_dataset

        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
            streaming=True,
        )
        if self.skip_texts:
            dataset = dataset.skip(self.skip_texts)
        if self.shuffle_buffer_size:
            dataset = dataset.shuffle(
                seed=self.seed + self.epoch,
                buffer_size=self.shuffle_buffer_size,
            )
        rows: Iterable[dict[str, Any]] = dataset
        if self.limit_texts is not None:
            rows = itertools.islice(rows, self.limit_texts)
        for row in rows:
            text = row.get(self.text_column)
            if isinstance(text, str) and text.strip():
                yield text


class TeacherForcedHiddenExtractor:
    """Extract every valid layer activation except the first token per text.

    The complete ground-truth token sequence is passed through the causal model
    in one forward call. Causal masking therefore gives teacher-forced states:
    position ``t`` is computed from the real tokens through ``t`` (inclusive).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        layer_index: int,
        layer_path: str | None,
        max_length: int,
        device: torch.device,
    ):
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.layer = resolve_layer(model, layer_index, layer_path)
        self.max_length = int(max_length)
        self.device = device
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("cannot infer hidden size from source model config")
        self.hidden_size = int(hidden_size)

    @torch.inference_mode()
    def __call__(self, texts: Sequence[str]) -> list[torch.Tensor]:
        if not texts:
            return []
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        if "input_ids" not in encoded:
            raise KeyError("tokenizer output has no input_ids")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
            self.device
        )
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
        if hidden.ndim != 3 or hidden.shape[:2] != attention_mask.shape:
            raise ValueError(
                "source layer output must have shape [text_batch, sequence, hidden]"
            )
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                "source layer output hidden size differs from model config"
            )

        result: list[torch.Tensor] = []
        for row_index in range(hidden.shape[0]):
            valid_positions = attention_mask[row_index].bool().nonzero().flatten()
            # The first real token has no preceding token in this document.
            positions = valid_positions[1:]
            result.append(hidden[row_index, positions].detach())
        return result


class ExactHiddenBatcher:
    """Pack variable-length token activations into exact-size batches."""

    def __init__(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self._parts: list[torch.Tensor] = []
        self._count = 0
        self._hidden_size: int | None = None

    def add(self, hidden: torch.Tensor) -> Iterator[torch.Tensor]:
        if hidden.ndim != 2:
            raise ValueError("hidden chunks must have shape [tokens, hidden_size]")
        if self._hidden_size is None:
            self._hidden_size = hidden.shape[-1]
        elif hidden.shape[-1] != self._hidden_size:
            raise ValueError("hidden size changed within one stream")
        offset = 0
        while offset < len(hidden):
            needed = self.batch_size - self._count
            take = min(needed, len(hidden) - offset)
            self._parts.append(hidden[offset : offset + take])
            self._count += take
            offset += take
            if self._count == self.batch_size:
                yield torch.cat(self._parts, dim=0)
                self._parts.clear()
                self._count = 0


class TeacherForcedActivationIterableDataset(IterableDataset[torch.Tensor]):
    """Stream texts, extract teacher-forced states, and emit exact ``k`` batches.

    The dataset performs batching itself. Iterate it directly or use a
    ``DataLoader`` with ``batch_size=None`` and ``num_workers=0``.
    """

    def __init__(
        self,
        text_stream: Iterable[str],
        extractor: HiddenExtractor,
        *,
        batch_size: int,
        text_batch_size: int,
        max_batches: int | None = None,
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if text_batch_size <= 0:
            raise ValueError("text_batch_size must be positive")
        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be positive when set")
        self.text_stream = text_stream
        self.extractor = extractor
        self.batch_size = int(batch_size)
        self.text_batch_size = int(text_batch_size)
        self.max_batches = max_batches
        self.hidden_size = extractor.hidden_size

    def __len__(self) -> int:
        if self.max_batches is None:
            raise TypeError("an unbounded streaming dataset has no length")
        return self.max_batches

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.text_stream, "set_epoch", None)
        if setter is not None:
            setter(epoch)

    def __iter__(self) -> Iterator[torch.Tensor]:
        if get_worker_info() is not None:
            raise RuntimeError(
                "TeacherForcedActivationIterableDataset must use num_workers=0 "
                "because it owns a model"
            )
        packer = ExactHiddenBatcher(self.batch_size)
        text_batch: list[str] = []
        emitted = 0

        def process(texts: Sequence[str]) -> Iterator[torch.Tensor]:
            nonlocal emitted
            chunks = self.extractor(texts)
            if len(chunks) != len(texts):
                raise RuntimeError("extractor must return one hidden chunk per text")
            for chunk in chunks:
                for batch in packer.add(chunk):
                    emitted += 1
                    yield batch
                    if self.max_batches is not None and emitted >= self.max_batches:
                        return

        for text in self.text_stream:
            if not isinstance(text, str) or not text.strip():
                continue
            text_batch.append(text)
            if len(text_batch) < self.text_batch_size:
                continue
            yield from process(text_batch)
            text_batch.clear()
            if self.max_batches is not None and emitted >= self.max_batches:
                return
        if text_batch:
            yield from process(text_batch)


def load_teacher_forced_source(
    *,
    model_name: str,
    tokenizer_name: str | None,
    layer_index: int,
    layer_path: str | None,
    max_length: int,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool = False,
) -> TeacherForcedHiddenExtractor:
    """Load the causal GPT-style backbone used by the iterable dataset."""

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.requires_grad_(False)
    return TeacherForcedHiddenExtractor(
        model,
        tokenizer,
        layer_index=layer_index,
        layer_path=layer_path,
        max_length=max_length,
        device=device,
    )
