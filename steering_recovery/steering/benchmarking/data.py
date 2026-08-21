from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkPrompt:
    sample_id: str
    source_label: int
    source_topic: str
    prompt_text: str
    prompt_token_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkPrompt":
        return cls(
            sample_id=str(payload["sample_id"]),
            source_label=int(payload["source_label"]),
            source_topic=str(payload["source_topic"]),
            prompt_text=str(payload["prompt_text"]),
            prompt_token_ids=[int(value) for value in payload["prompt_token_ids"]],
        )


def select_stratified_prompts(
    rows: Iterable[dict[str, Any]],
    *,
    tokenizer: Any,
    label_column: str,
    text_column: str,
    topics: dict[int, str],
    total_samples: int,
    prompt_tokens: int,
    seed: int,
    split: str,
) -> list[BenchmarkPrompt]:
    """Select an equal number of random-source prompts from every AG News class."""

    if total_samples <= 0 or total_samples % len(topics) != 0:
        raise ValueError("total_samples must be positive and divisible by topic count")
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    per_topic = total_samples // len(topics)
    counts: Counter[int] = Counter()
    prompts: list[BenchmarkPrompt] = []
    for row_index, row in enumerate(rows):
        try:
            label = int(row.get(label_column))
        except (TypeError, ValueError):
            continue
        text = row.get(text_column)
        if label not in topics or counts[label] >= per_topic:
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids = [
            int(token_id)
            for token_id in tokenizer.encode(text, add_special_tokens=False)
        ]
        if len(token_ids) < prompt_tokens:
            continue
        selected = token_ids[:prompt_tokens]
        prompts.append(
            BenchmarkPrompt(
                sample_id=f"{split}-{row_index:05d}",
                source_label=label,
                source_topic=topics[label],
                prompt_text=tokenizer.decode(selected, skip_special_tokens=True),
                prompt_token_ids=selected,
            )
        )
        counts[label] += 1
        if all(counts[topic] == per_topic for topic in topics):
            break
    missing = {
        label: per_topic - counts[label]
        for label in topics
        if counts[label] < per_topic
    }
    if missing:
        raise RuntimeError(f"dataset does not contain enough benchmark prompts: {missing}")
    random.Random(seed).shuffle(prompts)
    return prompts


def load_ag_news_prompts(
    config: Any,
    *,
    tokenizer: Any,
    topics: dict[int, str],
    total_samples: int,
    prompt_tokens: int,
    seed: int,
) -> list[BenchmarkPrompt]:
    from datasets import load_dataset

    dataset = load_dataset(
        str(config.name),
        config.config,
        split=str(config.split),
        streaming=bool(config.streaming),
    )
    if bool(config.shuffle):
        kwargs: dict[str, Any] = {"seed": seed}
        if bool(config.streaming):
            kwargs["buffer_size"] = int(config.shuffle_buffer_size)
        dataset = dataset.shuffle(**kwargs)
    return select_stratified_prompts(
        dataset,
        tokenizer=tokenizer,
        label_column=str(config.label_column),
        text_column=str(config.text_column),
        topics=topics,
        total_samples=total_samples,
        prompt_tokens=prompt_tokens,
        seed=seed,
        split=str(config.split),
    )


def select_examples(
    rows: Sequence[dict[str, Any]],
    *,
    source_labels: Sequence[int],
    examples_per_source_topic: int,
) -> list[dict[str, Any]]:
    if examples_per_source_topic <= 0:
        raise ValueError("examples_per_source_topic must be positive")
    selected: list[dict[str, Any]] = []
    for label in source_labels:
        candidates = sorted(
            (row for row in rows if int(row["source_label"]) == int(label)),
            key=lambda row: int(row["sample_index"]),
        )
        if len(candidates) < examples_per_source_topic:
            raise RuntimeError(f"not enough examples for source label {label}")
        selected.extend(candidates[:examples_per_source_topic])
    return selected
