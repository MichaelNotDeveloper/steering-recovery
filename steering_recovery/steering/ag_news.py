from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from steering_recovery.runtime import (
    config_to_dict,
    dtype_name,
    ensure_output_dir,
    resolve_device,
    resolve_model_dtype,
    seed_everything,
)
from steering_recovery.steering.artifacts import save_steering_artifacts
from steering_recovery.steering.core import (
    ContrastDefinition,
    LabeledText,
    LastTokenHiddenExtractor,
    PromptTokenBuilder,
    TopicDefinition,
    collect_group_moments,
    compute_contrasts,
)


AG_NEWS_TOPICS = (
    TopicDefinition(label=1, name="World", slug="world"),
    TopicDefinition(label=2, name="Sports", slug="sports"),
    TopicDefinition(label=3, name="Business", slug="business"),
    TopicDefinition(label=4, name="Sci/Tech", slug="sci_tech"),
)


def ag_news_one_vs_rest_contrasts() -> tuple[ContrastDefinition, ...]:
    labels = tuple(topic.label for topic in AG_NEWS_TOPICS)
    return tuple(
        ContrastDefinition(
            name=topic.name,
            slug=topic.slug,
            positive_labels=(topic.label,),
            negative_labels=tuple(label for label in labels if label != topic.label),
        )
        for topic in AG_NEWS_TOPICS
    )


def iter_ag_news(config: DictConfig, seed: int) -> Iterator[LabeledText]:
    """Yield labeled article descriptions from ``sh0416/ag_news``."""

    from datasets import load_dataset

    dataset = load_dataset(
        str(config.name),
        config.config,
        split=str(config.split),
        streaming=bool(config.streaming),
    )
    if bool(config.shuffle):
        shuffle_kwargs: dict[str, Any] = {"seed": seed}
        if bool(config.streaming):
            shuffle_kwargs["buffer_size"] = int(config.shuffle_buffer_size)
        dataset = dataset.shuffle(**shuffle_kwargs)
    rows = dataset
    if config.max_rows is not None:
        rows = itertools.islice(rows, int(config.max_rows))
    valid_labels = {topic.label for topic in AG_NEWS_TOPICS}
    for row in rows:
        raw_label = row.get(str(config.label_column))
        text = row.get(str(config.text_column))
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            continue
        if label in valid_labels and isinstance(text, str) and text.strip():
            yield LabeledText(label=label, text=text)


def _load_gpt2(config: DictConfig, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.tokenizer_name or config.model_name),
        trust_remote_code=bool(config.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(
        str(config.model_name),
        torch_dtype=dtype,
        trust_remote_code=bool(config.trust_remote_code),
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer


def generate_ag_news_vectors(config: DictConfig) -> dict[str, Any]:
    seed = int(config.seed)
    seed_everything(seed)
    samples_per_topic = int(config.collection.samples_per_topic)
    batch_size = int(config.collection.batch_size)
    if samples_per_topic <= 0:
        raise ValueError("collection.samples_per_topic must be positive")
    device = resolve_device(str(config.device))
    dtype = resolve_model_dtype(
        str(config.source.model_name), str(config.source.model_dtype), device
    )
    model, tokenizer = _load_gpt2(config.source, device, dtype)
    prompt_builder = PromptTokenBuilder(
        tokenizer,
        prefix=str(config.prompt.prefix),
        suffix=str(config.prompt.suffix),
        article_token_limit=int(config.prompt.article_tokens),
    )
    prompt_metadata = prompt_builder.metadata()
    expected_last_token = config.prompt.expected_last_token
    if (
        expected_last_token is not None
        and str(prompt_metadata["capture_token"]).strip()
        != str(expected_last_token).strip()
    ):
        raise ValueError(
            "prompt suffix must end in the expected capture token; got "
            f"{prompt_metadata['capture_token']!r}, expected "
            f"{expected_last_token!r}"
        )
    extractor = LastTokenHiddenExtractor(
        model,
        layer_index=int(config.source.layer_index),
        layer_path=config.source.layer_path,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    target_counts = {
        topic.label: samples_per_topic for topic in AG_NEWS_TOPICS
    }
    progress = tqdm(
        total=samples_per_topic * len(AG_NEWS_TOPICS),
        desc=f"AG News hidden states (layer {int(config.source.layer_index)})",
        unit="article",
        dynamic_ncols=True,
    )
    try:
        moments = collect_group_moments(
            iter_ag_news(config.dataset, seed),
            prompt_builder=prompt_builder,
            extractor=extractor,
            target_counts=target_counts,
            batch_size=batch_size,
            progress=progress,
        )
    finally:
        progress.close()
    contrasts = compute_contrasts(moments, ag_news_one_vs_rest_contrasts())
    metadata = {
        "generator": "ag_news_one_vs_rest",
        "dataset": {
            "name": str(config.dataset.name),
            "config": config.dataset.config,
            "split": str(config.dataset.split),
            "text_column": str(config.dataset.text_column),
            "label_column": str(config.dataset.label_column),
            "shuffled": bool(config.dataset.shuffle),
            "seed": seed,
            "topics": [
                {"label": topic.label, "name": topic.name}
                for topic in AG_NEWS_TOPICS
            ],
        },
        "source": {
            "model_name": str(config.source.model_name),
            "tokenizer_name": str(
                config.source.tokenizer_name or config.source.model_name
            ),
            "layer_path": config.source.layer_path,
            "layer_index": int(config.source.layer_index),
            "layer_number_one_based": int(config.source.layer_index) + 1,
            "hidden_size": extractor.hidden_size,
            "model_dtype": dtype_name(dtype),
        },
        "prompt": prompt_metadata,
        "collection": {
            "samples_per_topic": samples_per_topic,
            "batch_size": batch_size,
            "total_hidden_states": sum(item.count for item in moments.values()),
        },
        "method": {
            "name": "difference_of_means",
            "formula": "mean(positive_topic) - mean(all_other_topics)",
        },
        "resolved_config": config_to_dict(config),
    }
    output_dir = ensure_output_dir(config.output_dir)
    manifest = save_steering_artifacts(
        output_dir,
        topics=AG_NEWS_TOPICS,
        moments=moments,
        contrasts=contrasts,
        metadata=metadata,
    )
    OmegaConf.save(config, output_dir / "config.yaml")
    return {
        "output_dir": str(output_dir),
        "vectors": [entry["file"] for entry in manifest["vectors"]],
        "hidden_size": extractor.hidden_size,
        "total_hidden_states": samples_per_topic * len(AG_NEWS_TOPICS),
    }
