from __future__ import annotations

import gc
import json
import os
from collections import defaultdict
from pathlib import Path
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
from steering_recovery.steering.ag_news import (
    AG_NEWS_TOPICS,
    _load_gpt2,
    iter_ag_news,
)
from steering_recovery.steering.core import (
    AllTokenHiddenExtractor,
    FullTextTokenBuilder,
    LabeledText,
    collect_group_token_hiddens,
)
from steering_recovery.steering.logistic import (
    BalancedHiddenReservoir,
    EpochLogisticTrainer,
)
from steering_recovery.steering.logistic_reporting import (
    write_token_probability_report,
)


def _select_examples(
    config: DictConfig, *, seed: int, examples_per_class: int
) -> list[LabeledText]:
    selected: dict[int, list[LabeledText]] = defaultdict(list)
    for example in iter_ag_news(config, seed):
        if len(selected[int(example.label)]) < examples_per_class:
            selected[int(example.label)].append(example)
        if all(
            len(selected[int(topic.label)]) >= examples_per_class
            for topic in AG_NEWS_TOPICS
        ):
            break
    missing = {
        topic.slug: examples_per_class - len(selected[int(topic.label)])
        for topic in AG_NEWS_TOPICS
        if len(selected[int(topic.label)]) < examples_per_class
    }
    if missing:
        raise RuntimeError(f"not enough report examples for each class: {missing}")
    return [
        example for topic in AG_NEWS_TOPICS for example in selected[int(topic.label)]
    ]


def _extract_example_hiddens(
    examples: list[LabeledText],
    *,
    token_builder: FullTextTokenBuilder,
    extractor: AllTokenHiddenExtractor,
    batch_size: int,
) -> list[tuple[LabeledText, tuple[int, ...], torch.Tensor]]:
    result: list[tuple[LabeledText, tuple[int, ...], torch.Tensor]] = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        token_sequences = [token_builder(example.text) for example in batch_examples]
        hidden_chunks = extractor(token_sequences)
        result.extend(zip(batch_examples, token_sequences, hidden_chunks))
    return result


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def train_ag_news_topic_logistic_regressions(
    config: DictConfig,
) -> dict[str, Any]:
    """Extract a balanced AG News hidden sample and train topic classifiers."""

    seed = int(config.seed)
    seed_everything(seed)
    device = resolve_device(str(config.device))
    dtype = resolve_model_dtype(
        str(config.source.model_name), str(config.source.model_dtype), device
    )
    articles_per_class = int(config.sampling.articles_per_class)
    hidden_states_per_class = int(config.sampling.hidden_states_per_class)
    extraction_batch_size = int(config.sampling.extraction_batch_size)
    if articles_per_class <= 0:
        raise ValueError("sampling.articles_per_class must be positive")

    model, tokenizer = _load_gpt2(config.source, device, dtype)
    token_builder = FullTextTokenBuilder(
        tokenizer, max_length=int(config.extraction.max_length)
    )
    extractor = AllTokenHiddenExtractor(
        model,
        layer_index=int(config.source.layer_index),
        layer_path=config.source.layer_path,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    reservoir = BalancedHiddenReservoir(
        hidden_size=extractor.hidden_size,
        topics=AG_NEWS_TOPICS,
        samples_per_class=hidden_states_per_class,
        seed=seed,
    )
    targets = {topic.label: articles_per_class for topic in AG_NEWS_TOPICS}
    progress = tqdm(
        total=articles_per_class * len(AG_NEWS_TOPICS),
        desc=f"Balanced AG News h[{int(config.source.layer_index)}] sample",
        unit="article",
        dynamic_ncols=True,
    )
    try:
        article_counts = collect_group_token_hiddens(
            iter_ag_news(config.dataset, seed),
            token_builder=token_builder,
            extractor=extractor,
            target_articles=targets,
            batch_size=extraction_batch_size,
            batch_observer=reservoir.update,
            progress=progress,
        )
        report_examples = _select_examples(
            config.examples.dataset,
            seed=seed + int(config.examples.seed_offset),
            examples_per_class=int(config.examples.per_class),
        )
        example_hiddens = _extract_example_hiddens(
            report_examples,
            token_builder=token_builder,
            extractor=extractor,
            batch_size=int(config.examples.batch_size),
        )
    finally:
        progress.close()

    features_by_label = reservoir.finalize()
    sampling_metadata = {
        **reservoir.metadata(),
        "articles_per_class": articles_per_class,
        "article_counts": {
            topic.slug: article_counts[topic.label] for topic in AG_NEWS_TOPICS
        },
        "balanced": True,
    }
    hidden_size = extractor.hidden_size
    del extractor, model, reservoir
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer = EpochLogisticTrainer(
        hidden_size=hidden_size,
        topics=AG_NEWS_TOPICS,
        learning_rate=float(config.training.learning_rate),
        l2_strength=float(config.training.l2_strength),
        device=device,
        seed=seed,
    )
    epochs = int(config.training.epochs)
    batch_size_per_class = int(config.training.batch_size_per_class)
    evaluation_batch_size = int(config.training.evaluation_batch_size)
    trainer.fit(
        features_by_label,
        epochs=epochs,
        batch_size_per_class=batch_size_per_class,
        evaluation_batch_size=evaluation_batch_size,
    )

    metadata: dict[str, Any] = {
        "trainer": "ag_news_topic_logistic_regressions",
        "dataset": {
            "name": str(config.dataset.name),
            "config": config.dataset.config,
            "split": str(config.dataset.split),
            "text_column": str(config.dataset.text_column),
            "label_column": str(config.dataset.label_column),
            "shuffled": bool(config.dataset.shuffle),
            "seed": seed,
            "topics": [
                {"label": topic.label, "name": topic.name, "slug": topic.slug}
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
            "hidden_size": hidden_size,
            "model_dtype": dtype_name(dtype),
        },
        "extraction": token_builder.metadata(),
        "sampling": sampling_metadata,
        "training": {
            "epochs": epochs,
            "batch_size_per_class": batch_size_per_class,
            "effective_batch_size": batch_size_per_class * len(AG_NEWS_TOPICS),
            "evaluation_batch_size": evaluation_batch_size,
            "learning_rate": float(config.training.learning_rate),
            "l2_strength": float(config.training.l2_strength),
            "validation": False,
            "metrics_dataset": "balanced_training_sample",
        },
        "resolved_config": config_to_dict(config),
    }
    output_dir = ensure_output_dir(config.output_dir)
    artifacts = trainer.save(
        output_dir,
        metadata=metadata,
        sampling=sampling_metadata,
        training=metadata["training"],
    )

    topic_by_label = {topic.label: topic for topic in AG_NEWS_TOPICS}
    report_payload: list[dict[str, Any]] = []
    topic_example_index: dict[int, int] = defaultdict(int)
    for example, token_ids, hidden in example_hiddens:
        label = int(example.label)
        topic = topic_by_label[label]
        topic_example_index[label] += 1
        full_token_count = len(tokenizer.encode(example.text, add_special_tokens=False))
        probabilities = trainer.predict_proba(
            hidden, batch_size=evaluation_batch_size
        ).tolist()
        report_payload.append(
            {
                "id": f"{topic.slug}-{topic_example_index[label]:02d}",
                "true_label": label,
                "true_topic": topic.name,
                "truncated": full_token_count > len(token_ids),
                "tokens": [
                    {
                        "index": index,
                        "id": int(token_id),
                        "text": tokenizer.decode(
                            [int(token_id)], clean_up_tokenization_spaces=False
                        ),
                        "probabilities": token_probabilities,
                    }
                    for index, (token_id, token_probabilities) in enumerate(
                        zip(token_ids, probabilities)
                    )
                ],
                "metadata": {
                    "dataset": str(config.examples.dataset.name),
                    "split": str(config.examples.dataset.split),
                    "true_label": label,
                    "true_topic": topic.name,
                    "source_text": example.text,
                    "token_count": len(token_ids),
                    "full_token_count": full_token_count,
                    "max_length": int(config.extraction.max_length),
                    "source_model": str(config.source.model_name),
                    "source_layer_index": int(config.source.layer_index),
                    "source_model_dtype": dtype_name(dtype),
                    "classifier_checkpoint": artifacts["combined_file"],
                },
            }
        )
    report_file = "token_probability_examples.html"
    write_token_probability_report(
        output_dir / report_file,
        topics=AG_NEWS_TOPICS,
        examples=report_payload,
        metadata=metadata,
    )
    artifacts["token_probability_report_file"] = report_file
    manifest = {
        "format_version": 1,
        "artifacts": artifacts,
        "metadata": metadata,
    }
    _write_json(output_dir / "manifest.json", manifest)
    OmegaConf.save(config, output_dir / "config.yaml")
    return {
        "output_dir": str(output_dir),
        "hidden_size": hidden_size,
        "balanced_hidden_states": hidden_states_per_class * len(AG_NEWS_TOPICS),
        "artifacts": artifacts,
    }
