from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from steering_recovery.steering.benchmarking.data import (
    select_examples,
    select_stratified_prompts,
)
from steering_recovery.steering.benchmarking.generation import (
    generate_steered_continuation,
)
from steering_recovery.steering.benchmarking.plotting import plot_benchmark_series
from steering_recovery.steering.benchmarking.runner import load_steering_vectors
from steering_recovery.steering.benchmarking.scoring import (
    conditional_perplexities_from_logits,
)
from steering_recovery.steering.benchmarking.statistics import (
    bootstrap_mean_interval,
    summarize_condition,
)


class IntegerTokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [int(value) for value in text.split()]

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return " ".join(str(value) for value in token_ids)


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)
        self.embedding = nn.Embedding(10, 3)
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([nn.Identity()])
        self.head = nn.Linear(3, 10, bias=False)

    def forward(self, input_ids, attention_mask, use_cache, past_key_values=None):
        del attention_mask, use_cache, past_key_values
        hidden = self.transformer.h[0](self.embedding(input_ids))
        return SimpleNamespace(logits=self.head(hidden), past_key_values=())


class TrackingDenoiser:
    def __init__(self):
        self.model = nn.Linear(3, 3)
        self.calls = 0

    def denoise_steered(self, activations, raw_delta):
        assert activations.shape == raw_delta.shape
        self.calls += 1
        return activations


def test_stratified_prompt_selection_uses_exact_token_prefixes():
    rows = [
        {"label": label, "description": f"{label} 2 3 4 5"}
        for _ in range(3)
        for label in (1, 2, 3, 4)
    ]
    prompts = select_stratified_prompts(
        rows,
        tokenizer=IntegerTokenizer(),
        label_column="label",
        text_column="description",
        topics={1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"},
        total_samples=8,
        prompt_tokens=3,
        seed=7,
        split="test",
    )
    assert len(prompts) == 8
    assert {label: sum(item.source_label == label for item in prompts) for label in range(1, 5)} == {
        1: 2,
        2: 2,
        3: 2,
        4: 2,
    }
    assert all(len(item.prompt_token_ids) == 3 for item in prompts)


def test_benchmark_generation_keeps_exact_new_token_count():
    model = TinyCausalLM()
    denoiser = TrackingDenoiser()
    continuation = generate_steered_continuation(
        model,
        IntegerTokenizer(),
        [1, 2, 3],
        torch.ones(3),
        alpha=0.5,
        layer_index=0,
        layer_path="transformer.h",
        intervention_mode="every_step",
        entropy_threshold=0.35,
        denoiser=denoiser,
        max_new_tokens=4,
        temperature=0,
        top_p=1,
        seed=3,
        stop_on_eos=False,
    )
    assert len(continuation.generated_token_ids) == 4
    assert continuation.intervention_steps == 4
    assert continuation.forward_calls == 4
    assert denoiser.calls == 4


def test_conditional_perplexity_masks_prompt_targets():
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    logits = torch.zeros(1, 4, 4)
    # Only positions predicting generated labels 2 and 3 are accurate.
    logits[0, 1, 2] = 10
    logits[0, 2, 3] = 10
    perplexity = conditional_perplexities_from_logits(
        logits, input_ids, attention_mask, torch.tensor([2])
    )
    expected = np.exp(np.log(np.exp(10) + 3) - 10)
    np.testing.assert_allclose(perplexity.numpy(), [expected], rtol=1e-5)


def test_bootstrap_summary_and_example_quota_are_deterministic():
    mean, low, high = bootstrap_mean_interval(
        [0.5] * 10, confidence=0.95, resamples=100, seed=4
    )
    assert (mean, low, high) == (0.5, 0.5, 0.5)
    rows = []
    for source_label in range(1, 5):
        for index in range(3):
            rows.append(
                {
                    "condition_id": "raw__world__alpha_1",
                    "method": "raw",
                    "intervention_mode": "every_step",
                    "denoiser_checkpoint": None,
                    "vector_name": "World",
                    "vector_slug": "world",
                    "target_dataset_label": 1,
                    "target_classifier_index": 0,
                    "alpha": 1.0,
                    "sample_index": source_label * 10 + index,
                    "source_label": source_label,
                    "target_probability": 0.5,
                    "perplexity": 2.0,
                    "generated_token_ids": [1] * 40,
                }
            )
    selected = select_examples(
        rows, source_labels=[1, 2, 3, 4], examples_per_source_topic=2
    )
    assert len(selected) == 8
    assert all(sum(row["source_label"] == label for row in selected) == 2 for label in range(1, 5))
    summary = summarize_condition(
        rows, confidence=0.95, bootstrap_resamples=100, seed=8
    )
    assert summary["target_probability_mean"] == 0.5
    assert summary["perplexity_mean"] == 2.0
    assert summary["mean_generated_tokens"] == 40


def test_vector_artifact_loading_and_plotting(tmp_path):
    artifact = tmp_path / "vectors.pt"
    torch.save(
        {
            "steering_vectors": torch.arange(8).reshape(4, 2),
            "vector_names": ["World", "Sports", "Business", "Sci/Tech"],
            "vector_slugs": ["world", "sports", "business", "sci_tech"],
            "group_labels": [1, 2, 3, 4],
            "metadata": {},
        },
        artifact,
    )
    vectors, _ = load_steering_vectors(
        artifact, {"world": 0, "sports": 1, "business": 2, "sci_tech": 3}
    )
    assert len(vectors) == 4
    assert vectors[3].classifier_index == 3

    summaries = []
    for alpha in (0.0, 1.0):
        summaries.append(
            {
                "method": "raw",
                "vector_slug": "world",
                "vector_name": "World",
                "alpha": alpha,
                "confidence": 0.95,
                "target_probability_mean": 0.2 + alpha * 0.2,
                "target_probability_ci_low": 0.18 + alpha * 0.2,
                "target_probability_ci_high": 0.22 + alpha * 0.2,
                "perplexity_mean": 10 + alpha,
                "perplexity_ci_low": 9 + alpha,
                "perplexity_ci_high": 11 + alpha,
            }
        )
    paths = plot_benchmark_series(
        summaries,
        tmp_path / "plots",
        formats=["png"],
        dpi=72,
        log_perplexity_axis=True,
    )
    assert len(paths) == 1
    assert paths[0].is_file()
