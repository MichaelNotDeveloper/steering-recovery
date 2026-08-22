from types import SimpleNamespace

import pytest
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
from steering_recovery.steering.benchmarking.reporting import write_examples_html
from steering_recovery.steering.benchmarking.runner import (
    load_steering_vectors,
    validate_gpt2_small_vector_precision,
)
from steering_recovery.steering.benchmarking.scoring import (
    CausalLMSLORScorer,
    distinct_n,
    estimate_token_unigram_log_probabilities,
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


class WhitespaceBatchTokenizer:
    pad_token_id = 0

    def __call__(
        self, texts, *, add_special_tokens, padding, truncation
    ):
        assert add_special_tokens is False
        assert padding is False
        assert truncation is False
        return {"input_ids": [[int(value) for value in text.split()] for text in texts]}


class NextTokenCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=4, max_position_embeddings=16)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        logits = torch.zeros(*input_ids.shape, 4, device=input_ids.device)
        logits.scatter_(-1, ((input_ids + 1) % 4).unsqueeze(-1), 2.0)
        return SimpleNamespace(logits=logits)


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


def test_distinct_n_uses_generated_token_ngrams():
    assert distinct_n([1, 2, 3, 1, 2, 3], 3) == 3 / 4
    assert distinct_n([1, 2], 3) == 0


def test_slor_uses_contextual_lm_and_corpus_unigram_log_probabilities():
    estimate = estimate_token_unigram_log_probabilities(
        ["0 1", "2 3"],
        tokenizer=WhitespaceBatchTokenizer(),
        vocab_size=4,
        batch_size=2,
        smoothing=1.0,
        max_documents=None,
    )
    assert estimate.documents == 2
    assert estimate.tokens == 4
    assert torch.allclose(
        estimate.log_probabilities, torch.full((4,), torch.log(torch.tensor(0.25)))
    )

    scorer = CausalLMSLORScorer(
        NextTokenCausalLM(), WhitespaceBatchTokenizer(), torch.device("cpu")
    )
    score = scorer.score(
        [[0, 1]],
        [[2, 3]],
        unigram_log_probabilities=estimate.log_probabilities,
        batch_size=1,
    )[0]
    contextual = torch.log_softmax(torch.tensor([0.0, 0.0, 2.0, 0.0]), dim=0)[2]
    expected = (contextual - torch.log(torch.tensor(0.25))).item()
    assert score == pytest.approx(expected)


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
                    "distinct_1": 0.5,
                    "distinct_2": 0.65,
                    "distinct_3": 0.75,
                    "slor": 1.25,
                    "generated_token_ids": [1] * 40,
                }
            )
    selected = select_examples(
        rows, source_labels=[1, 2, 3, 4], examples_per_source_topic=2
    )
    assert len(selected) == 8
    assert all(sum(row["source_label"] == label for row in selected) == 2 for label in range(1, 5))
    summary = summarize_condition(
        rows,
        metric_fields=["distinct_1", "distinct_2", "distinct_3", "slor"],
        confidence=0.95,
        bootstrap_resamples=100,
        seed=8,
    )
    assert summary["target_probability_mean"] == 0.5
    assert summary["distinct_1_mean"] == 0.5
    assert summary["distinct_2_mean"] == pytest.approx(0.65)
    assert summary["distinct_3_mean"] == 0.75
    assert summary["slor_mean"] == 1.25
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
                "distinct_1_mean": 0.4 + alpha * 0.1,
                "distinct_1_ci_low": 0.38 + alpha * 0.1,
                "distinct_1_ci_high": 0.42 + alpha * 0.1,
                "distinct_2_mean": 0.5 + alpha * 0.1,
                "distinct_2_ci_low": 0.48 + alpha * 0.1,
                "distinct_2_ci_high": 0.52 + alpha * 0.1,
                "distinct_3_mean": 0.6 + alpha * 0.1,
                "distinct_3_ci_low": 0.58 + alpha * 0.1,
                "distinct_3_ci_high": 0.62 + alpha * 0.1,
                "slor_mean": 1.2 + alpha * 0.2,
                "slor_ci_low": 1.1 + alpha * 0.2,
                "slor_ci_high": 1.3 + alpha * 0.2,
            }
        )
    paths = plot_benchmark_series(
        summaries,
        tmp_path / "plots",
        distinct_orders=[1, 2, 3],
        slor_model_name="gpt2-large",
        formats=["png"],
        dpi=72,
    )
    assert len(paths) == 4
    assert all(path.is_file() for path in paths)
    assert {path.stem.rsplit("__", 1)[-1] for path in paths} == {
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "slor",
    }


def test_gpt2_benchmark_rejects_reduced_precision_vectors():
    validate_gpt2_small_vector_precision(
        {"metadata": {"source": {"model_dtype": "float32"}}},
        source_model_name="gpt2",
    )
    with pytest.raises(ValueError, match="vectors generated in float32"):
        validate_gpt2_small_vector_precision(
            {"metadata": {"source": {"model_dtype": "bfloat16"}}},
            source_model_name="gpt2",
        )


def test_examples_html_filters_alpha_and_embeds_all_metadata(tmp_path):
    row = {
        "method": "raw",
        "vector_slug": "world",
        "vector_name": "World",
        "alpha": 0.5,
        "source_topic": "Sports",
        "target_probability": 0.7,
        "distinct_1": 0.6,
        "distinct_2": 0.7,
        "distinct_3": 0.8,
        "slor": 1.4,
        "seed": 11,
        "prompt_text": "prompt <text>",
        "generated_text": " generated",
        "custom_metadata": {"keep": "everything"},
    }
    path = write_examples_html([row], tmp_path / "examples.html")
    document = path.read_text(encoding="utf-8")
    assert 'id="alpha"' in document
    assert "All generation metadata" in document
    assert "custom_metadata" in document
    assert "Dist-${order}" in document
    assert "SLOR:" in document
    assert "\\u003ctext>" in document
