from types import SimpleNamespace

import pytest
import torch
from torch import nn

from steering_recovery.denoiser import ActivationDenoiser, DenoiserBundle
from steering_recovery.normalization import ActivationNormalizer
from steering_recovery.steering.epistemic.generation import (
    generate_epistemic_continuation,
)
from steering_recovery.steering.epistemic.plotting import (
    _horizontal_values,
    _shared_y_limits,
    plot_epistemic_summaries,
)
from steering_recovery.steering.epistemic.reporting import (
    write_epistemic_examples_html,
)
from steering_recovery.steering.epistemic.runner import _select_examples
from steering_recovery.steering.epistemic.statistics import (
    METRIC_LABELS,
    denoising_geometry_statistics,
    mc_dropout_statistics,
    score_steering_geometry_statistics,
    summarize_token_metrics,
)


class IntegerTokenizer:
    eos_token_id = 9

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
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


def test_mc_dropout_statistics_match_requested_definitions():
    values = mc_dropout_statistics(
        torch.zeros(2),
        torch.tensor([[1.0, 0.0], [3.0, 0.0]]),
    )
    assert values["score_mean_deviation"] == 1.0
    assert values["score_length_variance"] == 1.0
    assert values["score_cosine_distance_variance"] == 0.0
    assert values["score_pairwise_cosine_distance"] == 0.0
    assert values["score_inverse_snr"] == 0.5
    assert values["prediction_mean_deviation"] == 1.0
    opposite = mc_dropout_statistics(
        torch.zeros(2), torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    )
    assert opposite["score_pairwise_cosine_distance"] == 2.0


def test_denoising_geometry_measures_raw_l2_and_projection_removal():
    original = torch.tensor([1.0, 2.0])
    vector = torch.tensor([3.0, 4.0])
    steered = original + 2 * vector
    recovered = original + vector
    values = denoising_geometry_statistics(
        original, steered, recovered, vector
    )
    assert values["denoiser_l2_error"] == 5.0
    assert values["steering_projection_before"] == 10.0
    assert values["steering_projection_after"] == 5.0
    assert values["steering_projection_removal"] == 5.0


def test_score_steering_geometry_uses_negative_direction_and_actual_score():
    values = score_steering_geometry_statistics(
        torch.zeros(2),
        torch.tensor([[-4.0, 8.0], [-4.0, -8.0]]),
        torch.tensor([1.0, 0.0]),
        noise_sigma=2.0,
    )
    expected_distance = 1 - 1 / 5**0.5
    assert values["score_negative_steering_cosine_distance"] == pytest.approx(
        expected_distance
    )
    assert values["score_orthogonal_norm"] == pytest.approx(2.0)


def test_epistemic_generation_records_one_mc_statistic_per_token():
    torch.manual_seed(4)
    model = TinyCausalLM().eval()
    denoiser = DenoiserBundle(
        ActivationDenoiser(hidden_size=3, latent_dim=5, num_layers=2, dropout=0.5),
        ActivationNormalizer(torch.zeros(3), torch.ones(3)),
    ).eval()
    continuation = generate_epistemic_continuation(
        model,
        IntegerTokenizer(),
        [1, 2, 3],
        torch.ones(3),
        denoiser,
        alpha=6,
        layer_index=0,
        layer_path="transformer.h",
        mc_samples=20,
        noise_sigma=0.5,
        max_new_tokens=4,
        temperature=0,
        top_p=1,
        generation_seed=7,
        dropout_seed=8,
        stop_on_eos=False,
    )
    assert len(continuation.generated_token_ids) == 4
    assert len(continuation.token_statistics) == 4
    assert continuation.forward_calls == 4
    assert denoiser.model.training is False
    assert all(
        set(METRIC_LABELS).issubset(token) for token in continuation.token_statistics
    )
    assert all(
        {"steering_projection_before", "steering_projection_after"}.issubset(token)
        for token in continuation.token_statistics
    )
    assert all(
        value >= 0
        for token in continuation.token_statistics
        for metric, value in token.items()
        if metric in METRIC_LABELS
    )


def _condition_rows():
    rows = []
    for sigma in (0.1, 0.2, 0.5, 1.0):
        for vector_index, vector_slug in enumerate(("world", "sports")):
            for alpha in (6.0, 8.0):
                token_statistics = []
                for step in range(3):
                    token_statistics.append(
                        {
                            "step": step,
                            "token_id": step,
                            "token_text": f" token-{step}<",
                            **{
                                metric: sigma + vector_index + alpha / 100 + step / 10
                                for metric in METRIC_LABELS
                            },
                        }
                    )
                rows.append(
                    {
                        "condition_id": f"{sigma}-{vector_slug}-{alpha}",
                        "sigma": sigma,
                        "vector_name": vector_slug.title(),
                        "vector_slug": vector_slug,
                        "alpha": alpha,
                        "sample_index": 0,
                        "sample_id": "test-1",
                        "source_label": 1,
                        "source_topic": "World",
                        "prompt_text": "prompt <unsafe>",
                        "token_statistics": token_statistics,
                        "metadata": {"keep": "all"},
                    }
                )
    return rows


def test_epistemic_summary_plots_and_html(tmp_path):
    rows = _condition_rows()
    summaries = summarize_token_metrics(rows)
    assert len(summaries) == 16
    assert all(summary["tokens"] == 3 for summary in summaries)
    plot_paths = plot_epistemic_summaries(
        summaries,
        tmp_path / "plots",
        vector_norms={"world": 2.0, "sports": 3.0},
        formats=["png"],
        dpi=72,
    )
    assert len(plot_paths) == len(METRIC_LABELS)
    assert all(path.is_file() for path in plot_paths)

    report = write_epistemic_examples_html(
        rows,
        tmp_path / "examples.html",
        metadata={"mc_samples": 20},
    )
    html = report.read_text(encoding="utf-8")
    assert "Статистика" in html
    assert "MC-dropout epistemic steering" in html
    assert "prompt <unsafe>" not in html
    assert r"prompt \u003cunsafe\u003e" in html
    assert "tokenTitle(token,metric)" in html
    assert "String(token[metric])" in html
    assert "span.title=title" in html
    for metric in METRIC_LABELS:
        assert metric in html


def test_shared_plot_limits_include_every_sigma_and_uncertainty_band():
    summaries = summarize_token_metrics(_condition_rows())
    metric = "score_mean_deviation"
    summaries[-1][f"{metric}_q75"] = 25.0
    lower, upper = _shared_y_limits(summaries, metric)
    assert lower == 0
    assert upper > 25.0


def test_projection_removal_uses_steering_displacement_norm_on_x_axis():
    rows = [{"alpha": 6.0}, {"alpha": 10.0}]
    values = _horizontal_values(rows, "steering_projection_removal", 2.5)
    assert values.tolist() == [15.0, 25.0]
    baseline_values = _horizontal_values(rows, "score_inverse_snr", 2.5)
    assert baseline_values.tolist() == [6.0, 10.0]


def test_epistemic_examples_are_balanced_across_source_topics():
    rows = [
        {
            "condition_id": "condition",
            "sample_index": label,
            "source_label": label,
        }
        for label in (1, 2, 3, 4)
    ]
    selected = _select_examples(rows, examples_per_condition=4)
    assert [row["source_label"] for row in selected] == [1, 2, 3, 4]
