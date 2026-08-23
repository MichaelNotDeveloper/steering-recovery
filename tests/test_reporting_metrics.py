import json

import pytest
import torch

from steering_recovery.metrics import denoising_metrics, generation_metrics
from steering_recovery.reporting import (
    load_prompt_records,
    write_html_report,
    write_jsonl,
)


def test_prompt_loading_jsonl_and_report_escaping(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"a","prompt":"<unsafe>"}\n', encoding="utf-8")
    records = load_prompt_records(prompts)
    rows = [
        {
            "prompt_id": "a",
            "prompt": records[0]["prompt"],
            "seed": 1,
            "generated_text": "& answer",
            "token_ids": [1, 2],
            "normalized_entropies": [0.2, 0.4],
            "intervention_steps": 1,
            "forward_calls": 2,
        }
    ]
    write_jsonl(rows, tmp_path / "rows.jsonl")
    assert json.loads((tmp_path / "rows.jsonl").read_text())["prompt_id"] == "a"
    write_html_report(rows, tmp_path / "report.html", {"scale": 1})
    report = (tmp_path / "report.html").read_text()
    assert "<unsafe>" not in report
    assert "&lt;unsafe&gt;" in report


def test_generation_metrics():
    metrics = generation_metrics(
        [
            {
                "token_ids": [1, 2],
                "normalized_entropies": [0.2, 0.4],
                "intervention_steps": 1,
                "forward_calls": 2,
            }
        ]
    )
    assert metrics["mean_generated_tokens"] == 2
    assert metrics["mean_normalized_entropy"] == pytest.approx(0.3)
    assert metrics["intervention_rate"] == 0.5


def test_denoising_metrics_include_rmse_and_cosine_distance():
    clean = torch.tensor([[1.0, 0.0]])
    noisy = torch.tensor([[0.0, 1.0]])
    metrics = denoising_metrics(clean, noisy, clean, noise_std=0.5)

    assert metrics["l2"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["cosine_distance"] == 0.0
    assert metrics["noisy_l2"] == 1.0
    assert metrics["noisy_rmse"] == 1.0
    assert metrics["noisy_cosine_distance"] == 1.0
    assert metrics["score_mse"] == 16.0
    assert metrics["score_rms"] == 4.0


def test_denoising_score_metrics_require_positive_sigma():
    values = torch.ones(1, 2)
    with pytest.raises(ValueError, match="noise_std must be finite and positive"):
        denoising_metrics(values, values, values, noise_std=0.0)


@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.5, 1.0])
def test_score_normalization_cancels_explicit_sigma_squared(sigma):
    clean = torch.zeros(1, 2)
    noisy = torch.zeros(1, 2)
    expected_score = torch.tensor([[2.0, -3.0]])
    denoised = noisy + sigma**2 * expected_score

    metrics = denoising_metrics(clean, noisy, denoised, noise_std=sigma)

    expected_score_mse = float(expected_score.square().mean())
    assert metrics["score_mse"] == pytest.approx(expected_score_mse)
    assert metrics["score_rms"] == pytest.approx(expected_score_mse**0.5)
