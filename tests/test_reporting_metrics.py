import json

import pytest

from steering_recovery.metrics import generation_metrics
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
