import pytest
import torch
from omegaconf import OmegaConf

import steering_recovery.statistics as statistics_module
from steering_recovery.statistics import (
    RunningHiddenStatistics,
    collect_hidden_statistics,
    load_normalization_statistics,
)


class ParsedExtractor:
    hidden_size = 1

    def __call__(self, texts):
        return [
            torch.tensor([[float(value)] for value in text.split(",")])
            for text in texts
        ]


def test_running_statistics_match_float64_reference_across_batches():
    values = torch.tensor(
        [
            [1.0e12 + 0.25, -1.0e12 + 2.0],
            [1.0e12 + 1.50, -1.0e12 - 1.0],
            [1.0e12 - 2.25, -1.0e12 + 4.0],
            [1.0e12 + 3.00, -1.0e12 - 3.0],
            [1.0e12 - 0.75, -1.0e12 + 1.0],
        ],
        dtype=torch.float64,
    )
    accumulator = RunningHiddenStatistics()
    accumulator.update(values[:2])
    accumulator.update(values[2:4])
    accumulator.update(values[4:])

    total, variance, count = accumulator.finalize()

    assert count == len(values)
    torch.testing.assert_close(total, values.sum(dim=0), rtol=1e-15, atol=1e-3)
    reference_variance = values.var(dim=0, correction=0)
    torch.testing.assert_close(variance, reference_variance, rtol=1e-6, atol=1e-6)
    naive_variance = values.square().mean(dim=0) - values.mean(dim=0).square()
    assert (naive_variance - reference_variance).abs().max() > 1.0


def test_load_normalization_statistics_requires_both_vectors(tmp_path):
    invalid_path = tmp_path / "invalid.pt"
    torch.save({"sum": torch.ones(2), "count": 3}, invalid_path)
    with pytest.raises(ValueError, match="variance"):
        load_normalization_statistics(invalid_path)


def test_collect_statistics_respects_exact_hidden_token_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        statistics_module,
        "load_teacher_forced_source",
        lambda **_kwargs: ParsedExtractor(),
    )
    monkeypatch.setattr(
        statistics_module,
        "HuggingFaceTextStream",
        lambda **_kwargs: ["1,2,3", "4,5,6", "7,8"],
    )
    output_path = tmp_path / "statistics.pt"
    config = OmegaConf.create(
        {
            "seed": 4,
            "device": "cpu",
            "dataset": {
                "name": "fake",
                "config": None,
                "split": "train",
                "text_column": "text",
                "skip_texts": 0,
                "shuffle_buffer_size": 0,
            },
            "source": {
                "model_name": "fake-gpt",
                "tokenizer_name": "fake-gpt",
                "model_dtype": "fp32",
                "trust_remote_code": False,
                "layer_path": "h",
                "layer_index": 0,
                "max_length": 8,
                "text_batch_size": 2,
            },
            "collection": {"max_tokens": 5, "batch_tokens": 4},
            "output_path": str(output_path),
        }
    )

    result = collect_hidden_statistics(config)
    payload = torch.load(output_path, map_location="cpu", weights_only=True)

    assert result["count"] == 5
    assert payload["count"] == 5
    assert payload["source"]["model_dtype"] == "float32"
    torch.testing.assert_close(payload["sum"], torch.tensor([15.0]).double())
    torch.testing.assert_close(payload["variance"], torch.tensor([2.0]).double())
    mean, std, _ = load_normalization_statistics(output_path)
    torch.testing.assert_close(mean, torch.tensor([3.0]))
    torch.testing.assert_close(std, torch.tensor([2.0]).sqrt())
