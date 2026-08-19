from types import SimpleNamespace

import torch
from torch import nn

from steering_recovery.streaming_data import (
    HuggingFaceTextStream,
    TeacherForcedActivationIterableDataset,
    TeacherForcedHiddenExtractor,
)


class FakeTokenizer:
    sequences = {
        "long": [1, 2, 3],
        "short": [4, 5],
        "single": [6],
    }

    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        assert return_tensors == "pt"
        rows = [self.sequences[text][:max_length] for text in texts]
        width = max(map(len, rows))
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeSourceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=1)
        self.h = nn.ModuleList([nn.Identity()])

    def forward(self, input_ids, attention_mask, use_cache):
        hidden = input_ids.float().unsqueeze(-1)
        hidden = self.h[0](hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class ParsedTextExtractor:
    hidden_size = 1

    def __call__(self, texts):
        return [
            torch.tensor([[float(value)] for value in text.split(",")])
            for text in texts
        ]


class FakeHuggingFaceDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def skip(self, count):
        return FakeHuggingFaceDataset(self.rows[count:])

    def shuffle(self, seed, buffer_size):
        self.shuffle_args = (seed, buffer_size)
        return self

    def __iter__(self):
        return iter(self.rows)


def test_huggingface_text_source_requests_streaming(monkeypatch):
    calls = []

    def fake_load_dataset(name, subset, **kwargs):
        calls.append((name, subset, kwargs))
        return FakeHuggingFaceDataset(
            [{"text": "zero"}, {"text": "one"}, {"text": "two"}, {"text": "three"}]
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    stream = HuggingFaceTextStream(
        dataset_name="Skylion007/openwebtext",
        skip_texts=1,
        limit_texts=2,
    )
    assert list(stream) == ["one", "two"]
    assert calls == [
        (
            "Skylion007/openwebtext",
            None,
            {"split": "train", "streaming": True},
        )
    ]


def test_teacher_forced_extractor_drops_first_real_token_and_padding():
    extractor = TeacherForcedHiddenExtractor(
        FakeSourceModel(),
        FakeTokenizer(),
        layer_index=0,
        layer_path="h",
        max_length=8,
        device=torch.device("cpu"),
    )
    chunks = extractor(["long", "short", "single"])
    torch.testing.assert_close(chunks[0], torch.tensor([[2.0], [3.0]]))
    torch.testing.assert_close(chunks[1], torch.tensor([[5.0]]))
    assert chunks[2].shape == (0, 1)


def test_iterable_dataset_packs_across_texts_into_exact_batches():
    dataset = TeacherForcedActivationIterableDataset(
        ["1,2", "3", "4,5,6", "7,8"],
        ParsedTextExtractor(),
        batch_size=4,
        text_batch_size=2,
        max_batches=2,
    )
    batches = list(dataset)
    assert len(dataset) == 2
    assert [tuple(batch.shape) for batch in batches] == [(4, 1), (4, 1)]
    torch.testing.assert_close(batches[0].flatten(), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(batches[1].flatten(), torch.tensor([5.0, 6.0, 7.0, 8.0]))


def test_iterable_dataset_drops_only_final_incomplete_batch():
    dataset = TeacherForcedActivationIterableDataset(
        ["1,2,3"],
        ParsedTextExtractor(),
        batch_size=2,
        text_batch_size=1,
    )
    batches = list(dataset)
    assert len(batches) == 1
    torch.testing.assert_close(batches[0].flatten(), torch.tensor([1.0, 2.0]))
