from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch import nn

from steering_recovery.baseline import run_generation_records


class Tokenizer:
    eos_token_id = None

    def __call__(self, _text, return_tensors):
        return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.ones(1, 1)}

    def decode(self, values, skip_special_tokens):
        return "".join(str(value) for value in values)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 3)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity()])
        self.head = nn.Linear(3, 4)

    def forward(self, input_ids, attention_mask, use_cache, past_key_values=None):
        hidden = self.model.layers[0](self.embedding(input_ids))
        return SimpleNamespace(logits=self.head(hidden), past_key_values=())


def test_run_generation_records_produces_serializable_rows():
    config = OmegaConf.create(
        {
            "generation": {
                "num_samples": 2,
                "seed": 5,
                "max_new_tokens": 2,
                "temperature": 0,
                "top_p": 1,
            },
            "steering": {
                "mode": "once_at_start",
                "scale": 0,
                "entropy_threshold": 0.35,
                "layer_index": 0,
                "layer_path": "model.layers",
            },
        }
    )
    rows = run_generation_records(
        model=Model(),
        tokenizer=Tokenizer(),
        records=[{"id": "one", "prompt": "test"}],
        steering_vector=torch.zeros(3),
        config=config,
    )
    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {5, 6}
    assert all(row["intervention_steps"] == 0 for row in rows)
