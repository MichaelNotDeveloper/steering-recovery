from types import SimpleNamespace

import torch
from torch import nn

from steering_recovery.generation import (
    generate_with_intervention,
    normalized_entropy,
    sample_token,
)
from steering_recovery.intervention import (
    ActivationIntervention,
    InterventionController,
)


class ToyTokenizer:
    eos_token_id = 5

    def __call__(self, _text, return_tensors):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}

    def decode(self, ids, skip_special_tokens):
        return " ".join(map(str, ids))


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(6, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity()])
        self.head = nn.Linear(4, 6, bias=False)

    def forward(self, input_ids, attention_mask, use_cache, past_key_values=None):
        hidden = self.model.layers[0](self.embedding(input_ids))
        return SimpleNamespace(logits=self.head(hidden), past_key_values=())


def test_entropy_and_sampling_contracts():
    uniform = torch.zeros(2, 4)
    torch.testing.assert_close(normalized_entropy(uniform), torch.ones(2))
    logits = torch.tensor([[0.0, 3.0, 1.0]])
    assert sample_token(logits, temperature=0, top_p=1).item() == 1


def test_generation_runs_with_kv_style_loop_and_hook():
    model = ToyLM()
    controller = InterventionController("once_at_start", scale=1)
    intervention = ActivationIntervention(
        model, torch.ones(4), layer_index=0, controller=controller
    )
    trace = generate_with_intervention(
        model,
        ToyTokenizer(),
        "prompt",
        intervention,
        controller,
        max_new_tokens=3,
        temperature=0,
        top_p=1,
        seed=4,
    )
    assert len(trace.token_ids) <= 3
    assert len(trace.normalized_entropies) == len(trace.token_ids)
    assert trace.intervention_steps == 1
