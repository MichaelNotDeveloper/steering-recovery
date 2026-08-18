from types import SimpleNamespace

import pytest
import torch
from torch import nn

from steering_recovery.intervention import (
    ActivationIntervention,
    InterventionController,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity()])

    def forward(self, value):
        return self.model.layers[0](value)


def test_once_at_start_only_edits_first_forward():
    model = ToyModel()
    controller = InterventionController("once_at_start", scale=2)
    hook = ActivationIntervention(
        model, torch.ones(4), layer_index=0, controller=controller
    )
    value = torch.zeros(1, 3, 4)
    with hook:
        first = model(value)
        second = model(value)
    torch.testing.assert_close(first[:, :-1], torch.zeros(1, 2, 4))
    torch.testing.assert_close(first[:, -1], torch.ones(1, 4) * 2)
    torch.testing.assert_close(second, value)
    assert controller.state.intervention_calls == 1


def test_entropy_policy_is_causal():
    controller = InterventionController(
        "entropy_threshold", scale=1, entropy_threshold=0.4
    )
    assert not controller.should_apply()
    controller.observe_entropy(0.5)
    assert controller.should_apply()
    controller.observe_entropy(0.2)
    assert not controller.should_apply()


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        InterventionController("sometimes", scale=1)

