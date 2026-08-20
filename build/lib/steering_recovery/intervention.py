from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from steering_recovery.denoiser import DenoiserBundle
from steering_recovery.layers import first_tensor, replace_first_tensor, resolve_layer


VALID_MODES = {"none", "once_at_start", "every_step", "entropy_threshold"}


@dataclass
class InterventionState:
    forward_calls: int = 0
    intervention_calls: int = 0
    last_entropy: float | None = None


class InterventionController:
    """Stateful, causal policy for deciding when to edit a hidden state."""

    def __init__(
        self,
        mode: str,
        scale: float,
        entropy_threshold: float = 0.35,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        if not 0 <= entropy_threshold <= 1:
            raise ValueError("entropy_threshold must be in [0, 1]")
        self.mode = mode
        self.scale = float(scale)
        self.entropy_threshold = float(entropy_threshold)
        self.state = InterventionState()

    def should_apply(self) -> bool:
        call = self.state.forward_calls
        self.state.forward_calls += 1
        if self.scale == 0 or self.mode == "none":
            return False
        if self.mode == "once_at_start":
            decision = call == 0
        elif self.mode == "every_step":
            decision = True
        else:
            decision = (
                self.state.last_entropy is not None
                and self.state.last_entropy >= self.entropy_threshold
            )
        if decision:
            self.state.intervention_calls += 1
        return decision

    def observe_entropy(self, normalized_entropy: float) -> None:
        if not 0 <= normalized_entropy <= 1 + 1e-6:
            raise ValueError("normalized entropy must be in [0, 1]")
        self.state.last_entropy = float(normalized_entropy)


class ActivationIntervention:
    """Forward hook adding a steering vector to the last sequence position."""

    def __init__(
        self,
        model: nn.Module,
        steering_vector: torch.Tensor,
        *,
        layer_index: int,
        controller: InterventionController,
        layer_path: str | None = None,
        denoiser: DenoiserBundle | None = None,
    ):
        vector = torch.as_tensor(steering_vector).flatten()
        if vector.ndim != 1 or vector.numel() == 0:
            raise ValueError("steering_vector must be a non-empty vector")
        self.model = model
        self.vector = vector
        self.layer_index = int(layer_index)
        self.layer_path = layer_path
        self.controller = controller
        self.denoiser = denoiser
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, _module: nn.Module, _inputs: tuple[object, ...], output: object):
        activations = first_tensor(output)
        if activations is None or not self.controller.should_apply():
            return output
        if activations.shape[-1] != self.vector.numel():
            raise ValueError(
                f"steering vector has {self.vector.numel()} values, "
                f"layer output has {activations.shape[-1]}"
            )
        delta = self.controller.scale * self.vector.to(activations)
        edited = activations.clone()
        if edited.ndim == 2:
            target = edited
            raw_delta = delta.expand_as(target)
            target = target + raw_delta
            if self.denoiser is not None:
                target = self._run_denoiser(target, raw_delta)
            edited = target
        elif edited.ndim == 3:
            target = edited[:, -1, :]
            raw_delta = delta.expand_as(target)
            target = target + raw_delta
            if self.denoiser is not None:
                target = self._run_denoiser(target, raw_delta)
            edited[:, -1, :] = target
        else:
            raise ValueError(f"unsupported layer output shape {activations.shape}")
        return replace_first_tensor(output, edited)

    def _run_denoiser(
        self, target: torch.Tensor, raw_delta: torch.Tensor
    ) -> torch.Tensor:
        assert self.denoiser is not None
        parameter = next(self.denoiser.model.parameters())
        denoiser_target = target.to(device=parameter.device, dtype=parameter.dtype)
        denoiser_delta = raw_delta.to(device=parameter.device, dtype=parameter.dtype)
        recovered = self.denoiser.denoise_steered(denoiser_target, denoiser_delta)
        return recovered.to(target)

    def __enter__(self) -> "ActivationIntervention":
        layer = resolve_layer(self.model, self.layer_index, self.layer_path)
        self._handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.remove()

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
