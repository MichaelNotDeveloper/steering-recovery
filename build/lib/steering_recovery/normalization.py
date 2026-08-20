from __future__ import annotations

import torch
from torch import nn


class ActivationNormalizer(nn.Module):
    """Feature-wise standardization for hidden states and steering deltas."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(std, dtype=torch.float32).flatten()
        if mean.shape != std.shape:
            raise ValueError("mean and std must have identical shapes")
        if mean.numel() == 0:
            raise ValueError("normalization statistics cannot be empty")
        if torch.any(std < 0):
            raise ValueError("std must be non-negative")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(eps))
        self.eps = float(eps)

    @property
    def hidden_size(self) -> int:
        return self.mean.numel()

    def normalize(self, activations: torch.Tensor) -> torch.Tensor:
        self._check_last_dimension(activations)
        return (activations - self.mean.to(activations)) / self.std.to(activations)

    def denormalize(self, activations: torch.Tensor) -> torch.Tensor:
        self._check_last_dimension(activations)
        return activations * self.std.to(activations) + self.mean.to(activations)

    def normalize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        self._check_last_dimension(delta)
        return delta / self.std.to(delta)

    def _check_last_dimension(self, value: torch.Tensor) -> None:
        if value.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {value.shape[-1]}"
            )
