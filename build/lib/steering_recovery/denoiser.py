from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from steering_recovery.normalization import ActivationNormalizer


class ResidualBlock(nn.Module):
    """Unnormalised residual MLP: Linear -> GELU -> Linear."""

    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        if hidden_size <= 0 or latent_dim <= 0:
            raise ValueError("hidden_size and latent_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(hidden_size, latent_dim, bias=True),
            nn.GELU(),
            nn.Linear(latent_dim, hidden_size, bias=True),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.network(hidden)


@dataclass(frozen=True)
class DenoiserConfig:
    hidden_size: int
    latent_dim: int = 768
    num_layers: int = 3


class ActivationDenoiser(nn.Module):
    """Map a noisy normalized activation directly to its clean value."""

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int = 768,
        num_layers: int = 3,
    ):
        super().__init__()
        if hidden_size <= 0 or latent_dim <= 0 or num_layers <= 0:
            raise ValueError("model dimensions and num_layers must be positive")
        self.config = DenoiserConfig(
            hidden_size=int(hidden_size),
            latent_dim=int(latent_dim),
            num_layers=int(num_layers),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, latent_dim) for _ in range(num_layers)]
        )

    def forward(self, noisy_activations: torch.Tensor) -> torch.Tensor:
        if noisy_activations.ndim not in {2, 3}:
            raise ValueError(
                "noisy_activations must have shape [batch, hidden] or "
                "[batch, sequence, hidden]"
            )
        if noisy_activations.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.hidden_size}, "
                f"got {noisy_activations.shape[-1]}"
            )
        hidden = noisy_activations
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class DenoiserBundle(nn.Module):
    """Denoiser plus the exact activation statistics used during training."""

    def __init__(self, model: ActivationDenoiser, normalizer: ActivationNormalizer):
        super().__init__()
        if model.config.hidden_size != normalizer.hidden_size:
            raise ValueError("model and normalizer hidden sizes differ")
        self.model = model
        self.normalizer = normalizer

    @torch.no_grad()
    def denoise(
        self,
        activations: torch.Tensor,
        _normalized_noise_level: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        normalized = self.normalizer.normalize(activations)
        recovered = self.model(normalized)
        return self.normalizer.denormalize(recovered)

    @torch.no_grad()
    def denoise_steered(
        self, steered_activations: torch.Tensor, _raw_delta: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.denoise(steered_activations)

    @property
    def model_config(self) -> dict[str, Any]:
        return asdict(self.model.config)
