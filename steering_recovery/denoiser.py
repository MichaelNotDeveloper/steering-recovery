from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from steering_recovery.normalization import ActivationNormalizer


class ResidualBlock(nn.Module):
    """Unnormalised residual MLP: Linear -> GELU -> Linear -> Dropout."""

    def __init__(
        self, hidden_size: int, latent_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        if hidden_size <= 0 or latent_dim <= 0:
            raise ValueError("hidden_size and latent_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.network = nn.Sequential(
            nn.Linear(hidden_size, latent_dim, bias=True),
            nn.GELU(),
            nn.Linear(latent_dim, hidden_size, bias=True),
            nn.Dropout(float(dropout)),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.network(hidden)


@dataclass(frozen=True)
class DenoiserConfig:
    hidden_size: int
    latent_dim: int = 768
    num_layers: int = 3
    dropout: float = 0.0


class ActivationDenoiser(nn.Module):
    """Map a noisy normalized activation directly to its clean value."""

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int = 768,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_size <= 0 or latent_dim <= 0 or num_layers <= 0:
            raise ValueError("model dimensions and num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.config = DenoiserConfig(
            hidden_size=int(hidden_size),
            latent_dim=int(latent_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_size, latent_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
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
        self,
        steered_activations: torch.Tensor,
        raw_delta: torch.Tensor | None = None,
        *,
        mode: str = "full",
    ) -> torch.Tensor:
        """Denoise fully or remove the score component parallel to steering.

        The model is trained in feature-normalized coordinates, where Tweedie's
        displacement is ``D(x) - x = sigma^2 * score(x)``.  Orthogonalization
        is therefore performed in that same coordinate system.
        """

        if mode not in {"full", "orthogonal"}:
            raise ValueError("mode must be 'full' or 'orthogonal'")
        normalized = self.normalizer.normalize(steered_activations)
        recovered = self.model(normalized)
        if mode == "full":
            return self.normalizer.denormalize(recovered)
        if raw_delta is None:
            raise ValueError("raw_delta is required for orthogonal denoising")
        normalized_delta = self.normalizer.normalize_delta(raw_delta)
        squared_norm = normalized_delta.square().sum(dim=-1, keepdim=True)
        if torch.any(squared_norm <= 0):
            raise ValueError("steering delta must have non-zero norm")
        score_displacement = recovered - normalized
        parallel = (
            (score_displacement * normalized_delta).sum(dim=-1, keepdim=True)
            / squared_norm
        ) * normalized_delta
        orthogonal_recovered = normalized + score_displacement - parallel
        return self.normalizer.denormalize(orthogonal_recovered)

    @property
    def model_config(self) -> dict[str, Any]:
        return asdict(self.model.config)
