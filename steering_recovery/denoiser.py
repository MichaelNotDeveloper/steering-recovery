from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from steering_recovery.normalization import ActivationNormalizer


def sinusoidal_embedding(values: torch.Tensor, dimension: int) -> torch.Tensor:
    if dimension < 2:
        raise ValueError("embedding dimension must be at least 2")
    values = values.float().flatten()
    half = dimension // 2
    denominator = max(half - 1, 1)
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(half, device=values.device) / denominator
    )
    angles = values[:, None] * frequencies[None]
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if dimension % 2:
        embedding = torch.cat((embedding, embedding.new_zeros(len(values), 1)), dim=-1)
    return embedding


class ConditionedResidualBlock(nn.Module):
    def __init__(self, width: int, expansion: int, dropout: float):
        super().__init__()
        inner = width * expansion
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, width * 3)
        self.up = nn.Linear(width, inner * 2)
        self.down = nn.Linear(inner, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale, residual_gate = self.modulation(condition).chunk(3, dim=-1)
        value = self.norm(hidden) * (1 + scale) + shift
        value, gate = self.up(value).chunk(2, dim=-1)
        value = self.down(torch.nn.functional.silu(gate) * value)
        return hidden + torch.sigmoid(residual_gate) * self.dropout(value)


@dataclass(frozen=True)
class DenoiserConfig:
    hidden_size: int
    width: int = 1024
    depth: int = 4
    expansion: int = 2
    dropout: float = 0.0


class ActivationDenoiser(nn.Module):
    """Predict an additive corruption from a noisy hidden state."""

    def __init__(
        self,
        hidden_size: int,
        width: int = 1024,
        depth: int = 4,
        expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min(hidden_size, width, depth, expansion) <= 0:
            raise ValueError("model dimensions and depth must be positive")
        self.config = DenoiserConfig(
            hidden_size=hidden_size,
            width=width,
            depth=depth,
            expansion=expansion,
            dropout=dropout,
        )
        self.input_projection = nn.Linear(hidden_size, width)
        self.noise_embedding = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            [ConditionedResidualBlock(width, expansion, dropout) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, hidden_size)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self, noisy_activations: torch.Tensor, noise_level: torch.Tensor | float
    ) -> torch.Tensor:
        if noisy_activations.ndim not in {2, 3}:
            raise ValueError(
                "noisy_activations must have shape [batch, hidden] or [batch, seq, hidden]"
            )
        if noisy_activations.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.hidden_size}, "
                f"got {noisy_activations.shape[-1]}"
            )
        original_shape = noisy_activations.shape
        flat = noisy_activations.reshape(-1, original_shape[-1])
        levels = self._expand_levels(noise_level, original_shape, flat.device)
        log_levels = torch.log(levels.float().clamp_min(1e-6))
        condition = sinusoidal_embedding(log_levels, self.config.width).to(flat.dtype)
        condition = self.noise_embedding(condition)
        hidden = self.input_projection(flat)
        for block in self.blocks:
            hidden = block(hidden, condition)
        predicted = self.output_projection(self.output_norm(hidden))
        return predicted.reshape(original_shape)

    @staticmethod
    def _expand_levels(
        noise_level: torch.Tensor | float,
        activation_shape: torch.Size,
        device: torch.device,
    ) -> torch.Tensor:
        levels = torch.as_tensor(noise_level, device=device).flatten()
        flat_size = math.prod(activation_shape[:-1])
        if levels.numel() == 1:
            return levels.expand(flat_size)
        if levels.numel() == flat_size:
            return levels
        if len(activation_shape) == 3 and levels.numel() == activation_shape[0]:
            return levels.repeat_interleave(activation_shape[1])
        raise ValueError(
            f"noise_level has {levels.numel()} values for activations {tuple(activation_shape)}"
        )


class DenoiserBundle(nn.Module):
    """Denoiser plus the exact statistics used during training."""

    def __init__(self, model: ActivationDenoiser, normalizer: ActivationNormalizer):
        super().__init__()
        if model.config.hidden_size != normalizer.hidden_size:
            raise ValueError("model and normalizer hidden sizes differ")
        self.model = model
        self.normalizer = normalizer

    @torch.no_grad()
    def denoise(
        self, activations: torch.Tensor, normalized_noise_level: torch.Tensor | float
    ) -> torch.Tensor:
        normalized = self.normalizer.normalize(activations)
        predicted_corruption = self.model(normalized, normalized_noise_level)
        return self.normalizer.denormalize(normalized - predicted_corruption)

    @torch.no_grad()
    def denoise_steered(
        self, steered_activations: torch.Tensor, raw_delta: torch.Tensor
    ) -> torch.Tensor:
        normalized_delta = self.normalizer.normalize_delta(raw_delta)
        level = normalized_delta.float().square().mean(dim=-1).sqrt()
        return self.denoise(steered_activations, level)

    @property
    def model_config(self) -> dict[str, Any]:
        return asdict(self.model.config)
