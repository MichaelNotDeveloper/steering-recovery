from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CorruptionBatch:
    noisy: torch.Tensor
    corruption: torch.Tensor
    noise_level: torch.Tensor
    gaussian_sigma: torch.Tensor
    used_steering: torch.Tensor


class OnlineCorruptor:
    """Create Gaussian and direction-aligned corruptions in normalized space."""

    def __init__(
        self,
        gaussian_std_min: float,
        gaussian_std_max: float,
        steering_probability: float = 0.0,
        steering_scale_min: float = 0.0,
        steering_scale_max: float = 0.0,
        identity_probability: float = 0.0,
        bidirectional: bool = True,
        steering_vectors: torch.Tensor | None = None,
    ):
        if not 0 <= gaussian_std_min <= gaussian_std_max:
            raise ValueError("Gaussian std bounds must satisfy 0 <= min <= max")
        if not 0 <= steering_scale_min <= steering_scale_max:
            raise ValueError("steering scale bounds must satisfy 0 <= min <= max")
        for name, value in {
            "steering_probability": steering_probability,
            "identity_probability": identity_probability,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        self.gaussian_std_min = float(gaussian_std_min)
        self.gaussian_std_max = float(gaussian_std_max)
        self.steering_probability = float(steering_probability)
        self.steering_scale_min = float(steering_scale_min)
        self.steering_scale_max = float(steering_scale_max)
        self.identity_probability = float(identity_probability)
        self.bidirectional = bool(bidirectional)
        self.steering_vectors = self._prepare_vectors(steering_vectors)

    @staticmethod
    def _prepare_vectors(vectors: torch.Tensor | None) -> torch.Tensor | None:
        if vectors is None:
            return None
        vectors = torch.as_tensor(vectors, dtype=torch.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError(
                "steering_vectors must have shape [n_vectors, hidden_size]"
            )
        rms = vectors.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        return vectors / rms

    def __call__(
        self, clean: torch.Tensor, generator: torch.Generator | None = None
    ) -> CorruptionBatch:
        if clean.ndim != 2:
            raise ValueError(
                f"expected clean activations [batch, hidden], got {clean.shape}"
            )
        batch_size = clean.shape[0]
        device, dtype = clean.device, clean.dtype
        sigma = self._uniform(
            batch_size, self.gaussian_std_min, self.gaussian_std_max, device, generator
        ).to(dtype)
        gaussian = (
            torch.randn(clean.shape, device=device, dtype=dtype, generator=generator)
            * sigma[:, None]
        )
        corruption = gaussian

        used_steering = torch.zeros(batch_size, device=device, dtype=torch.bool)
        if self.steering_probability > 0 and self.steering_vectors is not None:
            vectors = self.steering_vectors.to(device=device, dtype=dtype)
            used_steering = (
                torch.rand(batch_size, device=device, generator=generator)
                < self.steering_probability
            )
            indices = torch.randint(
                vectors.shape[0], (batch_size,), device=device, generator=generator
            )
            scales = self._uniform(
                batch_size,
                self.steering_scale_min,
                self.steering_scale_max,
                device,
                generator,
            ).to(dtype)
            if self.bidirectional:
                signs = (
                    torch.randint(
                        0, 2, (batch_size,), device=device, generator=generator
                    )
                    .mul(2)
                    .sub(1)
                )
                scales = scales * signs.to(dtype)
            steering = vectors[indices] * scales[:, None]
            corruption = corruption + steering * used_steering[:, None]

        if self.identity_probability > 0:
            identity = (
                torch.rand(batch_size, device=device, generator=generator)
                < self.identity_probability
            )
            corruption = corruption.masked_fill(identity[:, None], 0)
            sigma = sigma.masked_fill(identity, 0)
            used_steering = used_steering & ~identity

        level = corruption.float().square().mean(dim=-1).sqrt().to(dtype)
        return CorruptionBatch(
            noisy=clean + corruption,
            corruption=corruption,
            noise_level=level,
            gaussian_sigma=sigma,
            used_steering=used_steering,
        )

    @staticmethod
    def _uniform(
        size: int,
        low: float,
        high: float,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if low == high:
            return torch.full((size,), low, device=device)
        return low + (high - low) * torch.rand(size, device=device, generator=generator)
