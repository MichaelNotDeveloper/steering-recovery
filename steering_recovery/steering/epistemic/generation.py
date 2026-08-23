from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from torch import nn

from steering_recovery.denoiser import DenoiserBundle
from steering_recovery.generation import sample_token
from steering_recovery.layers import first_tensor, replace_first_tensor, resolve_layer
from steering_recovery.steering.epistemic.statistics import (
    denoising_geometry_statistics,
    mc_dropout_statistics,
)


@dataclass(frozen=True)
class EpistemicContinuation:
    prompt_text: str
    generated_text: str
    full_text: str
    generated_token_ids: list[int]
    token_statistics: list[dict[str, float | int]]
    forward_calls: int


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@contextmanager
def mc_dropout_enabled(model: nn.Module) -> Iterator[None]:
    """Enable Dropout modules while leaving the rest of the model in eval mode."""

    states = {module: module.training for module in model.modules()}
    model.eval()
    dropout_modules = [
        module for module in model.modules() if isinstance(module, nn.Dropout)
    ]
    if not dropout_modules:
        raise ValueError("MC-dropout inference requires at least one Dropout module")
    for module in dropout_modules:
        module.train()
    try:
        yield
    finally:
        for module, training in states.items():
            module.train(training)


class EpistemicActivationIntervention:
    """Steer the last hidden, run MC dropout and inject the predictive mean."""

    def __init__(
        self,
        model: nn.Module,
        denoiser: DenoiserBundle,
        steering_vector: torch.Tensor,
        *,
        alpha: float,
        layer_index: int,
        layer_path: str | None,
        mc_samples: int,
    ):
        if mc_samples < 2:
            raise ValueError("mc_samples must be at least two")
        vector = torch.as_tensor(steering_vector).float().flatten()
        if not vector.numel():
            raise ValueError("steering_vector must not be empty")
        if float(torch.linalg.vector_norm(vector)) == 0:
            raise ValueError("steering_vector must have non-zero norm")
        self.model = model
        self.denoiser = denoiser
        self.vector = vector
        self.alpha = float(alpha)
        self.layer_index = int(layer_index)
        self.layer_path = layer_path
        self.mc_samples = int(mc_samples)
        self.statistics: list[dict[str, float | int]] = []
        self.forward_calls = 0
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, _module: nn.Module, _inputs: tuple[object, ...], output: object):
        activations = first_tensor(output)
        if activations is None:
            return output
        if activations.shape[-1] != self.vector.numel():
            raise ValueError("steering vector and layer hidden sizes differ")
        self.forward_calls += 1
        edited = activations.clone()
        target = edited if edited.ndim == 2 else edited[:, -1, :]
        if target.ndim != 2:
            raise ValueError(f"unsupported layer output shape {activations.shape}")
        if len(target) != 1:
            raise ValueError("epistemic generation currently requires batch size one")
        steered = target + self.alpha * self.vector.to(target).unsqueeze(0)

        parameter = next(self.denoiser.model.parameters())
        denoiser_input = steered.to(device=parameter.device, dtype=parameter.dtype)
        normalized = self.denoiser.normalizer.normalize(denoiser_input)[0]
        predictions = self.denoiser.model(
            normalized.unsqueeze(0).expand(self.mc_samples, -1)
        )
        metrics = mc_dropout_statistics(normalized, predictions)
        recovered = self.denoiser.normalizer.denormalize(
            predictions.mean(dim=0, keepdim=True)
        ).to(target)
        metrics.update(
            denoising_geometry_statistics(
                target[0],
                steered[0],
                recovered[0],
                self.vector.to(target),
            )
        )
        self.statistics.append({"step": len(self.statistics), **metrics})
        if edited.ndim == 2:
            edited = recovered
        else:
            edited[:, -1, :] = recovered
        return replace_first_tensor(output, edited)

    def __enter__(self) -> "EpistemicActivationIntervention":
        layer = resolve_layer(self.model, self.layer_index, self.layer_path)
        self._handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(tokenizer.decode(token_ids, skip_special_tokens=True))


@torch.inference_mode()
def generate_epistemic_continuation(
    model: nn.Module,
    tokenizer: Any,
    prompt_token_ids: Sequence[int],
    steering_vector: torch.Tensor,
    denoiser: DenoiserBundle,
    *,
    alpha: float,
    layer_index: int,
    layer_path: str | None,
    mc_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    generation_seed: int,
    dropout_seed: int,
    stop_on_eos: bool,
) -> EpistemicContinuation:
    """Generate with an MC-dropout predictive mean at every steered step."""

    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must be non-empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    device = _model_device(model)
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    sampling_generator = torch.Generator(device=device).manual_seed(generation_seed)
    intervention = EpistemicActivationIntervention(
        model,
        denoiser,
        steering_vector,
        alpha=alpha,
        layer_index=layer_index,
        layer_path=layer_path,
        mc_samples=mc_samples,
    )
    generated: list[int] = []
    past_key_values = None
    current_ids = input_ids
    fork_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(dropout_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(dropout_seed)
        with mc_dropout_enabled(denoiser.model), intervention:
            for _ in range(max_new_tokens):
                kwargs: dict[str, Any] = {
                    "input_ids": current_ids,
                    "attention_mask": attention_mask,
                    "use_cache": True,
                }
                if past_key_values is not None:
                    kwargs["past_key_values"] = past_key_values
                output = model(**kwargs)
                next_token = sample_token(
                    output.logits[:, -1, :],
                    temperature=temperature,
                    top_p=top_p,
                    generator=sampling_generator,
                )
                token_id = int(next_token[0])
                generated.append(token_id)
                past_key_values = getattr(output, "past_key_values", None)
                if stop_on_eos and token_id == getattr(tokenizer, "eos_token_id", None):
                    break
                current_ids = next_token[:, None]
                attention_mask = torch.cat(
                    (attention_mask, attention_mask.new_ones((1, 1))), dim=1
                )

    if len(intervention.statistics) != len(generated):
        raise RuntimeError("each generated token must have one epistemic statistic")
    token_statistics = [
        {**statistics, "token_id": token_id}
        for token_id, statistics in zip(generated, intervention.statistics)
    ]
    return EpistemicContinuation(
        prompt_text=_decode(tokenizer, prompt_token_ids),
        generated_text=_decode(tokenizer, generated),
        full_text=_decode(tokenizer, [*prompt_token_ids, *generated]),
        generated_token_ids=generated,
        token_statistics=token_statistics,
        forward_calls=intervention.forward_calls,
    )
