from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from steering_recovery.denoiser import DenoiserBundle
from steering_recovery.generation import normalized_entropy, sample_token
from steering_recovery.intervention import (
    ActivationIntervention,
    InterventionController,
)


@dataclass(frozen=True)
class Continuation:
    prompt_text: str
    generated_text: str
    full_text: str
    generated_token_ids: list[int]
    intervention_steps: int
    denoiser_calls: int
    forward_calls: int


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.inference_mode()
def generate_steered_continuation(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_token_ids: Sequence[int],
    steering_vector: torch.Tensor,
    *,
    alpha: float,
    layer_index: int,
    layer_path: str | None,
    intervention_mode: str,
    entropy_threshold: float,
    denoiser: DenoiserBundle | None,
    denoising_mode: str = "full",
    beta: int = 1,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    stop_on_eos: bool,
) -> Continuation:
    """Generate from exact prompt IDs under ``h <- h + alpha * v``."""

    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must be non-empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    device = _model_device(model)
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    generator = torch.Generator(device=device).manual_seed(seed)
    controller = InterventionController(
        mode=intervention_mode,
        scale=float(alpha),
        entropy_threshold=float(entropy_threshold),
    )
    intervention = ActivationIntervention(
        model,
        steering_vector,
        layer_index=layer_index,
        layer_path=layer_path,
        controller=controller,
        denoiser=denoiser,
        denoising_mode=denoising_mode,
        beta=beta,
    )
    generated: list[int] = []
    past_key_values = None
    current_ids = input_ids
    with intervention:
        for _ in range(max_new_tokens):
            kwargs: dict[str, Any] = {
                "input_ids": current_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            output = model(**kwargs)
            logits = output.logits[:, -1, :]
            if intervention_mode == "entropy_threshold":
                controller.observe_entropy(float(normalized_entropy(logits)[0].item()))
            next_token = sample_token(
                logits,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
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

    prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    full_text = tokenizer.decode(
        [*prompt_token_ids, *generated], skip_special_tokens=True
    )
    return Continuation(
        prompt_text=prompt_text,
        generated_text=generated_text,
        full_text=full_text,
        generated_token_ids=generated,
        intervention_steps=controller.state.intervention_calls,
        denoiser_calls=intervention.denoiser_calls,
        forward_calls=controller.state.forward_calls,
    )
