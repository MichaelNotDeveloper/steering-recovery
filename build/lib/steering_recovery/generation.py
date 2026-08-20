from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from steering_recovery.intervention import (
    ActivationIntervention,
    InterventionController,
)


@dataclass(frozen=True)
class GenerationTrace:
    text: str
    token_ids: list[int]
    normalized_entropies: list[float]
    intervention_steps: int
    forward_calls: int


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] < 2:
        raise ValueError("entropy requires at least two vocabulary entries")
    probabilities = torch.softmax(logits.float(), dim=-1)
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    return entropy / math.log(logits.shape[-1])


def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0:
        return logits.argmax(dim=-1)
    scaled = logits.float() / temperature
    if top_p < 1:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        scaled = torch.full_like(scaled, -torch.inf).scatter(
            -1, sorted_indices, sorted_logits
        )
    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(
        -1
    )


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.inference_mode()
def generate_with_intervention(
    model: torch.nn.Module,
    tokenizer: object,
    prompt: str,
    intervention: ActivationIntervention,
    controller: InterventionController,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> GenerationTrace:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    encoded = tokenizer(prompt, return_tensors="pt")
    device = _model_device(model)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
        device
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    generated: list[int] = []
    entropies: list[float] = []
    past_key_values = None
    current_ids = input_ids

    with intervention:
        for _ in range(max_new_tokens):
            kwargs = {
                "input_ids": current_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            output = model(**kwargs)
            logits = output.logits[:, -1, :]
            entropy = float(normalized_entropy(logits)[0].item())
            entropies.append(entropy)
            controller.observe_entropy(entropy)
            next_token = sample_token(
                logits,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            token_id = int(next_token[0].item())
            generated.append(token_id)
            past_key_values = getattr(output, "past_key_values", None)
            if token_id == getattr(tokenizer, "eos_token_id", None):
                break
            current_ids = next_token[:, None]
            attention_mask = torch.cat(
                (attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))),
                dim=1,
            )

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return GenerationTrace(
        text=text,
        token_ids=generated,
        normalized_entropies=entropies,
        intervention_steps=controller.state.intervention_calls,
        forward_calls=controller.state.forward_calls,
    )
