from __future__ import annotations

import torch
from torch import nn


DEFAULT_LAYER_PATHS = (
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "encoder.layer",
    "block",
)


def resolve_object(root: object, dotted_path: str) -> object:
    current = root
    for component in dotted_path.split("."):
        if not hasattr(current, component):
            raise AttributeError(
                f"{type(current).__name__} has no attribute {component!r}"
            )
        current = getattr(current, component)
    return current


def resolve_layer(
    model: nn.Module, layer_index: int, layer_path: str | None = None
) -> nn.Module:
    candidates = (layer_path,) if layer_path else DEFAULT_LAYER_PATHS
    failures: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            layers = resolve_object(model, candidate)
            layer = layers[layer_index]  # type: ignore[index]
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            failures.append(f"{candidate}: {error}")
            continue
        if not isinstance(layer, nn.Module):
            failures.append(f"{candidate}[{layer_index}] is not a torch module")
            continue
        return layer
    details = "; ".join(failures)
    raise ValueError(f"could not resolve transformer layer {layer_index}. {details}")


def first_tensor(output: object) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    return None


def replace_first_tensor(output: object, value: torch.Tensor) -> object:
    if torch.is_tensor(output):
        return value
    if isinstance(output, tuple):
        return (value, *output[1:])
    if isinstance(output, list):
        return [value, *output[1:]]
    raise TypeError(f"cannot replace tensor in {type(output).__name__}")
