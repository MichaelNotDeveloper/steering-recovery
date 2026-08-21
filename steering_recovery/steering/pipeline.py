from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from steering_recovery.steering.ag_news import generate_ag_news_vectors


def generate_steering_vectors(config: DictConfig) -> dict[str, Any]:
    """Dispatch a configured steering-vector generator."""

    generator = str(config.generator)
    if generator == "ag_news":
        return generate_ag_news_vectors(config)
    raise ValueError(f"unknown steering-vector generator {generator!r}")
