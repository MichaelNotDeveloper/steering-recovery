from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class Tracker:
    def __init__(self, run: Any | None = None):
        self.run = run

    @classmethod
    def create(
        cls,
        *,
        enabled: bool,
        project: str,
        name: str | None,
        entity: str | None,
        mode: str,
        config: Mapping[str, Any],
        tags: list[str] | None = None,
    ) -> "Tracker":
        if not enabled:
            return cls()
        try:
            import wandb
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "W&B logging is enabled, but wandb cannot be imported. Reinstall the project environment."
            ) from error
        run = wandb.init(
            project=project,
            name=name,
            entity=entity,
            mode=mode,
            config=dict(config),
            tags=tags,
        )
        return cls(run)

    def log(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(dict(metrics), step=step)

    def save(self, path: str | Path) -> None:
        if self.run is not None:
            self.run.save(str(path), policy="now")

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
