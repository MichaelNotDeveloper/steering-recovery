import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from steering_recovery.checkpoint import load_checkpoint
from steering_recovery.training import train_denoiser


def test_cpu_training_smoke(tmp_path):
    data_path = tmp_path / "activations.npy"
    values = np.random.default_rng(4).normal(size=(24, 6)).astype("float32")
    np.save(data_path, values)
    statistics_path = tmp_path / "statistics.pt"
    torch.save(
        {
            "sum": torch.from_numpy(values).double().sum(dim=0),
            "variance": torch.from_numpy(values).double().var(dim=0, correction=0),
            "count": len(values),
        },
        statistics_path,
    )
    config = OmegaConf.create(
        {
            "seed": 3,
            "device": "cpu",
            "data": {
                "path": str(data_path),
                "key": "activations",
                "statistics_path": str(statistics_path),
                "val_fraction": 0.25,
                "num_workers": 0,
            },
            "model": {
                "hidden_size": 6,
                "latent_dims": [4, 8],
                "num_layers": [1],
                "sigmas": [0.1, 0.2],
            },
            "training": {
                "epochs": 1,
                "max_steps": 2,
                "batch_size": 4,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "betas": [0.9, 0.95],
                "max_grad_norm": 1.0,
                "precision": "fp32",
                "log_every_batches": 1,
                "validation_every_batches": 1,
                "validation_batches": 1,
            },
            "wandb": {
                "enabled": False,
                "project": "test",
                "name": None,
                "entity": None,
                "mode": "offline",
                "tags": [],
            },
        }
    )
    output = tmp_path / "run"
    result = train_denoiser(config, output)
    assert result["steps"] == 2
    assert result["models"] == 4
    model_directories = sorted((output / "models").iterdir())
    assert len(model_directories) == 4
    for directory in model_directories:
        bundle, metadata = load_checkpoint(directory / "best.pt")
        assert bundle.model.config.hidden_size == 6
        assert metadata["step"] in {1, 2}
        assert (directory / "last.pt").is_file()
        assert (directory / "metrics.jsonl").is_file()
        assert (directory / "summary.json").is_file()
        records = [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        assert sum(record["split"] == "validation" for record in records) == 2
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["best_validation"]["l2"] >= 0
        assert summary["best_validation"]["score_mse"] >= 0
        assert summary["best_validation"]["score_rms"] >= 0
    assert (output / "grid_summary.json").is_file()

    config.data.statistics_path = None
    with pytest.raises(ValueError, match="data.statistics_path is required"):
        train_denoiser(config, tmp_path / "missing-statistics")
