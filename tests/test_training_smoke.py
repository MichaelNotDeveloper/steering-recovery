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
                "width": 8,
                "depth": 1,
                "expansion": 2,
                "dropout": 0.0,
            },
            "corruption": {
                "gaussian_std_min": 0.1,
                "gaussian_std_max": 0.2,
                "steering_probability": 0.0,
                "steering_scale_min": 0.0,
                "steering_scale_max": 0.0,
                "steering_vectors_path": None,
                "steering_vectors_key": "steering_vectors",
                "identity_probability": 0.0,
                "bidirectional": True,
            },
            "training": {
                "epochs": 1,
                "max_steps": 2,
                "batch_size": 4,
                "gradient_accumulation_steps": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "betas": [0.9, 0.95],
                "warmup_ratio": 0.0,
                "min_lr_factor": 0.1,
                "max_grad_norm": 1.0,
                "precision": "fp32",
                "log_every_steps": 1,
                "save_every_steps": 100,
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
    bundle, metadata = load_checkpoint(output / "last.pt")
    assert bundle.model.config.hidden_size == 6
    assert metadata["step"] == 2

    config.data.statistics_path = None
    with pytest.raises(ValueError, match="data.statistics_path is required"):
        train_denoiser(config, tmp_path / "missing-statistics")
