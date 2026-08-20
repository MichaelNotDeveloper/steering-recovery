import itertools

import pytest
import torch
from omegaconf import OmegaConf

import steering_recovery.training as training_module
from steering_recovery.checkpoint import load_checkpoint
from steering_recovery.streaming_data import TeacherForcedActivationIterableDataset


class InfiniteTexts:
    def __iter__(self):
        return (str(index) for index in itertools.count())

    def set_epoch(self, epoch):
        self.epoch = epoch


class DeterministicExtractor:
    hidden_size = 6

    def __call__(self, texts):
        chunks = []
        for text in texts:
            generator = torch.Generator().manual_seed(int(text))
            chunks.append(torch.randn(3, self.hidden_size, generator=generator))
        return chunks


def test_streaming_training_smoke(monkeypatch, tmp_path):
    def fake_builder(data_config, training_config, *, device, seed):
        extractor = DeterministicExtractor()
        train = TeacherForcedActivationIterableDataset(
            InfiniteTexts(),
            extractor,
            batch_size=training_config.batch_size,
            text_batch_size=2,
            max_batches=training_config.max_steps,
        )
        validation = TeacherForcedActivationIterableDataset(
            InfiniteTexts(),
            extractor,
            batch_size=training_config.batch_size,
            text_batch_size=2,
            max_batches=training_config.validation_batches,
        )
        return train, validation

    monkeypatch.setattr(
        training_module, "build_streaming_activation_datasets", fake_builder
    )
    statistics_path = tmp_path / "statistics.pt"
    torch.save(
        {
            "sum": torch.zeros(6, dtype=torch.float64),
            "variance": torch.ones(6, dtype=torch.float64),
            "count": 1,
            "source": {
                "model_name": "fake-gpt",
                "tokenizer_name": "fake-gpt",
                "layer_path": "h",
                "layer_index": 0,
                "max_length": 8,
            },
        },
        statistics_path,
    )
    config = OmegaConf.create(
        {
            "seed": 3,
            "device": "cpu",
            "data": {
                "mode": "streaming",
                "statistics_path": str(statistics_path),
                "streaming": {
                    "model_name": "fake-gpt",
                    "tokenizer_name": "fake-gpt",
                    "layer_path": "h",
                    "layer_index": 0,
                    "max_length": 8,
                },
            },
            "model": {
                "hidden_size": 6,
                "latent_dims": [8],
                "num_layers": [1],
                "sigmas": [0.1],
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
    result = training_module.train_denoiser(config, output)
    assert result["steps"] == 2
    model_directory = next((output / "models").iterdir())
    bundle, metadata = load_checkpoint(model_directory / "last.pt")
    assert bundle.model.config.hidden_size == 6
    assert metadata["step"] == 2

    config.data.streaming.layer_index = 1
    with pytest.raises(ValueError, match="statistics source does not match"):
        training_module.train_denoiser(config, tmp_path / "wrong-layer")
