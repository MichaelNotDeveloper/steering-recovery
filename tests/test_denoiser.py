import pytest
import torch
from torch import nn

from steering_recovery.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_gpt2_small_denoiser_precision,
)
from steering_recovery.denoiser import (
    ActivationDenoiser,
    DenoiserBundle,
    ResidualBlock,
)
from steering_recovery.normalization import ActivationNormalizer


def test_residual_block_has_the_requested_architecture():
    block = ResidualBlock(hidden_size=6, latent_dim=8, dropout=0.1)
    assert isinstance(block.network[0], nn.Linear)
    assert block.network[0].in_features == 6
    assert block.network[0].out_features == 8
    assert block.network[0].bias is not None
    assert isinstance(block.network[1], nn.GELU)
    assert isinstance(block.network[2], nn.Linear)
    assert block.network[2].in_features == 8
    assert block.network[2].out_features == 6
    assert block.network[2].bias is not None
    assert isinstance(block.network[3], nn.Dropout)
    assert block.network[3].p == 0.1
    assert not any(isinstance(module, nn.LayerNorm) for module in block.modules())


def test_denoiser_preserves_2d_and_3d_shapes_and_backpropagates():
    model = ActivationDenoiser(hidden_size=6, latent_dim=8, num_layers=2)
    two_d = torch.randn(4, 6, requires_grad=True)
    three_d = torch.randn(4, 3, 6)
    assert model(two_d).shape == two_d.shape
    assert model(three_d).shape == three_d.shape
    model(two_d).sum().backward()
    assert two_d.grad is not None


def test_bundle_normalizes_and_checkpoint_roundtrips(tmp_path):
    model = ActivationDenoiser(hidden_size=4, latent_dim=8, num_layers=1, dropout=0.1)
    normalizer = ActivationNormalizer(torch.arange(4.0), torch.ones(4) * 2)
    bundle = DenoiserBundle(model, normalizer)
    bundle.eval()
    values = torch.randn(2, 4)
    expected = bundle.denoise(values)

    path = save_checkpoint(tmp_path / "model.pt", bundle, step=3, epoch=1)
    loaded, metadata = load_checkpoint(path)
    torch.testing.assert_close(loaded.denoise(values), expected)
    assert metadata["format_version"] == 2
    assert metadata["step"] == 3
    assert loaded.model_config == bundle.model_config
    assert loaded.model.config.dropout == 0.1


def test_checkpoint_without_dropout_field_remains_loadable(tmp_path):
    bundle = DenoiserBundle(
        ActivationDenoiser(hidden_size=4, latent_dim=8, num_layers=1),
        ActivationNormalizer(torch.zeros(4), torch.ones(4)),
    ).eval()
    path = save_checkpoint(tmp_path / "current.pt", bundle, step=1, epoch=0)
    payload = torch.load(path, weights_only=True)
    payload["model_config"].pop("dropout")
    legacy_path = tmp_path / "legacy.pt"
    torch.save(payload, legacy_path)
    loaded, _ = load_checkpoint(legacy_path)
    assert loaded.model.config.dropout == 0.0


def test_gpt2_denoiser_precision_provenance_is_required():
    fp32_metadata = {
        "config": {
            "experiment": {
                "data": {"streaming": {"model_dtype": "float32"}},
                "training": {"precision": "fp32"},
            }
        }
    }
    validate_gpt2_small_denoiser_precision(fp32_metadata, source_model_name="gpt2")
    reduced_metadata = {
        "config": {
            "experiment": {
                "data": {"streaming": {"model_dtype": "auto"}},
                "training": {"precision": "bf16"},
            }
        }
    }
    with pytest.raises(ValueError, match="trained entirely in float32"):
        validate_gpt2_small_denoiser_precision(
            reduced_metadata, source_model_name="gpt2"
        )
