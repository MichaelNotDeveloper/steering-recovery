import torch

from steering_recovery.checkpoint import load_checkpoint, save_checkpoint
from steering_recovery.denoiser import ActivationDenoiser, DenoiserBundle
from steering_recovery.normalization import ActivationNormalizer


def test_denoiser_preserves_2d_and_3d_shapes_and_backpropagates():
    model = ActivationDenoiser(hidden_size=6, width=8, depth=2, expansion=2)
    two_d = torch.randn(4, 6, requires_grad=True)
    three_d = torch.randn(4, 3, 6)
    assert model(two_d, torch.ones(4)).shape == two_d.shape
    assert model(three_d, torch.ones(4)).shape == three_d.shape
    model(two_d, 0.2).sum().backward()
    assert two_d.grad is not None


def test_new_bundle_is_identity_and_checkpoint_roundtrips(tmp_path):
    model = ActivationDenoiser(hidden_size=4, width=8, depth=1)
    normalizer = ActivationNormalizer(torch.arange(4.0), torch.ones(4) * 2)
    bundle = DenoiserBundle(model, normalizer)
    values = torch.randn(2, 4)
    torch.testing.assert_close(bundle.denoise(values, 0.3), values)

    path = save_checkpoint(tmp_path / "model.pt", bundle, step=3, epoch=1)
    loaded, metadata = load_checkpoint(path)
    torch.testing.assert_close(loaded.denoise(values, 0.3), values)
    assert metadata["step"] == 3
    assert loaded.model_config == bundle.model_config
