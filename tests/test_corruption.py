import torch

from steering_recovery.corruption import OnlineCorruptor


def test_corruptor_is_reproducible_and_reports_rms():
    clean = torch.zeros(8, 4)
    vectors = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    corruptor = OnlineCorruptor(
        0.1,
        0.2,
        steering_probability=1.0,
        steering_scale_min=0.5,
        steering_scale_max=0.5,
        steering_vectors=vectors,
    )
    first = corruptor(clean, torch.Generator().manual_seed(11))
    second = corruptor(clean, torch.Generator().manual_seed(11))
    torch.testing.assert_close(first.noisy, second.noisy)
    expected_level = first.corruption.square().mean(-1).sqrt()
    torch.testing.assert_close(first.noise_level, expected_level)
    assert first.used_steering.all()


def test_identity_probability_produces_clean_examples():
    clean = torch.randn(3, 5)
    corruptor = OnlineCorruptor(0.5, 0.5, identity_probability=1.0)
    result = corruptor(clean)
    torch.testing.assert_close(result.noisy, clean)
    assert torch.count_nonzero(result.noise_level) == 0
