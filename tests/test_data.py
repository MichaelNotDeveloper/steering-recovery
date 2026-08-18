import numpy as np
import pytest
import torch

from steering_recovery.data import ActivationDataset, compute_statistics, split_dataset


def test_activation_dataset_reads_and_flattens_shards(tmp_path):
    first = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    second = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    np.save(tmp_path / "a.npy", first)
    torch.save({"activations": second}, tmp_path / "b.pt")

    dataset = ActivationDataset(tmp_path)

    assert len(dataset) == 8
    assert dataset.hidden_size == 4
    torch.testing.assert_close(dataset[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    torch.testing.assert_close(dataset[-1], torch.tensor([4.0, 5.0, 6.0, 7.0]))


def test_activation_dataset_rejects_inconsistent_hidden_size(tmp_path):
    np.save(tmp_path / "a.npy", np.zeros((2, 4), dtype=np.float32))
    np.save(tmp_path / "b.npy", np.zeros((2, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="inconsistent hidden size"):
        ActivationDataset(tmp_path)


def test_statistics_and_split_are_deterministic(tmp_path):
    values = np.arange(40, dtype=np.float32).reshape(10, 4)
    np.save(tmp_path / "values.npy", values)
    dataset = ActivationDataset(tmp_path)
    mean, std = compute_statistics(dataset, batch_size=3)
    torch.testing.assert_close(mean, torch.from_numpy(values).mean(0))
    torch.testing.assert_close(std, torch.from_numpy(values).std(0, unbiased=False))
    first_train, first_val = split_dataset(dataset, 0.2, seed=7)
    second_train, second_val = split_dataset(dataset, 0.2, seed=7)
    assert first_train.indices == second_train.indices
    assert first_val.indices == second_val.indices
