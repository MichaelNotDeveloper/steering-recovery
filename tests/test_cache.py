import numpy as np
import torch

from steering_recovery.cache import NpyShardWriter, select_token_activations


def test_select_token_activations_handles_left_padding():
    hidden = torch.arange(24).reshape(2, 3, 4)
    mask = torch.tensor([[0, 1, 1], [0, 0, 1]])
    selected = select_token_activations(hidden, mask, "last")
    torch.testing.assert_close(selected, torch.stack((hidden[0, 2], hidden[1, 2])))
    assert select_token_activations(hidden, mask, "all").shape == (3, 4)


def test_shard_writer_respects_size(tmp_path):
    writer = NpyShardWriter(tmp_path, shard_size=3)
    writer.add(torch.arange(20).reshape(5, 4))
    shards = writer.close()
    assert [item["examples"] for item in shards] == [3, 2]
    np.testing.assert_array_equal(
        np.concatenate([np.load(tmp_path / item["file"]) for item in shards]),
        np.arange(20).reshape(5, 4),
    )
