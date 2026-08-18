from __future__ import annotations

import bisect
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


_TENSOR_SUFFIXES = {".npy", ".pt", ".pth"}
_IGNORED_STEMS = {"statistics", "steering_vectors"}


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location="cpu")


def extract_tensor(payload: Any, key: str = "activations") -> torch.Tensor:
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, dict):
        if key in payload and torch.is_tensor(payload[key]):
            return payload[key]
        tensor_values = [value for value in payload.values() if torch.is_tensor(value)]
        if len(tensor_values) == 1:
            return tensor_values[0]
        raise KeyError(
            f"cannot choose a tensor from mapping; expected key {key!r} or exactly one tensor"
        )
    raise TypeError(f"expected a tensor or mapping, got {type(payload).__name__}")


def load_tensor(path: str | Path, key: str = "activations") -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".npy":
        return torch.from_numpy(np.asarray(np.load(path)))
    if path.suffix in {".pt", ".pth"}:
        return extract_tensor(_safe_torch_load(path), key=key)
    raise ValueError(f"unsupported tensor file: {path}")


class ActivationDataset(Dataset[torch.Tensor]):
    """Lazy, shard-aware dataset for standard ``.npy`` and PyTorch tensors.

    Every shard may have shape ``[..., hidden_size]``. All leading dimensions
    are flattened into independent examples.
    """

    def __init__(self, path: str | Path, key: str = "activations"):
        self.path = Path(path)
        self.key = key
        self.files = self._discover_files(self.path)
        if not self.files:
            raise FileNotFoundError(f"no activation shards found under {self.path}")

        self._shapes: list[tuple[int, ...]] = []
        self._ends: list[int] = []
        self._cache_path: Path | None = None
        self._cache_value: torch.Tensor | np.ndarray | None = None
        total = 0
        hidden_size: int | None = None
        for file in self.files:
            shape = self._read_shape(file)
            if len(shape) < 2:
                raise ValueError(
                    f"activation shard must be at least 2-D: {file} has {shape}"
                )
            if hidden_size is None:
                hidden_size = shape[-1]
            elif shape[-1] != hidden_size:
                raise ValueError(
                    f"inconsistent hidden size: {file} has {shape[-1]}, expected {hidden_size}"
                )
            count = math.prod(shape[:-1])
            if count == 0:
                raise ValueError(f"activation shard is empty: {file}")
            total += count
            self._shapes.append(shape)
            self._ends.append(total)

        if total == 0 or hidden_size is None:
            raise ValueError("activation dataset is empty")
        self.hidden_size = int(hidden_size)

    @staticmethod
    def _discover_files(path: Path) -> list[Path]:
        if path.is_file():
            if path.suffix not in _TENSOR_SUFFIXES:
                raise ValueError(f"unsupported activation file: {path}")
            return [path]
        if not path.exists():
            raise FileNotFoundError(path)
        files = [
            item
            for item in path.iterdir()
            if item.is_file()
            and item.suffix in _TENSOR_SUFFIXES
            and item.stem not in _IGNORED_STEMS
        ]
        return sorted(files)

    def _read_shape(self, path: Path) -> tuple[int, ...]:
        if path.suffix == ".npy":
            return tuple(np.load(path, mmap_mode="r").shape)
        return tuple(extract_tensor(_safe_torch_load(path), self.key).shape)

    def _load_shard(self, shard_idx: int) -> torch.Tensor | np.ndarray:
        path = self.files[shard_idx]
        if self._cache_path == path and self._cache_value is not None:
            return self._cache_value
        if path.suffix == ".npy":
            value: torch.Tensor | np.ndarray = np.load(path, mmap_mode="r")
        else:
            value = extract_tensor(_safe_torch_load(path), self.key)
        self._cache_path = path
        self._cache_value = value
        return value

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_idx = bisect.bisect_right(self._ends, index)
        shard_start = 0 if shard_idx == 0 else self._ends[shard_idx - 1]
        local_idx = index - shard_start
        shard = self._load_shard(shard_idx)
        if isinstance(shard, np.ndarray):
            row = np.asarray(shard).reshape(-1, self.hidden_size)[local_idx].copy()
            return torch.from_numpy(row).float()
        return shard.reshape(-1, self.hidden_size)[local_idx].detach().float()


def split_dataset(
    dataset: Dataset[torch.Tensor], val_fraction: float, seed: int
) -> tuple[Dataset[torch.Tensor], Dataset[torch.Tensor] | None]:
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if val_fraction == 0 or len(dataset) < 2:
        return dataset, None
    val_size = max(1, round(len(dataset) * val_fraction))
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(dataset, [train_size, val_size], generator=generator)
    return train, val


@torch.no_grad()
def compute_statistics(
    dataset: Dataset[torch.Tensor], batch_size: int = 1024, num_workers: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute population mean/std with a numerically stable parallel update."""

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    count = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None
    for batch in loader:
        values = batch.double().reshape(-1, batch.shape[-1])
        batch_count = values.shape[0]
        batch_mean = values.mean(dim=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(dim=0)
        if mean is None:
            mean, m2, count = batch_mean, batch_m2, batch_count
            continue
        delta = batch_mean - mean
        new_count = count + batch_count
        mean = mean + delta * (batch_count / new_count)
        m2 = m2 + batch_m2 + delta.square() * count * batch_count / new_count
        count = new_count
    if count == 0 or mean is None or m2 is None:
        raise ValueError("cannot compute statistics for an empty dataset")
    variance = (m2 / count).clamp_min(0)
    return mean.float(), variance.sqrt().float()
