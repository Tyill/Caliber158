"""CAL158 chain dataset reader and synthetic generator."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rng import lcg_next, unit_float

MAGIC = b"CAL158"
VERSION = 1


@dataclass(frozen=True)
class ChainDataset:
    n_samples: int
    input_dim: int
    x: np.ndarray  # [n, input_dim] float32
    y: np.ndarray  # [n] float32


def read_dataset(path: Path | str) -> ChainDataset:
    """Read dataset written by python/extract_chain.py (see docs/architecture.md)."""
    data = Path(path).read_bytes()
    if len(data) < 18:
        raise ValueError("dataset file too small")
    if data[:6] != MAGIC:
        raise ValueError(f"bad magic: expected {MAGIC!r}")

    version, n_samples, input_dim = struct.unpack_from("<III", data, 6)
    if version != VERSION:
        raise ValueError(f"unsupported dataset version: {version}")

    x_bytes = n_samples * input_dim * 4
    y_bytes = n_samples * 4
    header = 18
    if len(data) < header + x_bytes + y_bytes:
        raise ValueError("dataset file truncated")

    x = np.frombuffer(data, dtype=np.float32, count=n_samples * input_dim, offset=header).reshape(
        n_samples, input_dim
    )
    y = np.frombuffer(
        data, dtype=np.float32, count=n_samples, offset=header + x_bytes
    ).copy()
    return ChainDataset(n_samples=n_samples, input_dim=input_dim, x=x.copy(), y=y)


def synthetic(n_samples: int, input_dim: int, seed: int = 42) -> ChainDataset:
    """Match ChainDataset.synthetic in src/chain/dataset.mojo."""
    x = np.empty((n_samples, input_dim), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.float32)
    rng = seed & ((1 << 64) - 1)
    for i in range(n_samples * input_dim):
        rng = lcg_next(rng)
        x.ravel()[i] = unit_float(rng)
    for i in range(n_samples):
        rng = lcg_next(rng)
        y[i] = unit_float(rng) * 2.0 - 1.0
    return ChainDataset(n_samples=n_samples, input_dim=input_dim, x=x, y=y)


def write_dataset(path: Path | str, x: np.ndarray, y: np.ndarray) -> None:
    """Write CAL158 binary (optional, for smoke fixtures)."""
    n, d = x.shape
    if y.shape != (n,):
        raise ValueError(f"expected y shape ({n},), got {y.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, n, d))
        f.write(x.astype(np.float32, copy=False).tobytes(order="C"))
        f.write(y.astype(np.float32, copy=False).tobytes(order="C"))
