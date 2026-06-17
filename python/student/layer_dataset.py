"""CAL158L layer FFN dataset reader/writer (vector Y)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rng import lcg_next, unit_float

MAGIC = b"CAL158L"
VERSION = 1
_HEADER_SIZE = len(MAGIC) + 16  # magic + 4 x uint32


@dataclass(frozen=True)
class LayerDataset:
    n_samples: int
    input_dim: int
    output_dim: int
    x: np.ndarray  # [n, input_dim] float32
    y: np.ndarray  # [n, output_dim] float32


def read_layer_dataset(path: Path | str) -> LayerDataset:
    """Read CAL158L binary written by python/extract_layer_ffn.py."""
    data = Path(path).read_bytes()
    if len(data) < _HEADER_SIZE:
        raise ValueError("layer dataset file too small")
    if data[: len(MAGIC)] != MAGIC:
        raise ValueError(f"bad magic: expected {MAGIC!r}")

    version, n_samples, input_dim, output_dim = struct.unpack_from("<IIII", data, len(MAGIC))
    if version != VERSION:
        raise ValueError(f"unsupported layer dataset version: {version}")

    x_bytes = n_samples * input_dim * 4
    y_bytes = n_samples * output_dim * 4
    if len(data) < _HEADER_SIZE + x_bytes + y_bytes:
        raise ValueError("layer dataset file truncated")

    x = np.frombuffer(
        data,
        dtype=np.float32,
        count=n_samples * input_dim,
        offset=_HEADER_SIZE,
    ).reshape(n_samples, input_dim)
    y = np.frombuffer(
        data,
        dtype=np.float32,
        count=n_samples * output_dim,
        offset=_HEADER_SIZE + x_bytes,
    ).reshape(n_samples, output_dim)
    return LayerDataset(
        n_samples=n_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        x=x.copy(),
        y=y.copy(),
    )


def write_layer_dataset(path: Path | str, x: np.ndarray, y: np.ndarray) -> None:
    """Write CAL158L binary (X [N,D], Y [N,D_out])."""
    n, d = x.shape
    if y.shape[0] != n:
        raise ValueError(f"x rows {n} != y rows {y.shape[0]}")
    d_out = y.shape[1]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", VERSION, n, d, d_out))
        f.write(x.astype(np.float32, copy=False).tobytes(order="C"))
        f.write(y.astype(np.float32, copy=False).tobytes(order="C"))


def synthetic_layer(
    n_samples: int,
    input_dim: int,
    output_dim: int,
    seed: int = 42,
) -> LayerDataset:
    """Small LCG synthetic dataset for smoke tests."""
    x = np.empty((n_samples, input_dim), dtype=np.float32)
    y = np.empty((n_samples, output_dim), dtype=np.float32)
    rng = seed & ((1 << 64) - 1)
    for i in range(n_samples * input_dim):
        rng = lcg_next(rng)
        x.ravel()[i] = unit_float(rng)
    for i in range(n_samples * output_dim):
        rng = lcg_next(rng)
        y.ravel()[i] = unit_float(rng) * 2.0 - 1.0
    return LayerDataset(
        n_samples=n_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        x=x,
        y=y,
    )
