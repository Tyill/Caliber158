"""Roundtrip tests for CAL158L layer dataset format."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from student.layer_dataset import read_layer_dataset, write_layer_dataset
from student.metrics import mean_dim_variance_y


def test_layer_dataset_roundtrip() -> None:
    rng = np.random.default_rng(0)
    n, d, d_out = 16, 8, 8
    x = rng.standard_normal((n, d), dtype=np.float32)
    y = rng.standard_normal((n, d_out), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "L00_ffn.bin"
        write_layer_dataset(path, x, y)
        loaded = read_layer_dataset(path)
    assert loaded.n_samples == n
    assert loaded.input_dim == d
    assert loaded.output_dim == d_out
    np.testing.assert_allclose(loaded.x, x)
    np.testing.assert_allclose(loaded.y, y)
    var = mean_dim_variance_y(y)
    assert var > 0.0
