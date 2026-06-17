"""Tests for shared-bottleneck chain group loading."""

from __future__ import annotations

import numpy as np
import pytest

from student.chain_group import load_chain_group_from_env, read_chain_group, resolve_chain_group_paths
from student.dataset import write_dataset
from student.shared_model import SharedMicroNetV0, shared_v0_param_count


def _write_chain(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    write_dataset(path, x, y)


def test_read_chain_group_shared_x(tmp_path: Path) -> None:
    n, d = 32, 8
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, d), dtype=np.float32)
    y0 = rng.standard_normal(n, dtype=np.float32)
    y1 = rng.standard_normal(n, dtype=np.float32)
    p0 = tmp_path / "L00_N0000.bin"
    p1 = tmp_path / "L00_N0001.bin"
    _write_chain(p0, x, y0)
    _write_chain(p1, x, y1)

    group = read_chain_group([p0, p1], layer=0, base_neuron=0)
    assert group.chain_group == 2
    assert group.y.shape == (n, 2)
    np.testing.assert_array_equal(group.y[:, 0], y0)
    np.testing.assert_array_equal(group.y[:, 1], y1)


def test_read_chain_group_x_mismatch(tmp_path: Path) -> None:
    n, d = 16, 4
    rng = np.random.default_rng(1)
    p0 = tmp_path / "L00_N0000.bin"
    p1 = tmp_path / "L00_N0001.bin"
    _write_chain(p0, rng.standard_normal((n, d), dtype=np.float32), rng.standard_normal(n))
    _write_chain(p1, rng.standard_normal((n, d), dtype=np.float32), rng.standard_normal(n))
    with pytest.raises(ValueError, match="X mismatch"):
        read_chain_group([p0, p1], layer=0, base_neuron=0)


def test_shared_param_count() -> None:
    assert shared_v0_param_count(896, 16, 1) == 2 * 16 * 896 + 16 + 1
    assert shared_v0_param_count(896, 16, 2) == 2 * 16 * 896 + 2 * 16 + 2
    assert shared_v0_param_count(896, 16, 10) == 2 * 16 * 896 + 10 * 16 + 10


def test_shared_model_forward_shape() -> None:
    import torch

    model = SharedMicroNetV0(896, 16, 3, use_ternary=False)
    x = torch.randn(4, 896)
    y = model(x)
    assert y.shape == (4, 3)


def test_load_chain_group_from_env(tmp_path: Path) -> None:
    n, d = 8, 4
    x = np.ones((n, d), dtype=np.float32)
    for neuron in range(2):
        path = tmp_path / f"L00_N{neuron:04d}.bin"
        _write_chain(path, x, np.full(n, float(neuron), dtype=np.float32))
    group = load_chain_group_from_env(tmp_path, layer=0, base_neuron=0, chain_group=2)
    assert group.chain_group == 2
    assert resolve_chain_group_paths(tmp_path, 0, 0, 2) == list(group.chain_paths)
