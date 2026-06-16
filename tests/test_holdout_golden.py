"""Holdout split indices must match Mojo holdout.mojo (gate for Torch prototype)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from student.dataset import synthetic
from student.holdout import split_holdout

ROOT = Path(__file__).resolve().parents[1]


def _mojo_parity_export() -> dict[str, object]:
    cmd = ["pixi", "run", "mojo", "main.mojo", "parity-export"]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = proc.stdout
    start = stdout.find('{"kind":"holdout"')
    if start < 0:
        raise RuntimeError(f"parity-export missing holdout payload:\n{stdout}")
    end = stdout.find("]}", start)
    if end < 0:
        raise RuntimeError(f"parity-export malformed holdout payload:\n{stdout}")
    holdout_line = stdout[start : end + 2]
    return json.loads(holdout_line)


def test_holdout_indices_match_mojo() -> None:
    n = 4096
    seed = 42
    fraction = 0.1

    data = synthetic(n, 32, seed=0)
    split = split_holdout(data, fraction, seed)

    ref = _mojo_parity_export()
    assert ref["n"] == n
    assert ref["holdout_count"] == 409
    assert ref["train_count"] == 3687

    mojo_holdout = np.array(ref["holdout_indices"], dtype=np.int64)
    np.testing.assert_array_equal(
        np.sort(split.holdout_original_indices),
        np.sort(mojo_holdout),
    )
    assert len(split.train_original_indices) == int(ref["train_count"])
    assert len(split.holdout_original_indices) == int(ref["holdout_count"])
