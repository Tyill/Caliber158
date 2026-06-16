"""One-batch train loss must match Mojo CPU train_step_cpu."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from student.dataset import synthetic
from student.model import MicroNet

LOSS_TOL = 1e-4


def _mojo_batch_loss() -> float:
    cmd = ["pixi", "run", "mojo", "main.mojo", "parity-export"]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith('{"kind":"batch_loss"'):
            payload = json.loads(line)
            return float(payload["loss"])
    stdout = proc.stdout
    start = stdout.find('{"kind":"batch_loss"')
    if start < 0:
        raise RuntimeError(f"parity-export missing batch_loss payload:\n{stdout}")
    end = stdout.find("}", start)
    payload = json.loads(stdout[start : end + 1])
    return float(payload["loss"])


def test_forward_batch_loss_matches_mojo_cpu() -> None:
    input_dim = 32
    hidden_dim = 8
    batch_size = 16

    data = synthetic(batch_size, input_dim, seed=42)
    model = MicroNet(
        input_dim,
        hidden_dim,
        arch="v0",
        use_ternary=True,
        init_scale=0.1,
    )
    model.train()
    x = torch.from_numpy(data.x)
    y = torch.from_numpy(data.y)
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y, reduction="mean")

    mojo_loss = _mojo_batch_loss()
    assert abs(float(loss.item()) - mojo_loss) < LOSS_TOL
