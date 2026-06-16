"""Optional gradcheck on tiny MicroNet."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from student.model import MicroNet


def test_gradcheck_tiny() -> None:
    input_dim = 8
    hidden_dim = 4
    batch = 2
    model = MicroNet(input_dim, hidden_dim, arch="v0", use_ternary=True, init_scale=0.1)
    model.train()
    x = torch.randn(batch, input_dim, dtype=torch.float64, requires_grad=True)
    y = torch.randn(batch, dtype=torch.float64)

    def fn(x_in: torch.Tensor) -> torch.Tensor:
        m = model.double()
        return m(x_in)

    assert torch.autograd.gradcheck(fn, x, eps=1e-6, atol=1e-4)
