"""Tests for FFN output scale fitting."""

from __future__ import annotations

import torch

from student.layer_scales import fit_channel_scales, fit_global_scale


def test_fit_global_scale() -> None:
    pred = torch.tensor([[2.0], [4.0], [6.0]])
    target = torch.tensor([[1.0], [2.0], [3.0]])
    scale = fit_global_scale(pred, target)
    assert abs(scale - 0.5) < 1e-5


def test_fit_channel_scales() -> None:
    pred = torch.tensor([[2.0, 4.0], [4.0, 8.0]])
    target = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    scales = fit_channel_scales(pred, target)
    assert scales.shape == (2,)
    assert torch.allclose(scales, torch.tensor([0.5, 0.5]))
