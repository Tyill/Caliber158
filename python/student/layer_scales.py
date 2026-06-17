"""FP32 output scaling for ternary FFN (global alpha or per-channel)."""

from __future__ import annotations

import torch


def fit_global_scale(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Closed-form scalar minimizing mean((scale * pred - target)^2)."""
    num = torch.sum(pred * target)
    den = torch.sum(pred * pred)
    if float(den.item()) <= 0.0:
        return 1.0
    return float((num / den).item())


def fit_channel_scales(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Per-output-dim scale [D] minimizing mean over batch of (s[d]*pred - target)^2."""
    num = torch.sum(pred * target, dim=0)
    den = torch.sum(pred * pred, dim=0).clamp(min=eps)
    return num / den


def fit_projection_scales(
    shadow_out: torch.Tensor,
    ternary_out: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-output scale [O] mapping ternary matmul output toward shadow (BitNet-style)."""
    num = torch.sum(shadow_out * ternary_out, dim=0)
    den = torch.sum(ternary_out * ternary_out, dim=0).clamp(min=eps)
    return num / den
