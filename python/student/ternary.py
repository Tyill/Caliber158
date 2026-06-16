"""Ternary quantization and STE matching src/chain/ternary.mojo."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quantize_ternary(w: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Map shadow FP32 weights to {-1, 0, 1}."""
    out = torch.zeros_like(w)
    out = out.masked_fill(w > threshold, 1.0)
    out = out.masked_fill(w < -threshold, -1.0)
    return out


class TernarySTE(torch.autograd.Function):
    """Straight-through estimator: forward quantize, backward through shadow."""

    @staticmethod
    def forward(ctx, shadow: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(shadow)
        ctx.threshold = threshold
        return quantize_ternary(shadow, threshold)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        shadow, = ctx.saved_tensors
        return grad_output, None


def ternary_linear(
    x: torch.Tensor,
    shadow: torch.Tensor,
    *,
    use_ternary: bool,
    training: bool,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Batched linear with optional ternary weights (STE in train, quantize in eval)."""
    if not use_ternary:
        return F.linear(x, shadow)
    if training:
        w = TernarySTE.apply(shadow, threshold)
    else:
        w = quantize_ternary(shadow, threshold)
    return F.linear(x, w)
