"""Dataset metrics matching src/chain/metrics.mojo."""

from __future__ import annotations

import numpy as np
import torch


def variance_y(y: np.ndarray | torch.Tensor) -> float:
    """Population variance of Y over all samples."""
    if isinstance(y, torch.Tensor):
        if y.numel() == 0:
            return 0.0
        y = y.detach().float()
        mean = y.mean()
        return float(((y - mean) ** 2).mean().item())

    if len(y) == 0:
        return 0.0
    mean = float(np.mean(y))
    return float(np.mean((y - mean) ** 2))


def relative_mse(mse: float, var_y: float) -> float:
    """MSE divided by Var(Y); 1.0 ≈ predicting the mean."""
    if var_y <= 0.0:
        return 0.0
    return mse / var_y


def batch_count(n_samples: int, start: int, requested: int) -> int:
    """Match ChainData.batch_size in src/chain/buffer.mojo."""
    end = start + requested
    if end > n_samples:
        end = n_samples
    return end - start
