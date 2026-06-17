"""Train/holdout split for layer FFN datasets (same LCG as chain holdout)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layer_dataset import LayerDataset
from .metrics import mean_dim_variance_y
from .rng import fisher_yates_indices


@dataclass(frozen=True)
class LayerHoldoutSplit:
    holdout: LayerDataset
    var_y_train: float
    var_y_holdout: float


@dataclass(frozen=True)
class LayerTrainSplit:
    train: LayerDataset
    holdout: LayerHoldoutSplit
    shuffled_indices: np.ndarray
    holdout_original_indices: np.ndarray
    train_original_indices: np.ndarray


def split_layer_holdout(
    data: LayerDataset,
    holdout_fraction: float,
    seed: int,
) -> LayerTrainSplit:
    """Shuffle sample indices with LCG(seed) and partition into train/holdout."""
    n = data.n_samples
    if n < 2:
        raise ValueError("holdout split requires at least 2 samples")

    holdout_count = int(float(n) * holdout_fraction)
    if holdout_count < 1:
        holdout_count = 1
    if holdout_count >= n:
        holdout_count = n - 1

    indices = fisher_yates_indices(n, seed)
    holdout_idx = indices[:holdout_count]
    train_idx = indices[holdout_count:]

    holdout_ds = LayerDataset(
        n_samples=holdout_count,
        input_dim=data.input_dim,
        output_dim=data.output_dim,
        x=data.x[holdout_idx],
        y=data.y[holdout_idx],
    )
    train_ds = LayerDataset(
        n_samples=n - holdout_count,
        input_dim=data.input_dim,
        output_dim=data.output_dim,
        x=data.x[train_idx],
        y=data.y[train_idx],
    )

    meta = LayerHoldoutSplit(
        holdout=holdout_ds,
        var_y_train=mean_dim_variance_y(train_ds.y),
        var_y_holdout=mean_dim_variance_y(holdout_ds.y),
    )
    return LayerTrainSplit(
        train=train_ds,
        holdout=meta,
        shuffled_indices=indices,
        holdout_original_indices=holdout_idx,
        train_original_indices=train_idx,
    )
