"""Deterministic train/holdout split matching src/chain/holdout.mojo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import ChainDataset
from .metrics import variance_y
from .rng import fisher_yates_indices


@dataclass(frozen=True)
class HoldoutSplit:
    holdout: ChainDataset
    var_y_train: float
    var_y_holdout: float


@dataclass(frozen=True)
class ChainTrainSplit:
    train: ChainDataset
    holdout: HoldoutSplit
    shuffled_indices: np.ndarray
    holdout_original_indices: np.ndarray
    train_original_indices: np.ndarray


def split_holdout(
    data: ChainDataset,
    holdout_fraction: float,
    seed: int,
) -> ChainTrainSplit:
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

    holdout_x = data.x[holdout_idx]
    holdout_y = data.y[holdout_idx]
    train_x = data.x[train_idx]
    train_y = data.y[train_idx]

    holdout_ds = ChainDataset(
        n_samples=holdout_count,
        input_dim=data.input_dim,
        x=holdout_x,
        y=holdout_y,
    )
    train_ds = ChainDataset(
        n_samples=n - holdout_count,
        input_dim=data.input_dim,
        x=train_x,
        y=train_y,
    )

    var_train = variance_y(train_ds.y)
    var_holdout = variance_y(holdout_ds.y)
    meta = HoldoutSplit(holdout=holdout_ds, var_y_train=var_train, var_y_holdout=var_holdout)
    return ChainTrainSplit(
        train=train_ds,
        holdout=meta,
        shuffled_indices=indices,
        holdout_original_indices=holdout_idx,
        train_original_indices=train_idx,
    )
