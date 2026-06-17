"""Load consecutive chain datasets that share the same X (one layer, same seed)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import ChainDataset, read_dataset
from .holdout import ChainTrainSplit, split_holdout
from .metrics import mean_dim_variance_y, variance_y


@dataclass(frozen=True)
class ChainGroupDataset:
    n_samples: int
    input_dim: int
    chain_group: int
    layer: int
    base_neuron: int
    x: np.ndarray  # [n, input_dim] float32
    y: np.ndarray  # [n, chain_group] float32
    chain_paths: tuple[Path, ...]


@dataclass(frozen=True)
class ChainGroupTrainSplit:
    train: ChainGroupDataset
    holdout: ChainGroupDataset
    var_y_train: float
    var_y_holdout: float
    shuffled_indices: np.ndarray
    holdout_original_indices: np.ndarray
    train_original_indices: np.ndarray


def chain_bin_path(data_dir: Path, layer: int, neuron: int) -> Path:
    return data_dir / f"L{layer:02d}_N{neuron:04d}.bin"


def resolve_chain_group_paths(
    data_dir: Path,
    layer: int,
    base_neuron: int,
    chain_group: int,
) -> list[Path]:
    if chain_group < 1:
        raise ValueError(f"chain_group must be >= 1, got {chain_group}")
    return [chain_bin_path(data_dir, layer, base_neuron + i) for i in range(chain_group)]


def _assert_shared_x(paths: list[Path], datasets: list[ChainDataset]) -> None:
    ref = datasets[0]
    for path, ds in zip(paths[1:], datasets[1:], strict=True):
        if ds.n_samples != ref.n_samples:
            raise ValueError(
                f"sample count mismatch: {paths[0].name} n={ref.n_samples} "
                f"vs {path.name} n={ds.n_samples}"
            )
        if ds.input_dim != ref.input_dim:
            raise ValueError(
                f"input_dim mismatch: {paths[0].name} d={ref.input_dim} "
                f"vs {path.name} d={ds.input_dim}"
            )
        if not np.allclose(ds.x, ref.x, rtol=0.0, atol=0.0):
            raise ValueError(
                f"X mismatch between {paths[0].name} and {path.name}; "
                "re-extract with the same CALIBER158_SEED and layer"
            )


def read_chain_group(
    paths: list[Path],
    *,
    layer: int,
    base_neuron: int,
) -> ChainGroupDataset:
    if not paths:
        raise ValueError("read_chain_group requires at least one dataset path")
    datasets = [read_dataset(p) for p in paths]
    _assert_shared_x(paths, datasets)
    ref = datasets[0]
    y = np.stack([ds.y for ds in datasets], axis=1).astype(np.float32, copy=False)
    return ChainGroupDataset(
        n_samples=ref.n_samples,
        input_dim=ref.input_dim,
        chain_group=len(paths),
        layer=layer,
        base_neuron=base_neuron,
        x=ref.x.copy(),
        y=y,
        chain_paths=tuple(paths),
    )


def load_chain_group_from_env(
    data_dir: Path,
    layer: int,
    base_neuron: int,
    chain_group: int,
) -> ChainGroupDataset:
    paths = resolve_chain_group_paths(data_dir, layer, base_neuron, chain_group)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"missing chain dataset(s) for group size {chain_group}: {names}; "
            f"run make extract-group CHAIN_GROUP={chain_group}"
        )
    return read_chain_group(paths, layer=layer, base_neuron=base_neuron)


def split_chain_group_holdout(
    data: ChainGroupDataset,
    holdout_fraction: float,
    seed: int,
) -> ChainGroupTrainSplit:
    single = ChainDataset(
        n_samples=data.n_samples,
        input_dim=data.input_dim,
        x=data.x,
        y=data.y[:, 0],
    )
    split = split_holdout(single, holdout_fraction, seed)

    holdout_y = data.y[split.holdout_original_indices]
    train_y = data.y[split.train_original_indices]

    holdout_ds = ChainGroupDataset(
        n_samples=split.holdout.holdout.n_samples,
        input_dim=data.input_dim,
        chain_group=data.chain_group,
        layer=data.layer,
        base_neuron=data.base_neuron,
        x=split.holdout.holdout.x,
        y=holdout_y,
        chain_paths=data.chain_paths,
    )
    train_ds = ChainGroupDataset(
        n_samples=split.train.n_samples,
        input_dim=data.input_dim,
        chain_group=data.chain_group,
        layer=data.layer,
        base_neuron=data.base_neuron,
        x=split.train.x,
        y=train_y,
        chain_paths=data.chain_paths,
    )

    var_train = mean_dim_variance_y(train_ds.y)
    var_holdout = mean_dim_variance_y(holdout_ds.y)
    return ChainGroupTrainSplit(
        train=train_ds,
        holdout=holdout_ds,
        var_y_train=var_train,
        var_y_holdout=var_holdout,
        shuffled_indices=split.shuffled_indices,
        holdout_original_indices=split.holdout_original_indices,
        train_original_indices=split.train_original_indices,
    )


def per_chain_variance_y(y: np.ndarray) -> tuple[float, ...]:
    if y.ndim != 2:
        raise ValueError(f"expected y [n, k], got shape {y.shape}")
    return tuple(variance_y(y[:, k]) for k in range(y.shape[1]))


def synthetic_chain_group(
    n_samples: int,
    input_dim: int,
    chain_group: int,
    seed: int = 42,
) -> ChainGroupDataset:
    """Synthetic multi-chain dataset with shared X and independent Y columns."""
    from .dataset import synthetic

    base = synthetic(n_samples, input_dim, seed=seed)
    rng = np.random.default_rng(seed + 1)
    y = np.empty((n_samples, chain_group), dtype=np.float32)
    y[:, 0] = base.y
    for k in range(1, chain_group):
        y[:, k] = rng.standard_normal(n_samples, dtype=np.float32)
    return ChainGroupDataset(
        n_samples=n_samples,
        input_dim=input_dim,
        chain_group=chain_group,
        layer=0,
        base_neuron=0,
        x=base.x,
        y=y,
        chain_paths=(),
    )
