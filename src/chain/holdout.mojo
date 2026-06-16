"""Deterministic train/holdout split for chain datasets."""

from .buffer import ChainData
from .metrics import variance_y
from .rng import lcg_next, unit_float


@fieldwise_init
struct HoldoutSplit(Copyable, Movable):
    var holdout: ChainData
    var var_y_train: Float32
    var var_y_holdout: Float32


@fieldwise_init
struct ChainTrainSplit(Copyable, Movable):
    var train: ChainData
    var holdout: HoldoutSplit


def split_holdout(
    data: ChainData,
    holdout_fraction: Float32,
    seed: UInt64,
) raises -> ChainTrainSplit:
    """Shuffle sample indices with LCG(seed) and partition into train/holdout."""
    var n = data.n_samples
    if n < 2:
        raise Error("holdout split requires at least 2 samples")

    var holdout_count = Int(Float32(n) * holdout_fraction)
    if holdout_count < 1:
        holdout_count = 1
    if holdout_count >= n:
        holdout_count = n - 1

    var train_count = n - holdout_count
    var indices = List[Int](capacity=n)
    for i in range(n):
        indices.append(i)

    var rng = seed
    for i in range(n - 1, 0, -1):
        rng = lcg_next(rng)
        var j = Int(unit_float(rng) * Float32(i + 1))
        var tmp = indices[i]
        indices[i] = indices[j]
        indices[j] = tmp

    var train_x = List[Float32](capacity=train_count * data.input_dim)
    var train_y = List[Float32](capacity=train_count)
    var holdout_x = List[Float32](capacity=holdout_count * data.input_dim)
    var holdout_y = List[Float32](capacity=holdout_count)

    for i in range(n):
        var idx = indices[i]
        if i < holdout_count:
            _copy_sample(data, idx, holdout_x, holdout_y)
        else:
            _copy_sample(data, idx, train_x, train_y)

    var train = ChainData(train_count, data.input_dim, train_x^, train_y^)
    var holdout_data = ChainData(holdout_count, data.input_dim, holdout_x^, holdout_y^)
    var var_train = variance_y(train)
    var var_holdout = variance_y(holdout_data)
    var meta = HoldoutSplit(holdout_data^, var_train, var_holdout)
    return ChainTrainSplit(train^, meta^)


def no_holdout(train_data: ChainData) -> HoldoutSplit:
    """Disable holdout eval (e.g. smoke tests)."""
    var empty_x = List[Float32]()
    var empty_y = List[Float32]()
    var empty = ChainData(0, train_data.input_dim, empty_x^, empty_y^)
    return HoldoutSplit(empty^, 0.0, 0.0)


def _copy_sample(
    src: ChainData,
    sample_index: Int,
    mut dst_x: List[Float32],
    mut dst_y: List[Float32],
) -> None:
    var base = src.x_offset(sample_index)
    for i in range(src.input_dim):
        dst_x.append(src.x_data[base + i])
    dst_y.append(src.y_data[sample_index])
