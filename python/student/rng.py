"""LCG RNG matching src/chain/rng.mojo and init_random_weights in train.mojo."""

from __future__ import annotations

import numpy as np

_LCG_MULT = 6364136223846793005
_UINT64_MASK = (1 << 64) - 1
_INIT_SEED = 0xC158_C158


def lcg_next(state: int) -> int:
    return (state * _LCG_MULT + 1) & _UINT64_MASK


def unit_float(bits: int) -> float:
    return float(bits >> 11) / float(1 << 53)


def fisher_yates_indices(n: int, seed: int) -> np.ndarray:
    """Return shuffled index array using Mojo holdout LCG algorithm."""
    indices = np.arange(n, dtype=np.int64)
    rng = seed & _UINT64_MASK
    for i in range(n - 1, 0, -1):
        rng = lcg_next(rng)
        j = int(unit_float(rng) * float(i + 1))
        indices[i], indices[j] = indices[j], indices[i]
    return indices


def init_model_weights(
    gate_size: int,
    head_size: int,
    block2_size: int,
    scale: float = 0.1,
    *,
    w_res_size: int = 0,
    block2_init: str = "zero",
    block2_init_scale: float | None = None,
) -> dict[str, np.ndarray]:
    """Match Mojo init_random_weights: interleaved gate/up LCG; block2 zero unless lcg."""
    b2_scale = scale if block2_init_scale is None else block2_init_scale
    seed = _INIT_SEED
    gate = np.empty(gate_size, dtype=np.float32)
    up = np.empty(gate_size, dtype=np.float32)
    for i in range(gate_size):
        seed = lcg_next(seed)
        gate[i] = (unit_float(seed) * 2.0 - 1.0) * scale
        seed = lcg_next(seed)
        up[i] = (unit_float(seed) * 2.0 - 1.0) * scale
    head = np.empty(head_size, dtype=np.float32)
    for i in range(head_size):
        seed = lcg_next(seed)
        head[i] = (unit_float(seed) * 2.0 - 1.0) * scale
    if block2_init == "lcg" and block2_size > 0:
        gate2 = np.empty(block2_size, dtype=np.float32)
        up2 = np.empty(block2_size, dtype=np.float32)
        for i in range(block2_size):
            seed = lcg_next(seed)
            gate2[i] = (unit_float(seed) * 2.0 - 1.0) * b2_scale
            seed = lcg_next(seed)
            up2[i] = (unit_float(seed) * 2.0 - 1.0) * b2_scale
    else:
        gate2 = np.zeros(block2_size, dtype=np.float32)
        up2 = np.zeros(block2_size, dtype=np.float32)
    w_res = np.zeros(w_res_size, dtype=np.float32)
    return {
        "gate": gate,
        "up": up,
        "head": head,
        "gate2": gate2,
        "up2": up2,
        "w_res": w_res,
    }


def init_ffn_shadow(
    input_dim: int,
    intermediate_dim: int,
    scale: float = 0.1,
) -> dict[str, np.ndarray]:
    """LCG init for full-rank FFN shadow weights (gate/up/down)."""
    seed = _INIT_SEED
    gate = np.empty((intermediate_dim, input_dim), dtype=np.float32)
    up = np.empty((intermediate_dim, input_dim), dtype=np.float32)
    down = np.empty((input_dim, intermediate_dim), dtype=np.float32)
    for arr in (gate, up, down):
        for i in range(arr.size):
            seed = lcg_next(seed)
            arr.ravel()[i] = (unit_float(seed) * 2.0 - 1.0) * scale
    return {"gate": gate, "up": up, "down": down}


def init_ffn_lowrank_shadow(
    input_dim: int,
    intermediate_dim: int,
    rank: int,
    scale: float = 0.1,
) -> dict[str, np.ndarray]:
    """LCG init for low-rank FFN factors (A [I,r], B [r,D] per projection)."""
    seed = _INIT_SEED

    def fill(shape: tuple[int, ...]) -> np.ndarray:
        nonlocal seed
        arr = np.empty(shape, dtype=np.float32)
        for i in range(arr.size):
            seed = lcg_next(seed)
            arr.ravel()[i] = (unit_float(seed) * 2.0 - 1.0) * scale
        return arr

    return {
        "gate_a": fill((intermediate_dim, rank)),
        "gate_b": fill((rank, input_dim)),
        "up_a": fill((intermediate_dim, rank)),
        "up_b": fill((rank, input_dim)),
        "down_a": fill((input_dim, rank)),
        "down_b": fill((rank, intermediate_dim)),
    }
