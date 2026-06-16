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
) -> dict[str, np.ndarray]:
    """Match Mojo init_random_weights: interleaved gate/up LCG; block2 zero."""
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
    gate2 = np.zeros(block2_size, dtype=np.float32)
    up2 = np.zeros(block2_size, dtype=np.float32)
    return {
        "gate": gate,
        "up": up,
        "head": head,
        "gate2": gate2,
        "up2": up2,
    }
