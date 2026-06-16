"""Minimal LCG for deterministic init and synthetic data."""

def lcg_next(state: UInt64) -> UInt64:
    return state * 6364136223846793005 + 1


def unit_float(bits: UInt64) -> Float32:
    return Float32(bits >> 11) / Float32(1 << 53)
