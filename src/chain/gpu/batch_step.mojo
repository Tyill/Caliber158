"""GPU batched train step (full device pipeline)."""

from ..adamw import AdamWConfig
from .buffer_pool import GpuTrainState


def train_step_gpu(
    mut state: GpuTrainState,
    start: Int,
    batch_size: Int,
    config: AdamWConfig,
) raises -> Float32:
    """GPU quantize + forward + backward + AdamW."""
    return state.train_step(start, batch_size, config)


def backward_only_gpu(
    mut state: GpuTrainState,
    start: Int,
    batch_size: Int,
) raises -> Float32:
    """Forward + backward only (grad regression)."""
    return state.backward_only(start, batch_size)
