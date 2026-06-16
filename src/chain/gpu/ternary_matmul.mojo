"""GPU kernels for batched ternary matmul and SwiGLU micro-network ops."""

from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.math import exp


@always_inline
def _sigmoid(x: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-x))


@always_inline
def _silu(x: Float32) -> Float32:
    return x * _sigmoid(x)


@always_inline
def _silu_derivative(x: Float32) -> Float32:
    var s = _sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def ternary_matmul_batch_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    w: UnsafePointer[Int8, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    in_dim: Int,
    hidden_dim: Int,
):
    """output[b,j] = sum_i w[j,i] * x[b,i] with ternary weights."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = batch_size * hidden_dim
    if idx >= total:
        return

    var b = idx // hidden_dim
    var j = idx % hidden_dim
    var acc: Float32 = 0.0
    var x_base = b * in_dim
    var w_row = j * in_dim
    for i in range(in_dim):
        var w_ij = w[w_row + i]
        if w_ij != 0:
            acc += Float32(w_ij) * x[x_base + i]
    output[idx] = acc


def float_matmul_batch_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    w: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    in_dim: Int,
    hidden_dim: Int,
):
    """output[b,j] = sum_i w[j,i] * x[b,i] with FP32 weights."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = batch_size * hidden_dim
    if idx >= total:
        return

    var b = idx // hidden_dim
    var j = idx % hidden_dim
    var acc: Float32 = 0.0
    var x_base = b * in_dim
    var w_row = j * in_dim
    for i in range(in_dim):
        acc += w[w_row + i] * x[x_base + i]
    output[idx] = acc


def swiglu_forward_kernel(
    gate: UnsafePointer[Float32, MutAnyOrigin],
    up: UnsafePointer[Float32, MutAnyOrigin],
    hidden: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    hidden_dim: Int,
):
    """hidden[b,j] = silu(gate[b,j]) * up[b,j]."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = batch_size * hidden_dim
    if idx >= total:
        return
    hidden[idx] = _silu(gate[idx]) * up[idx]


def head_reduce_kernel(
    hidden: UnsafePointer[Float32, MutAnyOrigin],
    head: UnsafePointer[Int8, MutAnyOrigin],
    y_tern: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    hidden_dim: Int,
):
    """y_tern[b] = sum_j head[j] * hidden[b,j]. One block per batch row."""
    var b = Int(block_idx.x)
    if b >= batch_size:
        return
    if thread_idx.x != 0:
        return

    var total: Float32 = 0.0
    var h_base = b * hidden_dim
    for j in range(hidden_dim):
        var w = head[j]
        if w != 0:
            total += Float32(w) * hidden[h_base + j]
    y_tern[b] = total


def head_reduce_f32_kernel(
    hidden: UnsafePointer[Float32, MutAnyOrigin],
    head: UnsafePointer[Float32, MutAnyOrigin],
    y_out: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    hidden_dim: Int,
):
    """y_out[b] = sum_j head[j] * hidden[b,j]. One block per batch row."""
    var b = Int(block_idx.x)
    if b >= batch_size:
        return
    if thread_idx.x != 0:
        return

    var total: Float32 = 0.0
    var h_base = b * hidden_dim
    for j in range(hidden_dim):
        total += head[j] * hidden[h_base + j]
    y_out[b] = total


def scale_kernel(
    values: UnsafePointer[Float32, MutAnyOrigin],
    scale: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    """values[i] *= *scale."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n:
        return
    values[i] = values[i] * scale[0]
