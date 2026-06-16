"""GPU STE backward kernels (partial + batch reduce, no atomics)."""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import exp

from .device import (
    _ceildiv,
    f32_ptr_offset,
    reduce_batch_dim_gate_up_kernel,
    reduce_batch_dim_head_kernel,
    reduce_sum_f32_kernel,
)


@always_inline
def _sigmoid(x: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-x))


@always_inline
def _silu_derivative(x: Float32) -> Float32:
    var s = _sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def mse_err_kernel(
    pred: UnsafePointer[Float32, MutAnyOrigin],
    y: UnsafePointer[Float32, MutAnyOrigin],
    err: UnsafePointer[Float32, MutAnyOrigin],
    loss_partial: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    inv_batch: Float32,
):
    """err[b] = pred[b] - y[b]; loss_partial[b] = err² * inv_batch."""
    var b = Int(block_idx.x * block_dim.x + thread_idx.x)
    if b >= batch_size:
        return
    var e = pred[b] - y[b]
    err[b] = e
    loss_partial[b] = e * e * inv_batch


def backward_alpha_partial_kernel(
    pred: UnsafePointer[Float32, MutAnyOrigin],
    err: UnsafePointer[Float32, MutAnyOrigin],
    alpha: UnsafePointer[Float32, MutAnyOrigin],
    alpha_partial: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    inv_batch: Float32,
):
    """alpha_partial[b] = dL_dout * y_tern for sample b."""
    var b = Int(block_idx.x * block_dim.x + thread_idx.x)
    if b >= batch_size:
        return
    var a = alpha[0]
    var dL_dout = 2.0 * err[b] * inv_batch
    var y_tern = pred[b] / a if a != 0.0 else 0.0
    alpha_partial[b] = dL_dout * y_tern


def backward_head_partial_kernel(
    err: UnsafePointer[Float32, MutAnyOrigin],
    hidden: UnsafePointer[Float32, MutAnyOrigin],
    alpha: UnsafePointer[Float32, MutAnyOrigin],
    head_partial: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    hidden_dim: Int,
    inv_batch: Float32,
):
    """head_partial[b,j] = dL_dy_tern * h_j."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = batch_size * hidden_dim
    if idx >= total:
        return
    var b = idx // hidden_dim
    var j = idx % hidden_dim
    var dL_dy_tern = 2.0 * err[b] * inv_batch * alpha[0]
    var h_base = b * hidden_dim
    head_partial[idx] = dL_dy_tern * hidden[h_base + j]


def backward_gate_up_partial_kernel(
    err: UnsafePointer[Float32, MutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    hidden: UnsafePointer[Float32, MutAnyOrigin],
    gate_act: UnsafePointer[Float32, MutAnyOrigin],
    up_act: UnsafePointer[Float32, MutAnyOrigin],
    head_tern: UnsafePointer[Int8, MutAnyOrigin],
    alpha: UnsafePointer[Float32, MutAnyOrigin],
    gate_partial: UnsafePointer[Float32, MutAnyOrigin],
    up_partial: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    hidden_dim: Int,
    input_dim: Int,
    inv_batch: Float32,
):
    """gate/up partial[b,j,i] from forward activations (no host recompute)."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = batch_size * hidden_dim * input_dim
    if idx >= total:
        return

    var elems_per_batch = hidden_dim * input_dim
    var b = idx // elems_per_batch
    var rem = idx % elems_per_batch
    var j = rem // input_dim
    var i = rem % input_dim

    var dL_dy_tern = 2.0 * err[b] * inv_batch * alpha[0]
    var act_base = b * hidden_dim
    var gate_j = gate_act[act_base + j]
    var up_j = up_act[act_base + j]
    var h_j = hidden[act_base + j]
    var silu_gate = h_j / up_j if up_j != 0.0 else 0.0

    var dL_dh = dL_dy_tern * Float32(head_tern[j])
    var dL_dgate = dL_dh * up_j * _silu_derivative(gate_j)
    var dL_dup = dL_dh * silu_gate

    var x_base = b * input_dim
    gate_partial[idx] = dL_dgate * x[x_base + i]
    up_partial[idx] = dL_dup * x[x_base + i]


def backward_gate_up_partial_f32_kernel(
    err: UnsafePointer[Float32, MutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    hidden: UnsafePointer[Float32, MutAnyOrigin],
    gate_act: UnsafePointer[Float32, MutAnyOrigin],
    up_act: UnsafePointer[Float32, MutAnyOrigin],
    head_shadow: UnsafePointer[Float32, MutAnyOrigin],
    alpha: UnsafePointer[Float32, MutAnyOrigin],
    gate_partial: UnsafePointer[Float32, MutAnyOrigin],
    up_partial: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    hidden_dim: Int,
    input_dim: Int,
    inv_batch: Float32,
):
    """FP32 shadow head weights in gate/up partial backward."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = batch_size * hidden_dim * input_dim
    if idx >= total:
        return

    var elems_per_batch = hidden_dim * input_dim
    var b = idx // elems_per_batch
    var rem = idx % elems_per_batch
    var j = rem // input_dim
    var i = rem % input_dim

    var dL_dy_tern = 2.0 * err[b] * inv_batch * alpha[0]
    var act_base = b * hidden_dim
    var gate_j = gate_act[act_base + j]
    var up_j = up_act[act_base + j]
    var h_j = hidden[act_base + j]
    var silu_gate = h_j / up_j if up_j != 0.0 else 0.0

    var dL_dh = dL_dy_tern * head_shadow[j]
    var dL_dgate = dL_dh * up_j * _silu_derivative(gate_j)
    var dL_dup = dL_dh * silu_gate

    var x_base = b * input_dim
    gate_partial[idx] = dL_dgate * x[x_base + i]
    up_partial[idx] = dL_dup * x[x_base + i]


def enqueue_backward(
    ctx: DeviceContext,
    x_dev: DeviceBuffer[DType.float32],
    x_batch_offset: Int,
    y_dev: DeviceBuffer[DType.float32],
    y_batch_offset: Int,
    gate_act_dev: DeviceBuffer[DType.float32],
    up_act_dev: DeviceBuffer[DType.float32],
    hidden_dev: DeviceBuffer[DType.float32],
    pred_dev: DeviceBuffer[DType.float32],
    head_tern_dev: DeviceBuffer[DType.int8],
    alpha_dev: DeviceBuffer[DType.float32],
    mut err_dev: DeviceBuffer[DType.float32],
    mut loss_partial_dev: DeviceBuffer[DType.float32],
    mut grad_alpha_partial_dev: DeviceBuffer[DType.float32],
    mut grad_head_partial_dev: DeviceBuffer[DType.float32],
    mut grad_gate_partial_dev: DeviceBuffer[DType.float32],
    mut grad_up_partial_dev: DeviceBuffer[DType.float32],
    mut grad_gate_dev: DeviceBuffer[DType.float32],
    mut grad_up_dev: DeviceBuffer[DType.float32],
    mut grad_head_dev: DeviceBuffer[DType.float32],
    mut grad_alpha_dev: DeviceBuffer[DType.float32],
    mut loss_scalar_dev: DeviceBuffer[DType.float32],
    batch_size: Int,
    hidden_dim: Int,
    input_dim: Int,
) raises -> None:
    """Full backward: partials + reduce over batch dim."""
    var inv_batch = 1.0 / Float32(batch_size)
    var bh = batch_size * hidden_dim
    var bhd = batch_size * hidden_dim * input_dim
    var blocks_b = _ceildiv(batch_size, 256)
    var blocks_bh = _ceildiv(bh, 256)
    var blocks_bhd = _ceildiv(bhd, 256)
    var blocks_hd = _ceildiv(hidden_dim * input_dim, 256)
    var blocks_h = _ceildiv(hidden_dim, 256)

    var x_ptr = f32_ptr_offset(x_dev.unsafe_ptr(), x_batch_offset)
    var y_ptr = f32_ptr_offset(y_dev.unsafe_ptr(), y_batch_offset)

    ctx.enqueue_function[mse_err_kernel, mse_err_kernel](
        pred_dev.unsafe_ptr(),
        y_ptr,
        err_dev.unsafe_ptr(),
        loss_partial_dev.unsafe_ptr(),
        batch_size,
        inv_batch,
        grid_dim=blocks_b,
        block_dim=256,
    )
    ctx.enqueue_function[backward_alpha_partial_kernel, backward_alpha_partial_kernel](
        pred_dev.unsafe_ptr(),
        err_dev.unsafe_ptr(),
        alpha_dev.unsafe_ptr(),
        grad_alpha_partial_dev.unsafe_ptr(),
        batch_size,
        inv_batch,
        grid_dim=blocks_b,
        block_dim=256,
    )
    ctx.enqueue_function[backward_head_partial_kernel, backward_head_partial_kernel](
        err_dev.unsafe_ptr(),
        hidden_dev.unsafe_ptr(),
        alpha_dev.unsafe_ptr(),
        grad_head_partial_dev.unsafe_ptr(),
        batch_size,
        hidden_dim,
        inv_batch,
        grid_dim=blocks_bh,
        block_dim=256,
    )
    ctx.enqueue_function[backward_gate_up_partial_kernel, backward_gate_up_partial_kernel](
        err_dev.unsafe_ptr(),
        x_ptr,
        hidden_dev.unsafe_ptr(),
        gate_act_dev.unsafe_ptr(),
        up_act_dev.unsafe_ptr(),
        head_tern_dev.unsafe_ptr(),
        alpha_dev.unsafe_ptr(),
        grad_gate_partial_dev.unsafe_ptr(),
        grad_up_partial_dev.unsafe_ptr(),
        batch_size,
        hidden_dim,
        input_dim,
        inv_batch,
        grid_dim=blocks_bhd,
        block_dim=256,
    )

    ctx.enqueue_function[reduce_sum_f32_kernel, reduce_sum_f32_kernel](
        loss_partial_dev.unsafe_ptr(),
        loss_scalar_dev.unsafe_ptr(),
        batch_size,
        grid_dim=1,
        block_dim=1,
    )
    ctx.enqueue_function[reduce_sum_f32_kernel, reduce_sum_f32_kernel](
        grad_alpha_partial_dev.unsafe_ptr(),
        grad_alpha_dev.unsafe_ptr(),
        batch_size,
        grid_dim=1,
        block_dim=1,
    )
    ctx.enqueue_function[reduce_batch_dim_head_kernel, reduce_batch_dim_head_kernel](
        grad_head_partial_dev.unsafe_ptr(),
        grad_head_dev.unsafe_ptr(),
        batch_size,
        hidden_dim,
        grid_dim=blocks_h,
        block_dim=256,
    )
    ctx.enqueue_function[reduce_batch_dim_gate_up_kernel, reduce_batch_dim_gate_up_kernel](
        grad_gate_partial_dev.unsafe_ptr(),
        grad_gate_dev.unsafe_ptr(),
        batch_size,
        hidden_dim,
        input_dim,
        grid_dim=blocks_hd,
        block_dim=256,
    )
    ctx.enqueue_function[reduce_batch_dim_gate_up_kernel, reduce_batch_dim_gate_up_kernel](
        grad_up_partial_dev.unsafe_ptr(),
        grad_up_dev.unsafe_ptr(),
        batch_size,
        hidden_dim,
        input_dim,
        grid_dim=blocks_hd,
        block_dim=256,
    )


def enqueue_backward_fp32(
    ctx: DeviceContext,
    x_dev: DeviceBuffer[DType.float32],
    x_batch_offset: Int,
    y_dev: DeviceBuffer[DType.float32],
    y_batch_offset: Int,
    gate_act_dev: DeviceBuffer[DType.float32],
    up_act_dev: DeviceBuffer[DType.float32],
    hidden_dev: DeviceBuffer[DType.float32],
    pred_dev: DeviceBuffer[DType.float32],
    head_shadow_dev: DeviceBuffer[DType.float32],
    alpha_dev: DeviceBuffer[DType.float32],
    mut err_dev: DeviceBuffer[DType.float32],
    mut loss_partial_dev: DeviceBuffer[DType.float32],
    mut grad_alpha_partial_dev: DeviceBuffer[DType.float32],
    mut grad_head_partial_dev: DeviceBuffer[DType.float32],
    mut grad_gate_partial_dev: DeviceBuffer[DType.float32],
    mut grad_up_partial_dev: DeviceBuffer[DType.float32],
    mut grad_gate_dev: DeviceBuffer[DType.float32],
    mut grad_up_dev: DeviceBuffer[DType.float32],
    mut grad_head_dev: DeviceBuffer[DType.float32],
    mut grad_alpha_dev: DeviceBuffer[DType.float32],
    mut loss_scalar_dev: DeviceBuffer[DType.float32],
    batch_size: Int,
    hidden_dim: Int,
    input_dim: Int,
) raises -> None:
    """Full backward with FP32 shadow head (no ternary)."""
    var inv_batch = 1.0 / Float32(batch_size)
    var bh = batch_size * hidden_dim
    var bhd = batch_size * hidden_dim * input_dim
    var blocks_b = _ceildiv(batch_size, 256)
    var blocks_bh = _ceildiv(bh, 256)
    var blocks_bhd = _ceildiv(bhd, 256)
    var blocks_hd = _ceildiv(hidden_dim * input_dim, 256)
    var blocks_h = _ceildiv(hidden_dim, 256)

    var x_ptr = f32_ptr_offset(x_dev.unsafe_ptr(), x_batch_offset)
    var y_ptr = f32_ptr_offset(y_dev.unsafe_ptr(), y_batch_offset)

    ctx.enqueue_function[mse_err_kernel, mse_err_kernel](
        pred_dev.unsafe_ptr(),
        y_ptr,
        err_dev.unsafe_ptr(),
        loss_partial_dev.unsafe_ptr(),
        batch_size,
        inv_batch,
        grid_dim=blocks_b,
        block_dim=256,
    )
    ctx.enqueue_function[backward_alpha_partial_kernel, backward_alpha_partial_kernel](
        pred_dev.unsafe_ptr(),
        err_dev.unsafe_ptr(),
        alpha_dev.unsafe_ptr(),
        grad_alpha_partial_dev.unsafe_ptr(),
        batch_size,
        inv_batch,
        grid_dim=blocks_b,
        block_dim=256,
    )
    ctx.enqueue_function[backward_head_partial_kernel, backward_head_partial_kernel](
        err_dev.unsafe_ptr(),
        hidden_dev.unsafe_ptr(),
        alpha_dev.unsafe_ptr(),
        grad_head_partial_dev.unsafe_ptr(),
        batch_size,
        hidden_dim,
        inv_batch,
        grid_dim=blocks_bh,
        block_dim=256,
    )
    ctx.enqueue_function[
        backward_gate_up_partial_f32_kernel, backward_gate_up_partial_f32_kernel
    ](
        err_dev.unsafe_ptr(),
        x_ptr,
        hidden_dev.unsafe_ptr(),
        gate_act_dev.unsafe_ptr(),
        up_act_dev.unsafe_ptr(),
        head_shadow_dev.unsafe_ptr(),
        alpha_dev.unsafe_ptr(),
        grad_gate_partial_dev.unsafe_ptr(),
        grad_up_partial_dev.unsafe_ptr(),
        batch_size,
        hidden_dim,
        input_dim,
        inv_batch,
        grid_dim=blocks_bhd,
        block_dim=256,
    )

    ctx.enqueue_function[reduce_sum_f32_kernel, reduce_sum_f32_kernel](
        loss_partial_dev.unsafe_ptr(),
        loss_scalar_dev.unsafe_ptr(),
        batch_size,
        grid_dim=1,
        block_dim=1,
    )
    ctx.enqueue_function[reduce_sum_f32_kernel, reduce_sum_f32_kernel](
        grad_alpha_partial_dev.unsafe_ptr(),
        grad_alpha_dev.unsafe_ptr(),
        batch_size,
        grid_dim=1,
        block_dim=1,
    )
    ctx.enqueue_function[reduce_batch_dim_head_kernel, reduce_batch_dim_head_kernel](
        grad_head_partial_dev.unsafe_ptr(),
        grad_head_dev.unsafe_ptr(),
        batch_size,
        hidden_dim,
        grid_dim=blocks_h,
        block_dim=256,
    )
    ctx.enqueue_function[reduce_batch_dim_gate_up_kernel, reduce_batch_dim_gate_up_kernel](
        grad_gate_partial_dev.unsafe_ptr(),
        grad_gate_dev.unsafe_ptr(),
        batch_size,
        hidden_dim,
        input_dim,
        grid_dim=blocks_hd,
        block_dim=256,
    )
    ctx.enqueue_function[reduce_batch_dim_gate_up_kernel, reduce_batch_dim_gate_up_kernel](
        grad_up_partial_dev.unsafe_ptr(),
        grad_up_dev.unsafe_ptr(),
        batch_size,
        hidden_dim,
        input_dim,
        grid_dim=blocks_hd,
        block_dim=256,
    )
