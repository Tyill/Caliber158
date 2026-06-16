"""AdamW optimizer kernels on device."""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt

from ..adamw import AdamWConfig
from .device import _ceildiv


def adamw_update_kernel(
    params: UnsafePointer[Float32, MutAnyOrigin],
    grads: UnsafePointer[Float32, MutAnyOrigin],
    m: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    bias_corr1: Float32,
    bias_corr2: Float32,
    learning_rate: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n:
        return

    var g = grads[i]
    m[i] = beta1 * m[i] + (1.0 - beta1) * g
    v[i] = beta2 * v[i] + (1.0 - beta2) * g * g

    var m_hat = m[i] / bias_corr1
    var v_hat = v[i] / bias_corr2
    var update = m_hat / (sqrt(v_hat) + eps) + weight_decay * params[i]
    params[i] -= learning_rate * update


def adamw_scalar_kernel(
    param: UnsafePointer[Float32, MutAnyOrigin],
    grad: UnsafePointer[Float32, MutAnyOrigin],
    m: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
    bias_corr1: Float32,
    bias_corr2: Float32,
    learning_rate: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
):
    if thread_idx.x != 0 or block_idx.x != 0:
        return

    var g = grad[0]
    m[0] = beta1 * m[0] + (1.0 - beta1) * g
    v[0] = beta2 * v[0] + (1.0 - beta2) * g * g

    var m_hat = m[0] / bias_corr1
    var v_hat = v[0] / bias_corr2
    var update = m_hat / (sqrt(v_hat) + eps) + weight_decay * param[0]
    param[0] -= learning_rate * update


def enqueue_adamw_weight_list(
    ctx: DeviceContext,
    mut params: DeviceBuffer[DType.float32],
    grads: DeviceBuffer[DType.float32],
    mut m: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32],
    n: Int,
    bias_corr1: Float32,
    bias_corr2: Float32,
    config: AdamWConfig,
) raises -> None:
    var blocks = _ceildiv(n, 256)
    ctx.enqueue_function[adamw_update_kernel, adamw_update_kernel](
        params.unsafe_ptr(),
        grads.unsafe_ptr(),
        m.unsafe_ptr(),
        v.unsafe_ptr(),
        n,
        bias_corr1,
        bias_corr2,
        config.learning_rate,
        config.beta1,
        config.beta2,
        config.eps,
        config.weight_decay,
        grid_dim=blocks,
        block_dim=256,
    )


def enqueue_adamw_apply(
    ctx: DeviceContext,
    mut gate_shadow: DeviceBuffer[DType.float32],
    mut up_shadow: DeviceBuffer[DType.float32],
    mut head_shadow: DeviceBuffer[DType.float32],
    mut alpha_dev: DeviceBuffer[DType.float32],
    grad_gate: DeviceBuffer[DType.float32],
    grad_up: DeviceBuffer[DType.float32],
    grad_head: DeviceBuffer[DType.float32],
    grad_alpha: DeviceBuffer[DType.float32],
    mut gate_m: DeviceBuffer[DType.float32],
    mut gate_v: DeviceBuffer[DType.float32],
    mut up_m: DeviceBuffer[DType.float32],
    mut up_v: DeviceBuffer[DType.float32],
    mut head_m: DeviceBuffer[DType.float32],
    mut head_v: DeviceBuffer[DType.float32],
    mut alpha_m: DeviceBuffer[DType.float32],
    mut alpha_v: DeviceBuffer[DType.float32],
    gate_size: Int,
    head_size: Int,
    bias_corr1: Float32,
    bias_corr2: Float32,
    config: AdamWConfig,
) raises -> None:
    var blocks_gate = _ceildiv(gate_size, 256)
    var blocks_head = _ceildiv(head_size, 256)

    var lr = config.learning_rate
    var b1 = config.beta1
    var b2 = config.beta2
    var eps = config.eps
    var wd = config.weight_decay

    ctx.enqueue_function[adamw_update_kernel, adamw_update_kernel](
        gate_shadow.unsafe_ptr(),
        grad_gate.unsafe_ptr(),
        gate_m.unsafe_ptr(),
        gate_v.unsafe_ptr(),
        gate_size,
        bias_corr1,
        bias_corr2,
        lr,
        b1,
        b2,
        eps,
        wd,
        grid_dim=blocks_gate,
        block_dim=256,
    )
    ctx.enqueue_function[adamw_update_kernel, adamw_update_kernel](
        up_shadow.unsafe_ptr(),
        grad_up.unsafe_ptr(),
        up_m.unsafe_ptr(),
        up_v.unsafe_ptr(),
        gate_size,
        bias_corr1,
        bias_corr2,
        lr,
        b1,
        b2,
        eps,
        wd,
        grid_dim=blocks_gate,
        block_dim=256,
    )
    ctx.enqueue_function[adamw_update_kernel, adamw_update_kernel](
        head_shadow.unsafe_ptr(),
        grad_head.unsafe_ptr(),
        head_m.unsafe_ptr(),
        head_v.unsafe_ptr(),
        head_size,
        bias_corr1,
        bias_corr2,
        lr,
        b1,
        b2,
        eps,
        wd,
        grid_dim=blocks_head,
        block_dim=256,
    )
    ctx.enqueue_function[adamw_scalar_kernel, adamw_scalar_kernel](
        alpha_dev.unsafe_ptr(),
        grad_alpha.unsafe_ptr(),
        alpha_m.unsafe_ptr(),
        alpha_v.unsafe_ptr(),
        bias_corr1,
        bias_corr2,
        lr,
        b1,
        b2,
        eps,
        wd,
        grid_dim=1,
        block_dim=1,
    )
