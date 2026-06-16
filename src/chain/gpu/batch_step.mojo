"""GPU batched train step."""

from ..buffer import ChainData
from ..grads import ModelGrads
from ..micro_net_batch import BatchMicroNet, _ceildiv
from ..ternary import silu, silu_derivative
from .device import GpuDevice
from .ternary_matmul import (
    head_reduce_kernel,
    scale_kernel,
    swiglu_forward_kernel,
    ternary_matmul_batch_kernel,
)


def train_step_gpu(
    mut model: BatchMicroNet,
    mut gpu: GpuDevice,
    data: ChainData,
    start: Int,
    batch_size: Int,
    mut grads: ModelGrads,
) raises -> Float32:
    """GPU forward (ternary matmul + SwiGLU) + CPU STE backward."""
    var count = data.batch_size(start, batch_size)
    if count == 0:
        return 0.0

    model.sync_ternary()
    grads.zero()

    var inv_batch = 1.0 / Float32(count)
    var bh = count * model.hidden_dim
    var bx = count * model.input_dim
    var gate_size = model.gate_size()

    var x_dev = gpu.upload_list_f32(data.x_data, data.x_offset(start), bx)
    var gate_w_dev = gpu.upload_i8(model.gate_ternary, gate_size)
    var up_w_dev = gpu.upload_i8(model.up_ternary, gate_size)
    var head_w_dev = gpu.upload_i8(model.head_ternary, model.hidden_dim)

    var gate_dev = gpu.create_device_f32(bh)
    var up_dev = gpu.create_device_f32(bh)
    var hidden_dev = gpu.create_device_f32(bh)
    var y_tern_dev = gpu.create_device_f32(count)

    var matmul_blocks = _ceildiv(bh, 256)
    gpu.ctx.enqueue_function[ternary_matmul_batch_kernel, ternary_matmul_batch_kernel](
        x_dev.unsafe_ptr(),
        gate_w_dev.unsafe_ptr(),
        gate_dev.unsafe_ptr(),
        count,
        model.input_dim,
        model.hidden_dim,
        grid_dim=matmul_blocks,
        block_dim=256,
    )
    gpu.ctx.enqueue_function[ternary_matmul_batch_kernel, ternary_matmul_batch_kernel](
        x_dev.unsafe_ptr(),
        up_w_dev.unsafe_ptr(),
        up_dev.unsafe_ptr(),
        count,
        model.input_dim,
        model.hidden_dim,
        grid_dim=matmul_blocks,
        block_dim=256,
    )
    gpu.ctx.enqueue_function[swiglu_forward_kernel, swiglu_forward_kernel](
        gate_dev.unsafe_ptr(),
        up_dev.unsafe_ptr(),
        hidden_dev.unsafe_ptr(),
        count,
        model.hidden_dim,
        grid_dim=matmul_blocks,
        block_dim=256,
    )
    gpu.ctx.enqueue_function[head_reduce_kernel, head_reduce_kernel](
        hidden_dev.unsafe_ptr(),
        head_w_dev.unsafe_ptr(),
        y_tern_dev.unsafe_ptr(),
        count,
        model.hidden_dim,
        grid_dim=count,
        block_dim=1,
    )
    gpu.ctx.enqueue_function[scale_kernel, scale_kernel](
        y_tern_dev.unsafe_ptr(),
        model.alpha,
        count,
        grid_dim=_ceildiv(count, 256),
        block_dim=256,
    )
    gpu.synchronize()

    var gate_host = gpu.download_f32(gate_dev, bh)
    var up_host = gpu.download_f32(up_dev, bh)
    var pred_host = gpu.download_f32(y_tern_dev, count)

    var loss: Float32 = 0.0
    for b in range(count):
        var sample = start + b
        var target = data.y_at(sample)
        var pred = pred_host[b]
        var err = pred - target
        loss += err * err * inv_batch

        var y_tern = pred / model.alpha if model.alpha != 0.0 else 0.0
        var dL_dout = 2.0 * err * inv_batch
        grads.alpha += dL_dout * y_tern

        var dL_dy_tern = dL_dout * model.alpha
        var x_base = data.x_offset(sample)
        var act_base = b * model.hidden_dim

        for j in range(model.hidden_dim):
            var gate_j = gate_host[act_base + j]
            var up_j = up_host[act_base + j]
            var silu_gate = silu(gate_j)
            var h_j = silu_gate * up_j

            grads.head[j] += dL_dy_tern * h_j

            var dL_dh = dL_dy_tern * Float32(model.head_ternary[j])
            var dL_dgate = dL_dh * up_j * silu_derivative(gate_j)
            var dL_dup = dL_dh * silu_gate

            var row_base = j * model.input_dim
            for i in range(model.input_dim):
                var x_i = data.x_data[x_base + i]
                grads.gate[row_base + i] += dL_dgate * x_i
                grads.up[row_base + i] += dL_dup * x_i

    return loss
