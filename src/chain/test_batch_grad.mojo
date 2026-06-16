"""Regression: batched CPU train_step matches per-sample reference gradients."""

from std.testing import assert_true

from .arch import ArchKind
from .buffer import ChainData
from .dataset import ChainDataset
from .device import cuda_available
from .grads import ModelGrads
from .gpu.batch_step import backward_only_gpu
from .gpu.buffer_pool import GpuTrainState
from .micro_net_batch import BatchMicroNet
from .ternary import silu, silu_derivative, ternary_matvec
from .train import init_random_weights


def _copy_model_weights(mut dst: BatchMicroNet, src: BatchMicroNet) -> None:
    for i in range(len(src.gate_shadow)):
        dst.gate_shadow[i] = src.gate_shadow[i]
        dst.up_shadow[i] = src.up_shadow[i]
    for i in range(len(src.gate2_shadow)):
        dst.gate2_shadow[i] = src.gate2_shadow[i]
        dst.up2_shadow[i] = src.up2_shadow[i]
    for i in range(len(src.head_shadow)):
        dst.head_shadow[i] = src.head_shadow[i]
    dst.alpha = src.alpha


def _reference_batch_grad_v0(
    mut model: BatchMicroNet,
    data: ChainData,
    start: Int,
    batch_size: Int,
    mut grads: ModelGrads,
) raises -> Float32:
    """Per-sample reference matching the old accumulate_grad path."""
    var count = data.batch_size(start, batch_size)
    grads.zero()
    model.sync_ternary()

    var inv_batch = 1.0 / Float32(count)
    var loss: Float32 = 0.0
    var gate_buf = List[Float32](capacity=model.hidden_dim)
    var up_buf = List[Float32](capacity=model.hidden_dim)
    for _ in range(model.hidden_dim):
        gate_buf.append(0.0)
        up_buf.append(0.0)

    var x_row = List[Float32](capacity=model.input_dim)

    for sample in range(start, start + count):
        x_row.clear()
        var x_base = data.x_offset(sample)
        for i in range(model.input_dim):
            x_row.append(data.x_data[x_base + i])
        var target = data.y_at(sample)

        ternary_matvec(
            model.gate_ternary,
            x_row,
            model.input_dim,
            model.hidden_dim,
            gate_buf,
        )
        ternary_matvec(
            model.up_ternary,
            x_row,
            model.input_dim,
            model.hidden_dim,
            up_buf,
        )

        var y_tern: Float32 = 0.0
        for j in range(model.hidden_dim):
            var h = silu(gate_buf[j]) * up_buf[j]
            var w = model.head_ternary[j]
            if w != 0:
                y_tern += Float32(w) * h

        var pred = model.alpha * y_tern
        var err = pred - target
        loss += err * err * inv_batch

        var dL_dout = 2.0 * err * inv_batch
        grads.alpha += dL_dout * y_tern

        var dL_dy_tern = dL_dout * model.alpha
        for j in range(model.hidden_dim):
            var gate_j = gate_buf[j]
            var up_j = up_buf[j]
            var silu_gate = silu(gate_j)
            var h_j = silu_gate * up_j

            grads.head[j] += dL_dy_tern * h_j

            var dL_dh = dL_dy_tern * Float32(model.head_ternary[j])
            var dL_dgate = dL_dh * up_j * silu_derivative(gate_j)
            var dL_dup = dL_dh * silu_gate

            var row_base = j * model.input_dim
            for i in range(model.input_dim):
                var x_i = x_row[i]
                grads.gate[row_base + i] += dL_dgate * x_i
                grads.up[row_base + i] += dL_dup * x_i

    return loss


def _reference_batch_grad_v1(
    mut model: BatchMicroNet,
    data: ChainData,
    start: Int,
    batch_size: Int,
    mut grads: ModelGrads,
) raises -> Float32:
    var count = data.batch_size(start, batch_size)
    grads.zero()
    model.sync_ternary()

    var inv_batch = 1.0 / Float32(count)
    var loss: Float32 = 0.0
    var h_dim = model.hidden_dim
    var d_dim = model.input_dim

    var gate1 = List[Float32](capacity=h_dim)
    var up1 = List[Float32](capacity=h_dim)
    var h0 = List[Float32](capacity=h_dim)
    var gate2 = List[Float32](capacity=h_dim)
    var up2 = List[Float32](capacity=h_dim)
    var h2 = List[Float32](capacity=h_dim)
    var h1 = List[Float32](capacity=h_dim)
    var x_row = List[Float32](capacity=d_dim)
    for _ in range(h_dim):
        gate1.append(0.0)
        up1.append(0.0)
        h0.append(0.0)
        gate2.append(0.0)
        up2.append(0.0)
        h2.append(0.0)
        h1.append(0.0)

    for sample in range(start, start + count):
        x_row.clear()
        var x_base = data.x_offset(sample)
        for i in range(d_dim):
            x_row.append(data.x_data[x_base + i])
        var target = data.y_at(sample)

        ternary_matvec(model.gate_ternary, x_row, d_dim, h_dim, gate1)
        ternary_matvec(model.up_ternary, x_row, d_dim, h_dim, up1)
        for j in range(h_dim):
            h0[j] = silu(gate1[j]) * up1[j]

        ternary_matvec(model.gate2_ternary, h0, h_dim, h_dim, gate2)
        ternary_matvec(model.up2_ternary, h0, h_dim, h_dim, up2)
        for j in range(h_dim):
            h2[j] = silu(gate2[j]) * up2[j]
            h1[j] = h0[j] + h2[j]

        var y_tern: Float32 = 0.0
        for j in range(h_dim):
            var w = model.head_ternary[j]
            if w != 0:
                y_tern += Float32(w) * h1[j]

        var pred = model.alpha * y_tern
        var err = pred - target
        loss += err * err * inv_batch

        var dL_dout = 2.0 * err * inv_batch
        grads.alpha += dL_dout * y_tern
        var dL_dy_tern = dL_dout * model.alpha

        var dL_dh1 = List[Float32](capacity=h_dim)
        var dL_dh0 = List[Float32](capacity=h_dim)
        for _ in range(h_dim):
            dL_dh1.append(0.0)
            dL_dh0.append(0.0)

        for j in range(h_dim):
            grads.head[j] += dL_dy_tern * h1[j]
            dL_dh1[j] = dL_dy_tern * Float32(model.head_ternary[j])

        for j in range(h_dim):
            var dL_dh2 = dL_dh1[j]
            var dL_dgate2 = dL_dh2 * up2[j] * silu_derivative(gate2[j])
            var dL_dup2 = dL_dh2 * silu(gate2[j])
            var row2 = j * h_dim
            for i in range(h_dim):
                var wg = Float32(model.gate2_ternary[row2 + i])
                var wu = Float32(model.up2_ternary[row2 + i])
                dL_dh0[i] += dL_dgate2 * wg + dL_dup2 * wu
                grads.gate2[row2 + i] += dL_dgate2 * h0[i]
                grads.up2[row2 + i] += dL_dup2 * h0[i]

        for j in range(h_dim):
            dL_dh0[j] += dL_dh1[j]

        for j in range(h_dim):
            var dL_dgate1 = dL_dh0[j] * up1[j] * silu_derivative(gate1[j])
            var dL_dup1 = dL_dh0[j] * silu(gate1[j])
            var row1 = j * d_dim
            for i in range(d_dim):
                grads.gate[row1 + i] += dL_dgate1 * x_row[i]
                grads.up[row1 + i] += dL_dup1 * x_row[i]

    return loss


def _max_grad_diff(a: ModelGrads, b: ModelGrads) -> Float32:
    var max_diff: Float32 = 0.0
    if abs(a.alpha - b.alpha) > max_diff:
        max_diff = abs(a.alpha - b.alpha)
    for i in range(len(a.gate)):
        var d = abs(a.gate[i] - b.gate[i])
        if d > max_diff:
            max_diff = d
    for i in range(len(a.up)):
        var d = abs(a.up[i] - b.up[i])
        if d > max_diff:
            max_diff = d
    for i in range(len(a.gate2)):
        var d = abs(a.gate2[i] - b.gate2[i])
        if d > max_diff:
            max_diff = d
    for i in range(len(a.up2)):
        var d = abs(a.up2[i] - b.up2[i])
        if d > max_diff:
            max_diff = d
    for i in range(len(a.head)):
        var d = abs(a.head[i] - b.head[i])
        if d > max_diff:
            max_diff = d
    return max_diff


def run_batch_grad_regression_test() raises -> None:
    var dataset = ChainDataset.synthetic(128, 32)
    var data = ChainData.from_dataset(dataset)
    var hidden_dim = 16

    var model_ref = BatchMicroNet(32, hidden_dim)
    var model_batch = BatchMicroNet(32, hidden_dim)
    init_random_weights(model_ref, 0.1)
    _copy_model_weights(model_batch, model_ref)

    var grads_ref = ModelGrads.zeros_for_model(32, hidden_dim, ArchKind.v0())
    var grads_batch = ModelGrads.zeros_for_model(32, hidden_dim, ArchKind.v0())

    var loss_ref = _reference_batch_grad_v0(model_ref, data, 0, 32, grads_ref)
    var loss_batch = model_batch.train_step_cpu(data, 0, 32, grads_batch)

    assert_true(abs(loss_ref - loss_batch) < 1e-5, "v0 loss mismatch")
    assert_true(_max_grad_diff(grads_ref, grads_batch) < 2e-4, "v0 grad mismatch")


def run_batch_grad_v1_regression_test() raises -> None:
    var dataset = ChainDataset.synthetic(128, 32)
    var data = ChainData.from_dataset(dataset)
    var hidden_dim = 16

    var model_ref = BatchMicroNet(32, hidden_dim, arch=ArchKind.v1())
    var model_batch = BatchMicroNet(32, hidden_dim, arch=ArchKind.v1())
    init_random_weights(model_ref, 0.1)
    _copy_model_weights(model_batch, model_ref)

    var grads_ref = ModelGrads.zeros_for_model(32, hidden_dim, ArchKind.v1())
    var grads_batch = ModelGrads.zeros_for_model(32, hidden_dim, ArchKind.v1())

    var loss_ref = _reference_batch_grad_v1(model_ref, data, 0, 32, grads_ref)
    var loss_batch = model_batch.train_step_cpu(data, 0, 32, grads_batch)

    assert_true(abs(loss_ref - loss_batch) < 1e-5, "v1 loss mismatch")
    assert_true(_max_grad_diff(grads_ref, grads_batch) < 2e-4, "v1 grad mismatch")


def run_gpu_backward_regression_test() raises -> None:
    if not cuda_available():
        print("test-grad-gpu: skipped (no CUDA device)")
        return

    var dataset = ChainDataset.synthetic(128, 32)
    var data = ChainData.from_dataset(dataset)
    var hidden_dim = 16
    var batch_size = 32

    var model_cpu = BatchMicroNet(32, hidden_dim)
    var model_gpu = BatchMicroNet(32, hidden_dim)
    init_random_weights(model_cpu, 0.1)
    _copy_model_weights(model_gpu, model_cpu)

    var grads_cpu = ModelGrads.zeros_for_model(32, hidden_dim, ArchKind.v0())
    var grads_gpu = ModelGrads.zeros_for_model(32, hidden_dim, ArchKind.v0())

    var loss_cpu = model_cpu.train_step_cpu(data, 0, batch_size, grads_cpu)

    var state = GpuTrainState(data, model_gpu, batch_size)
    var loss_gpu = backward_only_gpu(state, 0, batch_size)
    state.download_grads(grads_gpu)

    assert_true(abs(loss_cpu - loss_gpu) < 1e-5, "gpu v0 loss mismatch")
    assert_true(_max_grad_diff(grads_cpu, grads_gpu) < 2e-4, "gpu v0 grad mismatch")


def run_gpu_backward_v1_regression_test() raises -> None:
    if not cuda_available():
        print("test-grad-gpu-v1: skipped (no CUDA device)")
        return

    var dataset = ChainDataset.synthetic(128, 32)
    var data = ChainData.from_dataset(dataset)
    var hidden_dim = 16
    var batch_size = 32

    var model_cpu = BatchMicroNet(32, hidden_dim, arch=ArchKind.v1())
    var model_gpu = BatchMicroNet(32, hidden_dim, arch=ArchKind.v1())
    init_random_weights(model_cpu, 0.1)
    _copy_model_weights(model_gpu, model_cpu)

    var grads_cpu = ModelGrads.zeros_for_model(32, hidden_dim, ArchKind.v1())
    var grads_gpu = ModelGrads.zeros_for_model(32, hidden_dim, ArchKind.v1())

    var loss_cpu = model_cpu.train_step_cpu(data, 0, batch_size, grads_cpu)

    var state = GpuTrainState(data, model_gpu, batch_size)
    var loss_gpu = backward_only_gpu(state, 0, batch_size)
    state.download_grads(grads_gpu)

    assert_true(abs(loss_cpu - loss_gpu) < 1e-5, "gpu v1 loss mismatch")
    assert_true(_max_grad_diff(grads_cpu, grads_gpu) < 2e-4, "gpu v1 grad mismatch")
