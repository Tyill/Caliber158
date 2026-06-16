"""Regression: batched CPU train_step matches per-sample reference gradients."""

from std.testing import assert_true

from .buffer import ChainData
from .dataset import ChainDataset
from .grads import ModelGrads
from .micro_net_batch import BatchMicroNet
from .ternary import silu, silu_derivative, ternary_matvec
from .train import init_random_weights


def _reference_batch_grad(
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
    init_random_weights(model_batch, 0.1)

    for i in range(len(model_ref.gate_shadow)):
        model_batch.gate_shadow[i] = model_ref.gate_shadow[i]
        model_batch.up_shadow[i] = model_ref.up_shadow[i]
    for i in range(len(model_ref.head_shadow)):
        model_batch.head_shadow[i] = model_ref.head_shadow[i]
    model_batch.alpha = model_ref.alpha

    var grads_ref = ModelGrads.zeros(
        len(model_ref.gate_shadow),
        len(model_ref.up_shadow),
        len(model_ref.head_shadow),
    )
    var grads_batch = ModelGrads.zeros(
        len(model_batch.gate_shadow),
        len(model_batch.up_shadow),
        len(model_batch.head_shadow),
    )

    var loss_ref = _reference_batch_grad(model_ref, data, 0, 32, grads_ref)
    var loss_batch = model_batch.train_step_cpu(data, 0, 32, grads_batch)

    assert_true(abs(loss_ref - loss_batch) < 1e-5, "loss mismatch")
    assert_true(_max_grad_diff(grads_ref, grads_batch) < 1e-5, "grad mismatch")
