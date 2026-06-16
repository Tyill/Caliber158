"""Batched ternary SwiGLU micro-network with CPU and GPU train steps."""

from .arch import ArchKind
from .buffer import ChainData
from .grads import ModelGrads
from .ternary import TernaryWeight, quantize_weights, silu, silu_derivative


struct BatchMicroNet(Copyable, Movable):
    """Student network with batched forward/backward (v0 or v1)."""

    var arch: ArchKind
    var input_dim: Int
    var hidden_dim: Int
    var gate_shadow: List[Float32]
    var up_shadow: List[Float32]
    var head_shadow: List[Float32]
    var gate2_shadow: List[Float32]
    var up2_shadow: List[Float32]
    var alpha: Float32
    var block2_residual_scale: Float32
    var gate_ternary: List[TernaryWeight]
    var up_ternary: List[TernaryWeight]
    var head_ternary: List[TernaryWeight]
    var gate2_ternary: List[TernaryWeight]
    var up2_ternary: List[TernaryWeight]
    var _gate_buf: List[Float32]
    var _up_buf: List[Float32]
    var _gate2_buf: List[Float32]
    var _up2_buf: List[Float32]
    var _h1_buf: List[Float32]

    def __init__(
        out self,
        input_dim: Int,
        hidden_dim: Int,
        arch: ArchKind = ArchKind.v0(),
        alpha: Float32 = 1.0,
        block2_residual_scale: Float32 = 1.0,
    ) raises:
        self.arch = arch.copy()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.block2_residual_scale = block2_residual_scale

        var gate_size = hidden_dim * input_dim
        var head_size = hidden_dim
        var block2_size = 0
        if arch.is_v1():
            block2_size = hidden_dim * hidden_dim

        self.gate_shadow = List[Float32](capacity=gate_size)
        self.up_shadow = List[Float32](capacity=gate_size)
        self.head_shadow = List[Float32](capacity=head_size)
        self.gate2_shadow = List[Float32](capacity=block2_size)
        self.up2_shadow = List[Float32](capacity=block2_size)
        self.gate_ternary = List[TernaryWeight](capacity=gate_size)
        self.up_ternary = List[TernaryWeight](capacity=gate_size)
        self.head_ternary = List[TernaryWeight](capacity=head_size)
        self.gate2_ternary = List[TernaryWeight](capacity=block2_size)
        self.up2_ternary = List[TernaryWeight](capacity=block2_size)

        for _ in range(gate_size):
            self.gate_shadow.append(0.0)
            self.up_shadow.append(0.0)
            self.gate_ternary.append(0)
            self.up_ternary.append(0)

        for _ in range(head_size):
            self.head_shadow.append(0.0)
            self.head_ternary.append(0)

        for _ in range(block2_size):
            self.gate2_shadow.append(0.0)
            self.up2_shadow.append(0.0)
            self.gate2_ternary.append(0)
            self.up2_ternary.append(0)

        self.alpha = alpha
        self._gate_buf = List[Float32](capacity=hidden_dim)
        self._up_buf = List[Float32](capacity=hidden_dim)
        self._gate2_buf = List[Float32](capacity=hidden_dim)
        self._up2_buf = List[Float32](capacity=hidden_dim)
        self._h1_buf = List[Float32](capacity=hidden_dim)
        for _ in range(hidden_dim):
            self._gate_buf.append(0.0)
            self._up_buf.append(0.0)
            self._gate2_buf.append(0.0)
            self._up2_buf.append(0.0)
            self._h1_buf.append(0.0)

    def sync_ternary(mut self) -> None:
        quantize_weights(self.gate_shadow, self.gate_ternary)
        quantize_weights(self.up_shadow, self.up_ternary)
        quantize_weights(self.head_shadow, self.head_ternary)
        if self.arch.is_v1():
            quantize_weights(self.gate2_shadow, self.gate2_ternary)
            quantize_weights(self.up2_shadow, self.up2_ternary)

    def param_count(self) -> Int:
        var n = 2 * self.hidden_dim * self.input_dim + self.hidden_dim + 1
        if self.arch.is_v1():
            n += 2 * self.hidden_dim * self.hidden_dim
        return n

    def gate_size(self) -> Int:
        return self.hidden_dim * self.input_dim

    def block2_size(self) -> Int:
        if self.arch.is_v1():
            return self.hidden_dim * self.hidden_dim
        return 0

    def train_step_cpu(
        mut self,
        data: ChainData,
        start: Int,
        batch_size: Int,
        mut grads: ModelGrads,
    ) raises -> Float32:
        """Batched forward + MSE backward with STE (CPU)."""
        var count = data.batch_size(start, batch_size)
        if count == 0:
            return 0.0

        self.sync_ternary()
        grads.zero()

        var inv_batch = 1.0 / Float32(count)
        var loss: Float32 = 0.0

        for sample in range(start, start + count):
            var x_base = data.x_offset(sample)
            var target = data.y_at(sample)

            if self.arch.is_v1():
                loss += self._train_sample_v1(data, x_base, target, inv_batch, grads)
            else:
                loss += self._train_sample_v0(data, x_base, target, inv_batch, grads)

        return loss

    def _train_sample_v0(
        mut self,
        data: ChainData,
        x_base: Int,
        target: Float32,
        inv_batch: Float32,
        mut grads: ModelGrads,
    ) raises -> Float32:
        _ternary_matvec_rowmajor(
            self.gate_ternary,
            data.x_data,
            x_base,
            self.input_dim,
            self.hidden_dim,
            self._gate_buf,
        )
        _ternary_matvec_rowmajor(
            self.up_ternary,
            data.x_data,
            x_base,
            self.input_dim,
            self.hidden_dim,
            self._up_buf,
        )

        var y_tern: Float32 = 0.0
        for j in range(self.hidden_dim):
            var h = silu(self._gate_buf[j]) * self._up_buf[j]
            var w = self.head_ternary[j]
            if w != 0:
                y_tern += Float32(w) * h

        var pred = self.alpha * y_tern
        var err = pred - target
        var loss = err * err * inv_batch

        var dL_dout = 2.0 * err * inv_batch
        grads.alpha += dL_dout * y_tern

        var dL_dy_tern = dL_dout * self.alpha
        for j in range(self.hidden_dim):
            var gate_j = self._gate_buf[j]
            var up_j = self._up_buf[j]
            var silu_gate = silu(gate_j)
            var h_j = silu_gate * up_j

            grads.head[j] += dL_dy_tern * h_j

            var dL_dh = dL_dy_tern * Float32(self.head_ternary[j])
            var dL_dgate = dL_dh * up_j * silu_derivative(gate_j)
            var dL_dup = dL_dh * silu_gate

            var row_base = j * self.input_dim
            for i in range(self.input_dim):
                var x_i = data.x_data[x_base + i]
                grads.gate[row_base + i] += dL_dgate * x_i
                grads.up[row_base + i] += dL_dup * x_i

        return loss

    def _train_sample_v1(
        mut self,
        data: ChainData,
        x_base: Int,
        target: Float32,
        inv_batch: Float32,
        mut grads: ModelGrads,
    ) raises -> Float32:
        # Block 1: x -> h1
        _ternary_matvec_rowmajor(
            self.gate_ternary,
            data.x_data,
            x_base,
            self.input_dim,
            self.hidden_dim,
            self._gate_buf,
        )
        _ternary_matvec_rowmajor(
            self.up_ternary,
            data.x_data,
            x_base,
            self.input_dim,
            self.hidden_dim,
            self._up_buf,
        )
        for j in range(self.hidden_dim):
            self._h1_buf[j] = silu(self._gate_buf[j]) * self._up_buf[j]

        # Block 2: h1 -> h2
        _ternary_matvec_hidden(
            self.gate2_ternary,
            self._h1_buf,
            self.hidden_dim,
            self._gate2_buf,
        )
        _ternary_matvec_hidden(
            self.up2_ternary,
            self._h1_buf,
            self.hidden_dim,
            self._up2_buf,
        )

        var y_tern: Float32 = 0.0
        for j in range(self.hidden_dim):
            var h2_j = silu(self._gate2_buf[j]) * self._up2_buf[j]
            var h_j = self._h1_buf[j] + self.block2_residual_scale * h2_j
            var w = self.head_ternary[j]
            if w != 0:
                y_tern += Float32(w) * h_j

        var pred = self.alpha * y_tern
        var err = pred - target
        var loss = err * err * inv_batch

        var dL_dout = 2.0 * err * inv_batch
        grads.alpha += dL_dout * y_tern
        var dL_dy_tern = dL_dout * self.alpha

        var dL_dh1 = List[Float32](capacity=self.hidden_dim)
        for _ in range(self.hidden_dim):
            dL_dh1.append(0.0)

        for j in range(self.hidden_dim):
            var h2_j = silu(self._gate2_buf[j]) * self._up2_buf[j]
            var h_j = self._h1_buf[j] + self.block2_residual_scale * h2_j

            grads.head[j] += dL_dy_tern * h_j

            var dL_dh_j = dL_dy_tern * Float32(self.head_ternary[j])
            dL_dh1[j] += dL_dh_j

            var dL_dh2_j = dL_dh_j * self.block2_residual_scale
            var g2_j = self._gate2_buf[j]
            var u2_j = self._up2_buf[j]
            var silu_g2 = silu(g2_j)
            var dL_dg2 = dL_dh2_j * u2_j * silu_derivative(g2_j)
            var dL_du2 = dL_dh2_j * silu_g2
            var row2 = j * self.hidden_dim
            for i in range(self.hidden_dim):
                grads.gate2[row2 + i] += dL_dg2 * self._h1_buf[i]
                grads.up2[row2 + i] += dL_du2 * self._h1_buf[i]
                dL_dh1[i] += dL_dg2 * Float32(self.gate2_ternary[row2 + i])
                dL_dh1[i] += dL_du2 * Float32(self.up2_ternary[row2 + i])

        for j in range(self.hidden_dim):
            var dL_dh1_j = dL_dh1[j]
            var gate_j = self._gate_buf[j]
            var up_j = self._up_buf[j]
            var silu_gate = silu(gate_j)
            var dL_dgate = dL_dh1_j * up_j * silu_derivative(gate_j)
            var dL_dup = dL_dh1_j * silu_gate
            var row_base = j * self.input_dim
            for i in range(self.input_dim):
                var x_i = data.x_data[x_base + i]
                grads.gate[row_base + i] += dL_dgate * x_i
                grads.up[row_base + i] += dL_dup * x_i

        return loss

    def eval_mse(mut self, data: ChainData) raises -> Float32:
        """Mean squared error on ``data`` (inference only, no gradients)."""
        if data.n_samples == 0:
            return 0.0

        self.sync_ternary()
        var loss: Float32 = 0.0
        var inv_n = 1.0 / Float32(data.n_samples)

        for sample in range(data.n_samples):
            var pred = self._predict_at(data, sample)
            var err = pred - data.y_at(sample)
            loss += err * err * inv_n

        return loss

    def _predict_at(mut self, data: ChainData, sample: Int) -> Float32:
        var x_base = data.x_offset(sample)

        if self.arch.is_v1():
            return self._predict_v1(data, x_base)

        _ternary_matvec_rowmajor(
            self.gate_ternary,
            data.x_data,
            x_base,
            self.input_dim,
            self.hidden_dim,
            self._gate_buf,
        )
        _ternary_matvec_rowmajor(
            self.up_ternary,
            data.x_data,
            x_base,
            self.input_dim,
            self.hidden_dim,
            self._up_buf,
        )

        var y_tern: Float32 = 0.0
        for j in range(self.hidden_dim):
            var h = silu(self._gate_buf[j]) * self._up_buf[j]
            var w = self.head_ternary[j]
            if w != 0:
                y_tern += Float32(w) * h

        return self.alpha * y_tern

    def _predict_v1(mut self, data: ChainData, x_base: Int) -> Float32:
        _ternary_matvec_rowmajor(
            self.gate_ternary,
            data.x_data,
            x_base,
            self.input_dim,
            self.hidden_dim,
            self._gate_buf,
        )
        _ternary_matvec_rowmajor(
            self.up_ternary,
            data.x_data,
            x_base,
            self.input_dim,
            self.hidden_dim,
            self._up_buf,
        )
        for j in range(self.hidden_dim):
            self._h1_buf[j] = silu(self._gate_buf[j]) * self._up_buf[j]

        _ternary_matvec_hidden(
            self.gate2_ternary,
            self._h1_buf,
            self.hidden_dim,
            self._gate2_buf,
        )
        _ternary_matvec_hidden(
            self.up2_ternary,
            self._h1_buf,
            self.hidden_dim,
            self._up2_buf,
        )

        var y_tern: Float32 = 0.0
        for j in range(self.hidden_dim):
            var h = self._h1_buf[j] + self.block2_residual_scale * silu(self._gate2_buf[j]) * self._up2_buf[j]
            var w = self.head_ternary[j]
            if w != 0:
                y_tern += Float32(w) * h

        return self.alpha * y_tern


def _ceildiv(n: Int, d: Int) -> Int:
    return (n + d - 1) // d


def _ternary_matvec_rowmajor(
    weights: List[TernaryWeight],
    input: List[Float32],
    input_base: Int,
    in_dim: Int,
    out_dim: Int,
    mut output: List[Float32],
) -> None:
    """y[j] = sum_i W[j,i] * x[i] using a row slice of input."""
    for j in range(out_dim):
        var acc: Float32 = 0.0
        var row_base = j * in_dim
        for i in range(in_dim):
            var w = weights[row_base + i]
            if w != 0:
                acc += Float32(w) * input[input_base + i]
        output[j] = acc


def _ternary_matvec_hidden(
    weights: List[TernaryWeight],
    input: List[Float32],
    dim: Int,
    mut output: List[Float32],
) -> None:
    """Square ternary matvec: y = W @ x with W [dim x dim] row-major."""
    for j in range(dim):
        var acc: Float32 = 0.0
        var row_base = j * dim
        for i in range(dim):
            var w = weights[row_base + i]
            if w != 0:
                acc += Float32(w) * input[i]
        output[j] = acc
