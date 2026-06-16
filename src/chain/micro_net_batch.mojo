"""Batched ternary SwiGLU micro-network with CPU and GPU train steps."""

from .arch import ArchKind, block2_weight_count, param_count
from .buffer import ChainData
from .grads import ModelGrads
from .ternary import TernaryWeight, quantize_weights, silu, silu_derivative


struct BatchMicroNet(Copyable, Movable):
    """Student network with batched forward/backward."""

    var arch: ArchKind
    var input_dim: Int
    var hidden_dim: Int
    var gate_shadow: List[Float32]
    var up_shadow: List[Float32]
    var gate2_shadow: List[Float32]
    var up2_shadow: List[Float32]
    var head_shadow: List[Float32]
    var alpha: Float32
    var gate_ternary: List[TernaryWeight]
    var up_ternary: List[TernaryWeight]
    var gate2_ternary: List[TernaryWeight]
    var up2_ternary: List[TernaryWeight]
    var head_ternary: List[TernaryWeight]
    var _gate_buf: List[Float32]
    var _up_buf: List[Float32]
    var _h0_buf: List[Float32]
    var _gate2_buf: List[Float32]
    var _up2_buf: List[Float32]
    var _h1_buf: List[Float32]
    var use_ternary: Bool

    def __init__(
        out self,
        input_dim: Int,
        hidden_dim: Int,
        alpha: Float32 = 1.0,
        use_ternary: Bool = True,
        arch: ArchKind = ArchKind.v0(),
    ) raises:
        self.arch = arch
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        var gate_size = hidden_dim * input_dim
        var head_size = hidden_dim
        var block2_size = block2_weight_count(arch, hidden_dim)

        self.gate_shadow = List[Float32](capacity=gate_size)
        self.up_shadow = List[Float32](capacity=gate_size)
        self.gate2_shadow = List[Float32](capacity=block2_size)
        self.up2_shadow = List[Float32](capacity=block2_size)
        self.head_shadow = List[Float32](capacity=head_size)
        self.gate_ternary = List[TernaryWeight](capacity=gate_size)
        self.up_ternary = List[TernaryWeight](capacity=gate_size)
        self.gate2_ternary = List[TernaryWeight](capacity=block2_size)
        self.up2_ternary = List[TernaryWeight](capacity=block2_size)
        self.head_ternary = List[TernaryWeight](capacity=head_size)

        for _ in range(gate_size):
            self.gate_shadow.append(0.0)
            self.up_shadow.append(0.0)
            self.gate_ternary.append(0)
            self.up_ternary.append(0)

        for _ in range(block2_size):
            self.gate2_shadow.append(0.0)
            self.up2_shadow.append(0.0)
            self.gate2_ternary.append(0)
            self.up2_ternary.append(0)

        for _ in range(head_size):
            self.head_shadow.append(0.0)
            self.head_ternary.append(0)

        self.alpha = alpha
        self.use_ternary = use_ternary
        self._gate_buf = List[Float32](capacity=hidden_dim)
        self._up_buf = List[Float32](capacity=hidden_dim)
        self._h0_buf = List[Float32](capacity=hidden_dim)
        self._gate2_buf = List[Float32](capacity=hidden_dim)
        self._up2_buf = List[Float32](capacity=hidden_dim)
        self._h1_buf = List[Float32](capacity=hidden_dim)
        for _ in range(hidden_dim):
            self._gate_buf.append(0.0)
            self._up_buf.append(0.0)
            self._h0_buf.append(0.0)
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
        return param_count(self.arch, self.input_dim, self.hidden_dim)

    def gate_size(self) -> Int:
        return self.hidden_dim * self.input_dim

    def block2_size(self) -> Int:
        return block2_weight_count(self.arch, self.hidden_dim)

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

        if self.use_ternary:
            self.sync_ternary()
        grads.zero()

        var inv_batch = 1.0 / Float32(count)
        var loss: Float32 = 0.0

        for sample in range(start, start + count):
            if self.arch.is_v1():
                loss += self._train_sample_v1(data, sample, inv_batch, grads)
            else:
                loss += self._train_sample_v0(data, sample, inv_batch, grads)

        return loss

    def _train_sample_v0(
        mut self,
        data: ChainData,
        sample: Int,
        inv_batch: Float32,
        mut grads: ModelGrads,
    ) raises -> Float32:
        var x_base = data.x_offset(sample)
        var target = data.y_at(sample)

        self._block1_forward(data, x_base)

        var y_tern = self._dot_head(self._h0_buf)
        var pred = self.alpha * y_tern
        var err = pred - target
        var loss = err * err * inv_batch

        var dL_dout = 2.0 * err * inv_batch
        grads.alpha += dL_dout * y_tern
        var dL_dy_tern = dL_dout * self.alpha

        self._head_backward(dL_dy_tern, self._h0_buf, grads)
        self._block1_backward(dL_dy_tern, data, x_base, grads)

        return loss

    def _train_sample_v1(
        mut self,
        data: ChainData,
        sample: Int,
        inv_batch: Float32,
        mut grads: ModelGrads,
    ) raises -> Float32:
        var x_base = data.x_offset(sample)
        var target = data.y_at(sample)

        self._block1_forward(data, x_base)
        self._block2_forward()

        for j in range(self.hidden_dim):
            self._h1_buf[j] = self._h0_buf[j] + self._h1_buf[j]

        var y_tern = self._dot_head(self._h1_buf)
        var pred = self.alpha * y_tern
        var err = pred - target
        var loss = err * err * inv_batch

        var dL_dout = 2.0 * err * inv_batch
        grads.alpha += dL_dout * y_tern
        var dL_dy_tern = dL_dout * self.alpha

        var dL_dh1 = List[Float32](capacity=self.hidden_dim)
        var dL_dh0 = List[Float32](capacity=self.hidden_dim)
        for _ in range(self.hidden_dim):
            dL_dh1.append(0.0)
            dL_dh0.append(0.0)

        self._head_backward_into(dL_dy_tern, self._h1_buf, grads, dL_dh1)

        self._block2_backward(dL_dh1, grads, dL_dh0)

        for j in range(self.hidden_dim):
            dL_dh0[j] += dL_dh1[j]

        self._block1_backward_from_dL_dh(dL_dh0, data, x_base, grads)

        return loss

    def eval_mse(mut self, data: ChainData) raises -> Float32:
        if data.n_samples == 0:
            return 0.0

        if self.use_ternary:
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
        self._block1_forward(data, x_base)

        if self.arch.is_v1():
            self._block2_forward()
            for j in range(self.hidden_dim):
                self._h1_buf[j] = self._h0_buf[j] + self._h1_buf[j]
            return self.alpha * self._dot_head(self._h1_buf)

        return self.alpha * self._dot_head(self._h0_buf)

    def _block1_forward(mut self, data: ChainData, x_base: Int) -> None:
        if self.use_ternary:
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
        else:
            _float_matvec_rowmajor(
                self.gate_shadow,
                data.x_data,
                x_base,
                self.input_dim,
                self.hidden_dim,
                self._gate_buf,
            )
            _float_matvec_rowmajor(
                self.up_shadow,
                data.x_data,
                x_base,
                self.input_dim,
                self.hidden_dim,
                self._up_buf,
            )
        for j in range(self.hidden_dim):
            self._h0_buf[j] = silu(self._gate_buf[j]) * self._up_buf[j]

    def _block2_forward(mut self) -> None:
        if self.use_ternary:
            _ternary_matvec_from_vec(
                self.gate2_ternary, self._h0_buf, self.hidden_dim, self._gate2_buf
            )
            _ternary_matvec_from_vec(
                self.up2_ternary, self._h0_buf, self.hidden_dim, self._up2_buf
            )
        else:
            _float_matvec_from_vec(
                self.gate2_shadow, self._h0_buf, self.hidden_dim, self._gate2_buf
            )
            _float_matvec_from_vec(
                self.up2_shadow, self._h0_buf, self.hidden_dim, self._up2_buf
            )
        for j in range(self.hidden_dim):
            self._h1_buf[j] = silu(self._gate2_buf[j]) * self._up2_buf[j]

    def _head_weight(self, j: Int) -> Float32:
        if self.use_ternary:
            return Float32(self.head_ternary[j])
        return self.head_shadow[j]

    def _dot_head(self, hidden: List[Float32]) -> Float32:
        var y_tern: Float32 = 0.0
        for j in range(self.hidden_dim):
            var h = hidden[j]
            if self.use_ternary:
                var w = self.head_ternary[j]
                if w != 0:
                    y_tern += Float32(w) * h
            else:
                y_tern += self.head_shadow[j] * h
        return y_tern

    def _head_backward(
        self,
        dL_dy_tern: Float32,
        hidden: List[Float32],
        mut grads: ModelGrads,
    ) -> None:
        for j in range(self.hidden_dim):
            grads.head[j] += dL_dy_tern * hidden[j]

    def _head_backward_into(
        self,
        dL_dy_tern: Float32,
        hidden: List[Float32],
        mut grads: ModelGrads,
        mut dL_dhidden: List[Float32],
    ) -> None:
        for j in range(self.hidden_dim):
            grads.head[j] += dL_dy_tern * hidden[j]
            dL_dhidden[j] = dL_dy_tern * self._head_weight(j)

    def _block1_backward(
        self,
        dL_dy_tern: Float32,
        data: ChainData,
        x_base: Int,
        mut grads: ModelGrads,
    ) -> None:
        for j in range(self.hidden_dim):
            var dL_dh = dL_dy_tern * self._head_weight(j)
            self._swiglu_weight_grads_into(
                dL_dh,
                self._gate_buf[j],
                self._up_buf[j],
                self._h0_buf[j],
                data.x_data,
                x_base,
                j * self.input_dim,
                grads.gate,
                grads.up,
            )

    def _block1_backward_from_dL_dh(
        self,
        dL_dh0: List[Float32],
        data: ChainData,
        x_base: Int,
        mut grads: ModelGrads,
    ) -> None:
        for j in range(self.hidden_dim):
            self._swiglu_weight_grads_into(
                dL_dh0[j],
                self._gate_buf[j],
                self._up_buf[j],
                self._h0_buf[j],
                data.x_data,
                x_base,
                j * self.input_dim,
                grads.gate,
                grads.up,
            )

    def _block2_backward(
        self,
        dL_dh1: List[Float32],
        mut grads: ModelGrads,
        mut dL_dh0: List[Float32],
    ) -> None:
        var h_dim = self.hidden_dim
        for j in range(h_dim):
            var dL_dh2 = dL_dh1[j]
            var gate_j = self._gate2_buf[j]
            var up_j = self._up2_buf[j]
            var silu_gate = silu(gate_j)
            var dL_dgate = dL_dh2 * up_j * silu_derivative(gate_j)
            var dL_dup = dL_dh2 * silu_gate
            var row_base = j * h_dim

            for i in range(h_dim):
                var w_gate = self._block2_weight(self.gate2_shadow, self.gate2_ternary, row_base + i)
                var w_up = self._block2_weight(self.up2_shadow, self.up2_ternary, row_base + i)
                dL_dh0[i] += dL_dgate * w_gate + dL_dup * w_up
                grads.gate2[row_base + i] += dL_dgate * self._h0_buf[i]
                grads.up2[row_base + i] += dL_dup * self._h0_buf[i]

    def _block2_weight(
        self,
        shadow: List[Float32],
        ternary: List[TernaryWeight],
        idx: Int,
    ) -> Float32:
        if self.use_ternary:
            return Float32(ternary[idx])
        return shadow[idx]

    def _swiglu_weight_grads_into(
        self,
        dL_dh: Float32,
        gate_j: Float32,
        up_j: Float32,
        _h_j: Float32,
        x_data: List[Float32],
        x_base: Int,
        grad_row_base: Int,
        mut grad_gate: List[Float32],
        mut grad_up: List[Float32],
    ) -> None:
        var silu_gate = silu(gate_j)
        var dL_dgate = dL_dh * up_j * silu_derivative(gate_j)
        var dL_dup = dL_dh * silu_gate
        for i in range(self.input_dim):
            var x_i = x_data[x_base + i]
            grad_gate[grad_row_base + i] += dL_dgate * x_i
            grad_up[grad_row_base + i] += dL_dup * x_i


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
    for j in range(out_dim):
        var acc: Float32 = 0.0
        var row_base = j * in_dim
        for i in range(in_dim):
            var w = weights[row_base + i]
            if w != 0:
                acc += Float32(w) * input[input_base + i]
        output[j] = acc


def _float_matvec_rowmajor(
    weights: List[Float32],
    input: List[Float32],
    input_base: Int,
    in_dim: Int,
    out_dim: Int,
    mut output: List[Float32],
) -> None:
    for j in range(out_dim):
        var acc: Float32 = 0.0
        var row_base = j * in_dim
        for i in range(in_dim):
            acc += weights[row_base + i] * input[input_base + i]
        output[j] = acc


def _ternary_matvec_from_vec(
    weights: List[TernaryWeight],
    input: List[Float32],
    in_dim: Int,
    mut output: List[Float32],
) -> None:
    for j in range(in_dim):
        var acc: Float32 = 0.0
        var row_base = j * in_dim
        for i in range(in_dim):
            var w = weights[row_base + i]
            if w != 0:
                acc += Float32(w) * input[i]
        output[j] = acc


def _float_matvec_from_vec(
    weights: List[Float32],
    input: List[Float32],
    in_dim: Int,
    mut output: List[Float32],
) -> None:
    for j in range(in_dim):
        var acc: Float32 = 0.0
        var row_base = j * in_dim
        for i in range(in_dim):
            acc += weights[row_base + i] * input[i]
        output[j] = acc
