"""Batched ternary SwiGLU micro-network with CPU and GPU train steps."""

from .buffer import ChainData
from .grads import ModelGrads
from .ternary import TernaryWeight, quantize_weights, silu, silu_derivative


struct BatchMicroNet(Copyable, Movable):
    """Student network with batched forward/backward."""

    var input_dim: Int
    var hidden_dim: Int
    var gate_shadow: List[Float32]
    var up_shadow: List[Float32]
    var head_shadow: List[Float32]
    var alpha: Float32
    var gate_ternary: List[TernaryWeight]
    var up_ternary: List[TernaryWeight]
    var head_ternary: List[TernaryWeight]
    var _gate_buf: List[Float32]
    var _up_buf: List[Float32]
    var use_ternary: Bool

    def __init__(
        out self,
        input_dim: Int,
        hidden_dim: Int,
        alpha: Float32 = 1.0,
        use_ternary: Bool = True,
    ) raises:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        var gate_size = hidden_dim * input_dim
        var head_size = hidden_dim

        self.gate_shadow = List[Float32](capacity=gate_size)
        self.up_shadow = List[Float32](capacity=gate_size)
        self.head_shadow = List[Float32](capacity=head_size)
        self.gate_ternary = List[TernaryWeight](capacity=gate_size)
        self.up_ternary = List[TernaryWeight](capacity=gate_size)
        self.head_ternary = List[TernaryWeight](capacity=head_size)

        for _ in range(gate_size):
            self.gate_shadow.append(0.0)
            self.up_shadow.append(0.0)
            self.gate_ternary.append(0)
            self.up_ternary.append(0)

        for _ in range(head_size):
            self.head_shadow.append(0.0)
            self.head_ternary.append(0)

        self.alpha = alpha
        self.use_ternary = use_ternary
        self._gate_buf = List[Float32](capacity=hidden_dim)
        self._up_buf = List[Float32](capacity=hidden_dim)
        for _ in range(hidden_dim):
            self._gate_buf.append(0.0)
            self._up_buf.append(0.0)

    def sync_ternary(mut self) -> None:
        quantize_weights(self.gate_shadow, self.gate_ternary)
        quantize_weights(self.up_shadow, self.up_ternary)
        quantize_weights(self.head_shadow, self.head_ternary)

    def param_count(self) -> Int:
        return 2 * self.hidden_dim * self.input_dim + self.hidden_dim + 1

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
            var x_base = data.x_offset(sample)
            var target = data.y_at(sample)

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

            var y_tern: Float32 = 0.0
            for j in range(self.hidden_dim):
                var h = silu(self._gate_buf[j]) * self._up_buf[j]
                if self.use_ternary:
                    var w = self.head_ternary[j]
                    if w != 0:
                        y_tern += Float32(w) * h
                else:
                    y_tern += self.head_shadow[j] * h

            var pred = self.alpha * y_tern
            var err = pred - target
            loss += err * err * inv_batch

            var dL_dout = 2.0 * err * inv_batch
            grads.alpha += dL_dout * y_tern

            var dL_dy_tern = dL_dout * self.alpha
            for j in range(self.hidden_dim):
                var gate_j = self._gate_buf[j]
                var up_j = self._up_buf[j]
                var silu_gate = silu(gate_j)
                var h_j = silu_gate * up_j

                grads.head[j] += dL_dy_tern * h_j

                var head_w = (
                    Float32(self.head_ternary[j])
                    if self.use_ternary
                    else self.head_shadow[j]
                )
                var dL_dh = dL_dy_tern * head_w
                var dL_dgate = dL_dh * up_j * silu_derivative(gate_j)
                var dL_dup = dL_dh * silu_gate

                var row_base = j * self.input_dim
                for i in range(self.input_dim):
                    var x_i = data.x_data[x_base + i]
                    grads.gate[row_base + i] += dL_dgate * x_i
                    grads.up[row_base + i] += dL_dup * x_i

        return loss

    def gate_size(self) -> Int:
        return self.hidden_dim * self.input_dim

    def eval_mse(mut self, data: ChainData) raises -> Float32:
        """Mean squared error on data (inference only, no gradients)."""
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

        var y_tern: Float32 = 0.0
        for j in range(self.hidden_dim):
            var h = silu(self._gate_buf[j]) * self._up_buf[j]
            if self.use_ternary:
                var w = self.head_ternary[j]
                if w != 0:
                    y_tern += Float32(w) * h
            else:
                y_tern += self.head_shadow[j] * h

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


def _float_matvec_rowmajor(
    weights: List[Float32],
    input: List[Float32],
    input_base: Int,
    in_dim: Int,
    out_dim: Int,
    mut output: List[Float32],
) -> None:
    """y[j] = sum_i W[j,i] * x[i] with FP32 shadow weights."""
    for j in range(out_dim):
        var acc: Float32 = 0.0
        var row_base = j * in_dim
        for i in range(in_dim):
            acc += weights[row_base + i] * input[input_base + i]
        output[j] = acc
