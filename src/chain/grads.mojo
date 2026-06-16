"""Gradient buffers for MicroNet (STE accumulates into shadow-weight grads)."""

from .arch import ArchKind, block2_weight_count


@fieldwise_init
struct ModelGrads(Copyable, Movable):
    var gate: List[Float32]
    var up: List[Float32]
    var gate2: List[Float32]
    var up2: List[Float32]
    var head: List[Float32]
    var alpha: Float32

    @staticmethod
    def zeros(
        gate_len: Int,
        up_len: Int,
        head_len: Int,
        gate2_len: Int = 0,
        up2_len: Int = 0,
    ) -> ModelGrads:
        return ModelGrads(
            _zeros_list(gate_len),
            _zeros_list(up_len),
            _zeros_list(gate2_len),
            _zeros_list(up2_len),
            _zeros_list(head_len),
            0.0,
        )

    @staticmethod
    def zeros_for_model(
        input_dim: Int,
        hidden_dim: Int,
        arch: ArchKind,
    ) -> ModelGrads:
        var gate_len = hidden_dim * input_dim
        var b2 = block2_weight_count(arch, hidden_dim)
        return ModelGrads.zeros(gate_len, gate_len, hidden_dim, b2, b2)

    def zero(mut self) -> None:
        _zero_list(self.gate)
        _zero_list(self.up)
        _zero_list(self.gate2)
        _zero_list(self.up2)
        _zero_list(self.head)
        self.alpha = 0.0


def _zeros_list(n: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for _ in range(n):
        out.append(0.0)
    return out^


def _zero_list(mut xs: List[Float32]) -> None:
    for i in range(len(xs)):
        xs[i] = 0.0
