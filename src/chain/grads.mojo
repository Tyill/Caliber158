"""Gradient buffers for BatchMicroNet (STE accumulates into shadow-weight grads)."""


@fieldwise_init
struct ModelGrads(Copyable, Movable):
    var gate: List[Float32]
    var up: List[Float32]
    var head: List[Float32]
    var gate2: List[Float32]
    var up2: List[Float32]
    var alpha: Float32

    @staticmethod
    def zeros(
        gate_len: Int,
        up_len: Int,
        head_len: Int,
        gate2_len: Int = 0,
        up2_len: Int = 0,
    ) -> ModelGrads:
        var gate = List[Float32](capacity=gate_len)
        var up = List[Float32](capacity=up_len)
        var head = List[Float32](capacity=head_len)
        var gate2 = List[Float32](capacity=gate2_len)
        var up2 = List[Float32](capacity=up2_len)

        for _ in range(gate_len):
            gate.append(0.0)
        for _ in range(up_len):
            up.append(0.0)
        for _ in range(head_len):
            head.append(0.0)
        for _ in range(gate2_len):
            gate2.append(0.0)
        for _ in range(up2_len):
            up2.append(0.0)

        return ModelGrads(gate^, up^, head^, gate2^, up2^, 0.0)

    def zero(mut self) -> None:
        for i in range(len(self.gate)):
            self.gate[i] = 0.0
        for i in range(len(self.up)):
            self.up[i] = 0.0
        for i in range(len(self.head)):
            self.head[i] = 0.0
        for i in range(len(self.gate2)):
            self.gate2[i] = 0.0
        for i in range(len(self.up2)):
            self.up2[i] = 0.0
        self.alpha = 0.0
