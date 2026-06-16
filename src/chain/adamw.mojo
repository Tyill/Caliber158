"""AdamW optimizer for BatchMicroNet shadow weights and α."""

from std.math import sqrt

from .grads import ModelGrads
from .micro_net_batch import BatchMicroNet


@fieldwise_init
struct AdamWConfig(Copyable, Movable):
    var learning_rate: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32


@fieldwise_init
struct AdamWState(Copyable, Movable):
    var gate_m: List[Float32]
    var gate_v: List[Float32]
    var up_m: List[Float32]
    var up_v: List[Float32]
    var head_m: List[Float32]
    var head_v: List[Float32]
    var gate2_m: List[Float32]
    var gate2_v: List[Float32]
    var up2_m: List[Float32]
    var up2_v: List[Float32]
    var alpha_m: Float32
    var alpha_v: Float32
    var timestep: Int

    @staticmethod
    def from_model(model: BatchMicroNet) -> AdamWState:
        var gate_m = List[Float32](capacity=len(model.gate_shadow))
        var gate_v = List[Float32](capacity=len(model.gate_shadow))
        var up_m = List[Float32](capacity=len(model.up_shadow))
        var up_v = List[Float32](capacity=len(model.up_shadow))
        var head_m = List[Float32](capacity=len(model.head_shadow))
        var head_v = List[Float32](capacity=len(model.head_shadow))
        var gate2_m = List[Float32](capacity=len(model.gate2_shadow))
        var gate2_v = List[Float32](capacity=len(model.gate2_shadow))
        var up2_m = List[Float32](capacity=len(model.up2_shadow))
        var up2_v = List[Float32](capacity=len(model.up2_shadow))

        for _ in range(len(model.gate_shadow)):
            gate_m.append(0.0)
            gate_v.append(0.0)
        for _ in range(len(model.up_shadow)):
            up_m.append(0.0)
            up_v.append(0.0)
        for _ in range(len(model.head_shadow)):
            head_m.append(0.0)
            head_v.append(0.0)
        for _ in range(len(model.gate2_shadow)):
            gate2_m.append(0.0)
            gate2_v.append(0.0)
        for _ in range(len(model.up2_shadow)):
            up2_m.append(0.0)
            up2_v.append(0.0)

        return AdamWState(
            gate_m^,
            gate_v^,
            up_m^,
            up_v^,
            head_m^,
            head_v^,
            gate2_m^,
            gate2_v^,
            up2_m^,
            up2_v^,
            0.0,
            0.0,
            0,
        )

    def apply(
        mut self,
        mut model: BatchMicroNet,
        grads: ModelGrads,
        config: AdamWConfig,
    ) -> None:
        self.timestep += 1
        var bias_corr1 = _one_minus_pow(config.beta1, self.timestep)
        var bias_corr2 = _one_minus_pow(config.beta2, self.timestep)

        _adamw_update_list(
            model.gate_shadow,
            grads.gate,
            self.gate_m,
            self.gate_v,
            bias_corr1,
            bias_corr2,
            config,
        )
        _adamw_update_list(
            model.up_shadow,
            grads.up,
            self.up_m,
            self.up_v,
            bias_corr1,
            bias_corr2,
            config,
        )
        _adamw_update_list(
            model.head_shadow,
            grads.head,
            self.head_m,
            self.head_v,
            bias_corr1,
            bias_corr2,
            config,
        )
        if model.arch.is_v1():
            _adamw_update_list(
                model.gate2_shadow,
                grads.gate2,
                self.gate2_m,
                self.gate2_v,
                bias_corr1,
                bias_corr2,
                config,
            )
            _adamw_update_list(
                model.up2_shadow,
                grads.up2,
                self.up2_m,
                self.up2_v,
                bias_corr1,
                bias_corr2,
                config,
            )
        _adamw_update_scalar(
            model.alpha,
            grads.alpha,
            self.alpha_m,
            self.alpha_v,
            bias_corr1,
            bias_corr2,
            config,
        )


def one_minus_pow(base: Float32, exp: Int) -> Float32:
    var p: Float32 = 1.0
    for _ in range(exp):
        p *= base
    return 1.0 - p


def _one_minus_pow(base: Float32, exp: Int) -> Float32:
    return one_minus_pow(base, exp)


def _adamw_update_list(
    mut params: List[Float32],
    grads: List[Float32],
    mut m: List[Float32],
    mut v: List[Float32],
    bias_corr1: Float32,
    bias_corr2: Float32,
    config: AdamWConfig,
) -> None:
    for i in range(len(params)):
        m[i] = config.beta1 * m[i] + (1.0 - config.beta1) * grads[i]
        v[i] = config.beta2 * v[i] + (1.0 - config.beta2) * grads[i] * grads[i]

        var m_hat = m[i] / bias_corr1
        var v_hat = v[i] / bias_corr2
        var update = m_hat / (sqrt(v_hat) + config.eps) + config.weight_decay * params[i]
        params[i] -= config.learning_rate * update


def _adamw_update_scalar(
    mut param: Float32,
    grad: Float32,
    mut m: Float32,
    mut v: Float32,
    bias_corr1: Float32,
    bias_corr2: Float32,
    config: AdamWConfig,
) -> None:
    m = config.beta1 * m + (1.0 - config.beta1) * grad
    v = config.beta2 * v + (1.0 - config.beta2) * grad * grad

    var m_hat = m / bias_corr1
    var v_hat = v / bias_corr2
    var update = m_hat / (sqrt(v_hat) + config.eps) + config.weight_decay * param
    param -= config.learning_rate * update
