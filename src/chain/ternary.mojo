"""Ternary weight quantization and matvec kernels."""

from std.math import exp

from .env import ternary_threshold

comptime TernaryWeight = Int8  # -1, 0, or 1


@always_inline
def quantize_ternary(w: Float32) -> TernaryWeight:
    """Map shadow FP32 weight to {-1, 0, 1} (threshold from env)."""
    var threshold = ternary_threshold()
    if w > threshold:
        return 1
    if w < -threshold:
        return -1
    return 0


@always_inline
def sigmoid(x: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-x))


@always_inline
def silu(x: Float32) -> Float32:
    """SiLU activation: x * sigmoid(x)."""
    return x * sigmoid(x)


@always_inline
def silu_derivative(x: Float32) -> Float32:
    """d/dx SiLU(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))."""
    var s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def ternary_matvec(
    weights: List[TernaryWeight],
    input: List[Float32],
    in_dim: Int,
    out_dim: Int,
    mut output: List[Float32],
) raises:
    """y[j] = sum_i W[j,i] * x[i] with ternary W. Row-major [out_dim, in_dim]."""
    for j in range(out_dim):
        var acc: Float32 = 0.0
        var row_base = j * in_dim
        for i in range(in_dim):
            var w = weights[row_base + i]
            if w != 0:
                acc += Float32(w) * input[i]
        output[j] = acc


def quantize_weights(shadow: List[Float32], mut ternary: List[TernaryWeight]) -> None:
    """Fill ternary buffer from shadow weights (same length)."""
    for i in range(len(shadow)):
        ternary[i] = quantize_ternary(shadow[i])
