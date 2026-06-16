"""On-device STE ternary quantization."""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer


def quantize_ste_kernel(
    shadow: UnsafePointer[Float32, MutAnyOrigin],
    ternary: UnsafePointer[Int8, MutAnyOrigin],
    n: Int,
    threshold: Float32,
):
    """Map shadow FP32 weights to {-1, 0, 1}."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n:
        return
    var w = shadow[i]
    if w > threshold:
        ternary[i] = 1
    elif w < -threshold:
        ternary[i] = -1
    else:
        ternary[i] = 0


def enqueue_quantize_ste(
    ctx: DeviceContext,
    shadow: DeviceBuffer[DType.float32],
    mut ternary: DeviceBuffer[DType.int8],
    n: Int,
    threshold: Float32,
) raises -> None:
    var blocks = (n + 255) // 256
    ctx.enqueue_function[quantize_ste_kernel, quantize_ste_kernel](
        shadow.unsafe_ptr(),
        ternary.unsafe_ptr(),
        n,
        threshold,
        grid_dim=blocks,
        block_dim=256,
    )
