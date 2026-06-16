"""GPU device context and buffer management for student training."""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from ..device import cuda_available
from ..ternary import TernaryWeight


struct GpuDevice(Copyable, Movable):
    """Thin wrapper around DeviceContext for train buffers."""

    var ctx: DeviceContext

    def __init__(out self) raises:
        if not cuda_available():
            raise Error("GPU requested but no compatible accelerator found")
        self.ctx = DeviceContext()

    @staticmethod
    def is_available() -> Bool:
        return cuda_available()

    def create_device_f32(self, size: Int) raises -> DeviceBuffer[DType.float32]:
        var buf = self.ctx.enqueue_create_buffer[DType.float32](size)
        self.ctx.synchronize()
        return buf

    def create_device_i8(self, size: Int) raises -> DeviceBuffer[DType.int8]:
        var buf = self.ctx.enqueue_create_buffer[DType.int8](size)
        self.ctx.synchronize()
        return buf

    def create_host_f32(self, size: Int) raises -> HostBuffer[DType.float32]:
        var buf = self.ctx.enqueue_create_host_buffer[DType.float32](size)
        self.ctx.synchronize()
        return buf

    def upload_list_f32(
        self,
        host: List[Float32],
        offset: Int,
        count: Int,
    ) raises -> DeviceBuffer[DType.float32]:
        var host_buf = self.create_host_f32(count)
        for i in range(count):
            host_buf[i] = host[offset + i]
        var dev_buf = self.create_device_f32(count)
        self.ctx.enqueue_copy(dev_buf, host_buf)
        self.ctx.synchronize()
        return dev_buf^

    def upload_to_device_f32(
        self,
        host: List[Float32],
        host_offset: Int,
        mut dev: DeviceBuffer[DType.float32],
        count: Int,
    ) raises -> None:
        """Copy host slice into device buffer (full buffer, host_offset start)."""
        var host_buf = self.create_host_f32(count)
        for i in range(count):
            host_buf[i] = host[host_offset + i]
        self.ctx.enqueue_copy(dev, host_buf)
        self.ctx.synchronize()

    def upload_scalar_f32(self, value: Float32) raises -> DeviceBuffer[DType.float32]:
        var dev = self.create_device_f32(1)
        var host_buf = self.create_host_f32(1)
        host_buf[0] = value
        self.ctx.enqueue_copy(dev, host_buf)
        self.ctx.synchronize()
        return dev^

    def upload_i8(self, host: List[TernaryWeight], count: Int) raises -> DeviceBuffer[DType.int8]:
        var host_buf = self.ctx.enqueue_create_host_buffer[DType.int8](count)
        self.ctx.synchronize()
        for i in range(count):
            host_buf[i] = host[i]
        var dev_buf = self.create_device_i8(count)
        self.ctx.enqueue_copy(dev_buf, host_buf)
        self.ctx.synchronize()
        return dev_buf^

    def upload_to_device_i8(
        self,
        host: List[TernaryWeight],
        mut dev: DeviceBuffer[DType.int8],
        count: Int,
    ) raises -> None:
        var host_buf = self.ctx.enqueue_create_host_buffer[DType.int8](count)
        self.ctx.synchronize()
        for i in range(count):
            host_buf[i] = host[i]
        self.ctx.enqueue_copy(dev, host_buf)
        self.ctx.synchronize()

    def download_f32(
        self,
        dev: DeviceBuffer[DType.float32],
        count: Int,
    ) raises -> List[Float32]:
        var host_buf = self.create_host_f32(count)
        self.ctx.enqueue_copy(host_buf, dev)
        self.ctx.synchronize()
        var out = List[Float32](capacity=count)
        for i in range(count):
            out.append(host_buf[i])
        return out^

    def download_scalar_f32(self, dev: DeviceBuffer[DType.float32]) raises -> Float32:
        var host_buf = self.create_host_f32(1)
        self.ctx.enqueue_copy(host_buf, dev)
        self.ctx.synchronize()
        return host_buf[0]

    def zero_f32(self, mut dev: DeviceBuffer[DType.float32], count: Int) raises -> None:
        var blocks = _ceildiv(count, 256)
        self.ctx.enqueue_function[zero_f32_kernel, zero_f32_kernel](
            dev.unsafe_ptr(),
            count,
            grid_dim=blocks,
            block_dim=256,
        )

    def synchronize(self) raises -> None:
        self.ctx.synchronize()


def f32_ptr_offset(ptr: UnsafePointer[Float32, MutAnyOrigin], offset: Int) -> UnsafePointer[
    Float32, MutAnyOrigin
]:
    """Element offset into a device float buffer (no copy)."""
    return ptr + offset


def i8_ptr_offset(ptr: UnsafePointer[Int8, MutAnyOrigin], offset: Int) -> UnsafePointer[
    Int8, MutAnyOrigin
]:
    return ptr + offset


def _ceildiv(n: Int, d: Int) -> Int:
    return (n + d - 1) // d


def zero_f32_kernel(
    data: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n:
        return
    data[i] = 0.0


def reduce_sum_f32_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    """Single-block reduction; n must fit one block's serial loop or use small n."""
    if thread_idx.x != 0 or block_idx.x != 0:
        return
    var total: Float64 = 0.0
    for i in range(n):
        total += Float64(input[i])
    output[0] = Float32(total)


def reduce_batch_dim_gate_up_kernel(
    partial: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    hidden_dim: Int,
    input_dim: Int,
):
    """Sum partial[B,H,D] over batch -> output[H,D]. One thread per (j,i)."""
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total_elems = hidden_dim * input_dim
    if idx >= total_elems:
        return
    var j = idx // input_dim
    var i = idx % input_dim
    var acc: Float64 = 0.0
    var stride = hidden_dim * input_dim
    for b in range(batch_size):
        acc += Float64(partial[b * stride + j * input_dim + i])
    output[idx] = Float32(acc)


def reduce_batch_dim_head_kernel(
    partial: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    hidden_dim: Int,
):
    """Sum partial[B,H] over batch -> output[H]."""
    var j = Int(block_idx.x * block_dim.x + thread_idx.x)
    if j >= hidden_dim:
        return
    var acc: Float64 = 0.0
    for b in range(batch_size):
        acc += Float64(partial[b * hidden_dim + j])
    output[j] = Float32(acc)
