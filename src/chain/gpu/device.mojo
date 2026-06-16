"""GPU device context and buffer management for student training."""

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

    def upload_i8(self, host: List[TernaryWeight], count: Int) raises -> DeviceBuffer[DType.int8]:
        var host_buf = self.ctx.enqueue_create_host_buffer[DType.int8](count)
        self.ctx.synchronize()
        for i in range(count):
            host_buf[i] = host[i]
        var dev_buf = self.ctx.enqueue_create_buffer[DType.int8](count)
        self.ctx.enqueue_copy(dev_buf, host_buf)
        self.ctx.synchronize()
        return dev_buf^

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

    def synchronize(self) raises -> None:
        self.ctx.synchronize()
