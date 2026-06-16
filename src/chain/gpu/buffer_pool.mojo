"""Persistent GPU buffers for full-device training."""

from std.gpu.host import DeviceBuffer

from ..adamw import AdamWConfig, one_minus_pow
from ..buffer import ChainData
from ..env import ternary_threshold
from ..grads import ModelGrads
from ..micro_net_batch import BatchMicroNet, _ceildiv
from .adamw import enqueue_adamw_apply
from .backward import enqueue_backward, enqueue_backward_fp32
from .device import GpuDevice, f32_ptr_offset
from .quantize import enqueue_quantize_ste
from .ternary_matmul import (
    float_matmul_batch_kernel,
    head_reduce_f32_kernel,
    head_reduce_kernel,
    scale_kernel,
    swiglu_forward_kernel,
    ternary_matmul_batch_kernel,
)


struct GpuTrainState(Movable):
    """Owns all device buffers for one GPU train run."""

    var gpu: GpuDevice
    var input_dim: Int
    var hidden_dim: Int
    var max_batch_size: Int
    var gate_size: Int
    var head_size: Int
    var timestep: Int
    var threshold: Float32
    var use_ternary: Bool

    var x_dev: DeviceBuffer[DType.float32]
    var y_dev: DeviceBuffer[DType.float32]
    var gate_shadow_dev: DeviceBuffer[DType.float32]
    var up_shadow_dev: DeviceBuffer[DType.float32]
    var head_shadow_dev: DeviceBuffer[DType.float32]
    var alpha_dev: DeviceBuffer[DType.float32]
    var gate_tern_dev: DeviceBuffer[DType.int8]
    var up_tern_dev: DeviceBuffer[DType.int8]
    var head_tern_dev: DeviceBuffer[DType.int8]
    var gate_act_dev: DeviceBuffer[DType.float32]
    var up_act_dev: DeviceBuffer[DType.float32]
    var hidden_dev: DeviceBuffer[DType.float32]
    var pred_dev: DeviceBuffer[DType.float32]
    var err_dev: DeviceBuffer[DType.float32]
    var loss_partial_dev: DeviceBuffer[DType.float32]
    var loss_scalar_dev: DeviceBuffer[DType.float32]
    var grad_gate_dev: DeviceBuffer[DType.float32]
    var grad_up_dev: DeviceBuffer[DType.float32]
    var grad_head_dev: DeviceBuffer[DType.float32]
    var grad_alpha_dev: DeviceBuffer[DType.float32]
    var grad_gate_partial_dev: DeviceBuffer[DType.float32]
    var grad_up_partial_dev: DeviceBuffer[DType.float32]
    var grad_head_partial_dev: DeviceBuffer[DType.float32]
    var grad_alpha_partial_dev: DeviceBuffer[DType.float32]
    var gate_m_dev: DeviceBuffer[DType.float32]
    var gate_v_dev: DeviceBuffer[DType.float32]
    var up_m_dev: DeviceBuffer[DType.float32]
    var up_v_dev: DeviceBuffer[DType.float32]
    var head_m_dev: DeviceBuffer[DType.float32]
    var head_v_dev: DeviceBuffer[DType.float32]
    var alpha_m_dev: DeviceBuffer[DType.float32]
    var alpha_v_dev: DeviceBuffer[DType.float32]

    def __init__(
        out self,
        data: ChainData,
        model: BatchMicroNet,
        max_batch_size: Int,
    ) raises:
        self.gpu = GpuDevice()
        self.input_dim = model.input_dim
        self.hidden_dim = model.hidden_dim
        self.max_batch_size = max_batch_size
        self.gate_size = model.gate_size()
        self.head_size = model.hidden_dim
        self.timestep = 0
        self.threshold = ternary_threshold()
        self.use_ternary = model.use_ternary

        var n_x = data.n_samples * data.input_dim
        var bh = max_batch_size * model.hidden_dim
        var bhd = max_batch_size * model.hidden_dim * model.input_dim

        self.x_dev = self.gpu.create_device_f32(n_x)
        self.y_dev = self.gpu.create_device_f32(data.n_samples)
        self.gate_shadow_dev = self.gpu.create_device_f32(self.gate_size)
        self.up_shadow_dev = self.gpu.create_device_f32(self.gate_size)
        self.head_shadow_dev = self.gpu.create_device_f32(self.head_size)
        self.alpha_dev = self.gpu.create_device_f32(1)
        self.gate_tern_dev = self.gpu.create_device_i8(self.gate_size)
        self.up_tern_dev = self.gpu.create_device_i8(self.gate_size)
        self.head_tern_dev = self.gpu.create_device_i8(self.head_size)
        self.gate_act_dev = self.gpu.create_device_f32(bh)
        self.up_act_dev = self.gpu.create_device_f32(bh)
        self.hidden_dev = self.gpu.create_device_f32(bh)
        self.pred_dev = self.gpu.create_device_f32(max_batch_size)
        self.err_dev = self.gpu.create_device_f32(max_batch_size)
        self.loss_partial_dev = self.gpu.create_device_f32(max_batch_size)
        self.loss_scalar_dev = self.gpu.create_device_f32(1)
        self.grad_gate_dev = self.gpu.create_device_f32(self.gate_size)
        self.grad_up_dev = self.gpu.create_device_f32(self.gate_size)
        self.grad_head_dev = self.gpu.create_device_f32(self.head_size)
        self.grad_alpha_dev = self.gpu.create_device_f32(1)
        self.grad_gate_partial_dev = self.gpu.create_device_f32(bhd)
        self.grad_up_partial_dev = self.gpu.create_device_f32(bhd)
        self.grad_head_partial_dev = self.gpu.create_device_f32(bh)
        self.grad_alpha_partial_dev = self.gpu.create_device_f32(max_batch_size)
        self.gate_m_dev = self.gpu.create_device_f32(self.gate_size)
        self.gate_v_dev = self.gpu.create_device_f32(self.gate_size)
        self.up_m_dev = self.gpu.create_device_f32(self.gate_size)
        self.up_v_dev = self.gpu.create_device_f32(self.gate_size)
        self.head_m_dev = self.gpu.create_device_f32(self.head_size)
        self.head_v_dev = self.gpu.create_device_f32(self.head_size)
        self.alpha_m_dev = self.gpu.create_device_f32(1)
        self.alpha_v_dev = self.gpu.create_device_f32(1)

        # One-time upload of dataset and initial weights.
        self.gpu.upload_to_device_f32(data.x_data, 0, self.x_dev, len(data.x_data))
        self.gpu.upload_to_device_f32(data.y_data, 0, self.y_dev, len(data.y_data))
        self.gpu.upload_to_device_f32(model.gate_shadow, 0, self.gate_shadow_dev, self.gate_size)
        self.gpu.upload_to_device_f32(model.up_shadow, 0, self.up_shadow_dev, self.gate_size)
        self.gpu.upload_to_device_f32(model.head_shadow, 0, self.head_shadow_dev, self.head_size)
        var alpha_host = List[Float32](capacity=1)
        alpha_host.append(model.alpha)
        self.gpu.upload_to_device_f32(alpha_host, 0, self.alpha_dev, 1)
        self.gpu.zero_f32(self.gate_m_dev, self.gate_size)
        self.gpu.zero_f32(self.gate_v_dev, self.gate_size)
        self.gpu.zero_f32(self.up_m_dev, self.gate_size)
        self.gpu.zero_f32(self.up_v_dev, self.gate_size)
        self.gpu.zero_f32(self.head_m_dev, self.head_size)
        self.gpu.zero_f32(self.head_v_dev, self.head_size)
        self.gpu.zero_f32(self.alpha_m_dev, 1)
        self.gpu.zero_f32(self.alpha_v_dev, 1)
        self.gpu.synchronize()

    def download_shadow(mut self, mut model: BatchMicroNet) raises -> None:
        """Optional: sync shadow weights back to host model."""
        model.gate_shadow = self.gpu.download_f32(self.gate_shadow_dev, self.gate_size)
        model.up_shadow = self.gpu.download_f32(self.up_shadow_dev, self.gate_size)
        model.head_shadow = self.gpu.download_f32(self.head_shadow_dev, self.head_size)
        model.alpha = self.gpu.download_scalar_f32(self.alpha_dev)

    def download_grads(self, mut grads: ModelGrads) raises -> None:
        """Download accumulated grads for regression tests."""
        grads.gate = self.gpu.download_f32(self.grad_gate_dev, self.gate_size)
        grads.up = self.gpu.download_f32(self.grad_up_dev, self.gate_size)
        grads.head = self.gpu.download_f32(self.grad_head_dev, self.head_size)
        grads.alpha = self.gpu.download_scalar_f32(self.grad_alpha_dev)

    def _enqueue_forward(
        mut self,
        start: Int,
        batch_size: Int,
    ) raises -> None:
        var ctx = self.gpu.ctx
        var bh = batch_size * self.hidden_dim
        var x_ptr = f32_ptr_offset(self.x_dev.unsafe_ptr(), start * self.input_dim)

        var matmul_blocks = _ceildiv(bh, 256)
        ctx.enqueue_function[ternary_matmul_batch_kernel, ternary_matmul_batch_kernel](
            x_ptr,
            self.gate_tern_dev.unsafe_ptr(),
            self.gate_act_dev.unsafe_ptr(),
            batch_size,
            self.input_dim,
            self.hidden_dim,
            grid_dim=matmul_blocks,
            block_dim=256,
        )
        ctx.enqueue_function[ternary_matmul_batch_kernel, ternary_matmul_batch_kernel](
            x_ptr,
            self.up_tern_dev.unsafe_ptr(),
            self.up_act_dev.unsafe_ptr(),
            batch_size,
            self.input_dim,
            self.hidden_dim,
            grid_dim=matmul_blocks,
            block_dim=256,
        )
        ctx.enqueue_function[swiglu_forward_kernel, swiglu_forward_kernel](
            self.gate_act_dev.unsafe_ptr(),
            self.up_act_dev.unsafe_ptr(),
            self.hidden_dev.unsafe_ptr(),
            batch_size,
            self.hidden_dim,
            grid_dim=matmul_blocks,
            block_dim=256,
        )
        ctx.enqueue_function[head_reduce_kernel, head_reduce_kernel](
            self.hidden_dev.unsafe_ptr(),
            self.head_tern_dev.unsafe_ptr(),
            self.pred_dev.unsafe_ptr(),
            batch_size,
            self.hidden_dim,
            grid_dim=batch_size,
            block_dim=1,
        )
        ctx.enqueue_function[scale_kernel, scale_kernel](
            self.pred_dev.unsafe_ptr(),
            self.alpha_dev.unsafe_ptr(),
            batch_size,
            grid_dim=_ceildiv(batch_size, 256),
            block_dim=256,
        )

    def _enqueue_forward_fp32(
        mut self,
        start: Int,
        batch_size: Int,
    ) raises -> None:
        var ctx = self.gpu.ctx
        var bh = batch_size * self.hidden_dim
        var x_ptr = f32_ptr_offset(self.x_dev.unsafe_ptr(), start * self.input_dim)

        var matmul_blocks = _ceildiv(bh, 256)
        ctx.enqueue_function[float_matmul_batch_kernel, float_matmul_batch_kernel](
            x_ptr,
            self.gate_shadow_dev.unsafe_ptr(),
            self.gate_act_dev.unsafe_ptr(),
            batch_size,
            self.input_dim,
            self.hidden_dim,
            grid_dim=matmul_blocks,
            block_dim=256,
        )
        ctx.enqueue_function[float_matmul_batch_kernel, float_matmul_batch_kernel](
            x_ptr,
            self.up_shadow_dev.unsafe_ptr(),
            self.up_act_dev.unsafe_ptr(),
            batch_size,
            self.input_dim,
            self.hidden_dim,
            grid_dim=matmul_blocks,
            block_dim=256,
        )
        ctx.enqueue_function[swiglu_forward_kernel, swiglu_forward_kernel](
            self.gate_act_dev.unsafe_ptr(),
            self.up_act_dev.unsafe_ptr(),
            self.hidden_dev.unsafe_ptr(),
            batch_size,
            self.hidden_dim,
            grid_dim=matmul_blocks,
            block_dim=256,
        )
        ctx.enqueue_function[head_reduce_f32_kernel, head_reduce_f32_kernel](
            self.hidden_dev.unsafe_ptr(),
            self.head_shadow_dev.unsafe_ptr(),
            self.pred_dev.unsafe_ptr(),
            batch_size,
            self.hidden_dim,
            grid_dim=batch_size,
            block_dim=1,
        )
        ctx.enqueue_function[scale_kernel, scale_kernel](
            self.pred_dev.unsafe_ptr(),
            self.alpha_dev.unsafe_ptr(),
            batch_size,
            grid_dim=_ceildiv(batch_size, 256),
            block_dim=256,
        )

    def _enqueue_quantize(mut self) raises -> None:
        var ctx = self.gpu.ctx
        enqueue_quantize_ste(
            ctx, self.gate_shadow_dev, self.gate_tern_dev, self.gate_size, self.threshold
        )
        enqueue_quantize_ste(
            ctx, self.up_shadow_dev, self.up_tern_dev, self.gate_size, self.threshold
        )
        enqueue_quantize_ste(
            ctx, self.head_shadow_dev, self.head_tern_dev, self.head_size, self.threshold
        )

    def _zero_grads(mut self) raises -> None:
        self.gpu.zero_f32(self.grad_gate_dev, self.gate_size)
        self.gpu.zero_f32(self.grad_up_dev, self.gate_size)
        self.gpu.zero_f32(self.grad_head_dev, self.head_size)
        self.gpu.zero_f32(self.grad_alpha_dev, 1)

    def backward_only(mut self, start: Int, batch_size: Int) raises -> Float32:
        """Forward + backward without AdamW (for grad regression)."""
        if self.use_ternary:
            self._enqueue_quantize()
            self._enqueue_forward(start, batch_size)
        else:
            self._enqueue_forward_fp32(start, batch_size)
        self._zero_grads()
        if self.use_ternary:
            enqueue_backward(
                self.gpu.ctx,
                self.x_dev,
                start * self.input_dim,
                self.y_dev,
                start,
                self.gate_act_dev,
                self.up_act_dev,
                self.hidden_dev,
                self.pred_dev,
                self.head_tern_dev,
                self.alpha_dev,
                self.err_dev,
                self.loss_partial_dev,
                self.grad_alpha_partial_dev,
                self.grad_head_partial_dev,
                self.grad_gate_partial_dev,
                self.grad_up_partial_dev,
                self.grad_gate_dev,
                self.grad_up_dev,
                self.grad_head_dev,
                self.grad_alpha_dev,
                self.loss_scalar_dev,
                batch_size,
                self.hidden_dim,
                self.input_dim,
            )
        else:
            enqueue_backward_fp32(
                self.gpu.ctx,
                self.x_dev,
                start * self.input_dim,
                self.y_dev,
                start,
                self.gate_act_dev,
                self.up_act_dev,
                self.hidden_dev,
                self.pred_dev,
                self.head_shadow_dev,
                self.alpha_dev,
                self.err_dev,
                self.loss_partial_dev,
                self.grad_alpha_partial_dev,
                self.grad_head_partial_dev,
                self.grad_gate_partial_dev,
                self.grad_up_partial_dev,
                self.grad_gate_dev,
                self.grad_up_dev,
                self.grad_head_dev,
                self.grad_alpha_dev,
                self.loss_scalar_dev,
                batch_size,
                self.hidden_dim,
                self.input_dim,
            )
        self.gpu.synchronize()
        return self.gpu.download_scalar_f32(self.loss_scalar_dev)

    def train_step(mut self, start: Int, batch_size: Int, config: AdamWConfig) raises -> Float32:
        """Full GPU train step: quantize (optional), forward, backward, AdamW."""
        if self.use_ternary:
            self._enqueue_quantize()
            self._enqueue_forward(start, batch_size)
        else:
            self._enqueue_forward_fp32(start, batch_size)
        self._zero_grads()
        if self.use_ternary:
            enqueue_backward(
                self.gpu.ctx,
                self.x_dev,
                start * self.input_dim,
                self.y_dev,
                start,
                self.gate_act_dev,
                self.up_act_dev,
                self.hidden_dev,
                self.pred_dev,
                self.head_tern_dev,
                self.alpha_dev,
                self.err_dev,
                self.loss_partial_dev,
                self.grad_alpha_partial_dev,
                self.grad_head_partial_dev,
                self.grad_gate_partial_dev,
                self.grad_up_partial_dev,
                self.grad_gate_dev,
                self.grad_up_dev,
                self.grad_head_dev,
                self.grad_alpha_dev,
                self.loss_scalar_dev,
                batch_size,
                self.hidden_dim,
                self.input_dim,
            )
        else:
            enqueue_backward_fp32(
                self.gpu.ctx,
                self.x_dev,
                start * self.input_dim,
                self.y_dev,
                start,
                self.gate_act_dev,
                self.up_act_dev,
                self.hidden_dev,
                self.pred_dev,
                self.head_shadow_dev,
                self.alpha_dev,
                self.err_dev,
                self.loss_partial_dev,
                self.grad_alpha_partial_dev,
                self.grad_head_partial_dev,
                self.grad_gate_partial_dev,
                self.grad_up_partial_dev,
                self.grad_gate_dev,
                self.grad_up_dev,
                self.grad_head_dev,
                self.grad_alpha_dev,
                self.loss_scalar_dev,
                batch_size,
                self.hidden_dim,
                self.input_dim,
            )

        self.timestep += 1
        var bias_corr1 = one_minus_pow(config.beta1, self.timestep)
        var bias_corr2 = one_minus_pow(config.beta2, self.timestep)
        enqueue_adamw_apply(
            self.gpu.ctx,
            self.gate_shadow_dev,
            self.up_shadow_dev,
            self.head_shadow_dev,
            self.alpha_dev,
            self.grad_gate_dev,
            self.grad_up_dev,
            self.grad_head_dev,
            self.grad_alpha_dev,
            self.gate_m_dev,
            self.gate_v_dev,
            self.up_m_dev,
            self.up_v_dev,
            self.head_m_dev,
            self.head_v_dev,
            self.alpha_m_dev,
            self.alpha_v_dev,
            self.gate_size,
            self.head_size,
            bias_corr1,
            bias_corr2,
            config,
        )
        self.gpu.synchronize()
        return self.gpu.download_scalar_f32(self.loss_scalar_dev)
