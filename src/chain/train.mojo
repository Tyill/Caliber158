"""Training loop: batched MSE + STE backward + AdamW."""

from .adamw import AdamWConfig, AdamWState
from .buffer import ChainData
from .device import DeviceKind, cuda_available
from .grads import ModelGrads
from .gpu.batch_step import train_step_gpu
from .gpu.buffer_pool import GpuTrainState
from .holdout import HoldoutSplit
from .metrics import relative_mse
from .micro_net_batch import BatchMicroNet
from .rng import lcg_next, unit_float


@fieldwise_init
struct TrainConfig(Copyable, Movable):
    var hidden_dim: Int
    var epochs: Int
    var batch_size: Int
    var learning_rate: Float32
    var weight_decay: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var log_every: Int
    var device: DeviceKind

    def adamw_config(self) -> AdamWConfig:
        return AdamWConfig(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )


def _log_epoch_metrics(
    epoch: Int,
    train_mse: Float32,
    holdout: HoldoutSplit,
    mut model: BatchMicroNet,
) raises -> None:
    if holdout.holdout.n_samples == 0:
        print("epoch ", epoch, " train_mse=", train_mse)
        return

    var holdout_mse = model.eval_mse(holdout.holdout)
    var rel = relative_mse(holdout_mse, holdout.var_y_holdout)
    print(
        "epoch ",
        epoch,
        " train_mse=",
        train_mse,
        " holdout_mse=",
        holdout_mse,
        " rel_holdout=",
        rel,
    )


def _train_epochs_cpu(
    mut model: BatchMicroNet,
    mut optimizer: AdamWState,
    mut grads: ModelGrads,
    train_data: ChainData,
    holdout: HoldoutSplit,
    config: TrainConfig,
) raises -> None:
    for epoch in range(config.epochs):
        var total_loss: Float32 = 0.0
        var batches = 0
        var i = 0
        while i < train_data.n_samples:
            var loss = model.train_step_cpu(train_data, i, config.batch_size, grads)
            optimizer.apply(model, grads, config.adamw_config())
            total_loss += loss
            batches += 1
            i += config.batch_size
        if epoch % config.log_every == 0 or epoch == config.epochs - 1:
            var avg = total_loss / Float32(batches)
            _log_epoch_metrics(epoch, avg, holdout, model)


def _train_epochs_gpu(
    mut model: BatchMicroNet,
    train_data: ChainData,
    holdout: HoldoutSplit,
    config: TrainConfig,
) raises -> None:
    var state = GpuTrainState(train_data, model, config.batch_size)
    var adamw_cfg = config.adamw_config()

    for epoch in range(config.epochs):
        var total_loss: Float32 = 0.0
        var batches = 0
        var i = 0
        while i < train_data.n_samples:
            var loss = train_step_gpu(state, i, config.batch_size, adamw_cfg)
            total_loss += loss
            batches += 1
            i += config.batch_size
        if epoch % config.log_every == 0 or epoch == config.epochs - 1:
            var avg = total_loss / Float32(batches)
            if holdout.holdout.n_samples > 0:
                state.download_shadow(model)
            _log_epoch_metrics(epoch, avg, holdout, model)

    if holdout.holdout.n_samples > 0:
        state.download_shadow(model)


def train_chain(
    mut model: BatchMicroNet,
    train_data: ChainData,
    holdout: HoldoutSplit,
    config: TrainConfig,
) raises -> None:
    """Epoch loop with STE + AdamW."""
    var use_gpu = config.device.is_cuda() and cuda_available()

    print(
        "training chain: train_samples=",
        train_data.n_samples,
        " holdout_samples=",
        holdout.holdout.n_samples,
        " hidden=",
        config.hidden_dim,
        " params=",
        model.param_count(),
        " lr=",
        config.learning_rate,
        " device=",
        config.device.label() if use_gpu else DeviceKind.cpu().label(),
        " backend=mojo",
    )
    if holdout.holdout.n_samples > 0:
        print(
            "holdout metrics: var_y_train=",
            holdout.var_y_train,
            " var_y_holdout=",
            holdout.var_y_holdout,
        )

    if use_gpu:
        _train_epochs_gpu(model, train_data, holdout, config)
    else:
        var optimizer = AdamWState.from_model(model)
        var grads = ModelGrads.zeros(
            len(model.gate_shadow),
            len(model.up_shadow),
            len(model.head_shadow),
        )
        _train_epochs_cpu(model, optimizer, grads, train_data, holdout, config)

    if holdout.holdout.n_samples > 0:
        var final_holdout = model.eval_mse(holdout.holdout)
        print(
            "final holdout_mse=",
            final_holdout,
            " rel_holdout=",
            relative_mse(final_holdout, holdout.var_y_holdout),
            " (phase1 rel target 0.001)",
        )


def init_random_weights(mut model: BatchMicroNet, scale: Float32 = 0.1) -> None:
    """Small random init for shadow weights (LCG, no external RNG)."""
    var seed: UInt64 = 0xC158_C158
    for i in range(len(model.gate_shadow)):
        seed = lcg_next(seed)
        model.gate_shadow[i] = (unit_float(seed) * 2.0 - 1.0) * scale
        seed = lcg_next(seed)
        model.up_shadow[i] = (unit_float(seed) * 2.0 - 1.0) * scale
    for i in range(len(model.head_shadow)):
        seed = lcg_next(seed)
        model.head_shadow[i] = (unit_float(seed) * 2.0 - 1.0) * scale
