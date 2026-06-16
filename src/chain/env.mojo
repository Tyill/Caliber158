"""Runtime configuration from CALIBER158_* environment variables."""

from std.os import getenv

from .arch import ArchKind, resolve_arch_from_env
from .device import DeviceKind, resolve_device_from_env, train_backend_from_env


@fieldwise_init
struct TrainEnv(Copyable, Movable):
    var hidden_dim: Int
    var dataset_path: String
    var hidden_size: Int
    var epochs: Int
    var batch_size: Int
    var learning_rate: Float32
    var weight_decay: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var log_every: Int
    var init_scale: Float32
    var init_scale_block2: Float32
    var block2_residual_scale: Float32
    var ternary_threshold: Float32
    var holdout_fraction: Float32
    var seed: UInt64
    var smoke_epochs: Int
    var smoke_batch_size: Int
    var smoke_samples: Int
    var model_name: String
    var arch: ArchKind
    var device: DeviceKind
    var train_backend: String

    @staticmethod
    def load() raises -> TrainEnv:
        var hidden_dim = _env_int("CALIBER158_HIDDEN_DIM", 128)
        var init_scale = _env_float("CALIBER158_INIT_SCALE", 0.1)
        return TrainEnv(
            hidden_dim=hidden_dim,
            dataset_path=_env_string("CALIBER158_DATASET", "data/chains/L00_N0000.bin"),
            hidden_size=_env_int("CALIBER158_HIDDEN_SIZE", 896),
            epochs=_env_int("CALIBER158_EPOCHS", 10),
            batch_size=_env_int("CALIBER158_BATCH_SIZE", 64),
            learning_rate=_env_float("CALIBER158_LR", 0.001),
            weight_decay=_env_float("CALIBER158_WEIGHT_DECAY", 0.01),
            beta1=_env_float("CALIBER158_ADAM_BETA1", 0.9),
            beta2=_env_float("CALIBER158_ADAM_BETA2", 0.999),
            eps=_env_float("CALIBER158_ADAM_EPS", 1e-8),
            log_every=_env_int("CALIBER158_LOG_EVERY", 1),
            init_scale=init_scale,
            init_scale_block2=_resolve_init_scale_block2(init_scale, hidden_dim),
            block2_residual_scale=_resolve_block2_residual_scale(hidden_dim),
            ternary_threshold=_env_float("CALIBER158_TERNARY_THRESHOLD", 0.0),
            holdout_fraction=_env_float("CALIBER158_HOLDOUT_FRACTION", 0.1),
            seed=_env_seed("CALIBER158_SEED", 42),
            smoke_epochs=_env_int("CALIBER158_SMOKE_EPOCHS", 3),
            smoke_batch_size=_env_int("CALIBER158_SMOKE_BATCH_SIZE", 32),
            smoke_samples=_env_int("CALIBER158_SMOKE_SAMPLES", 128),
            model_name=_env_string("CALIBER158_MODEL", "Qwen/Qwen2.5-0.5B"),
            arch=resolve_arch_from_env(),
            device=resolve_device_from_env(),
            train_backend=train_backend_from_env(),
        )


def ternary_threshold() -> Float32:
    return _env_float("CALIBER158_TERNARY_THRESHOLD", 0.0)


def _resolve_init_scale_block2(init_scale: Float32, hidden_dim: Int) raises -> Float32:
    """Block2 (v1 H→H) init; env override or init_scale/hidden_dim (fan-in scaling)."""
    var raw = getenv("CALIBER158_INIT_SCALE_BLOCK2", "").strip()
    if raw.byte_length() > 0:
        return _env_float("CALIBER158_INIT_SCALE_BLOCK2", init_scale)
    if hidden_dim < 1:
        raise Error("CALIBER158_HIDDEN_DIM must be >= 1 for block2 init scale")
    return init_scale / Float32(hidden_dim)


def _resolve_block2_residual_scale(hidden_dim: Int) raises -> Float32:
    """Scale h2 before residual add; env override or 1/hidden_dim."""
    var raw = getenv("CALIBER158_BLOCK2_RESIDUAL_SCALE", "").strip()
    if raw.byte_length() > 0:
        return _env_float("CALIBER158_BLOCK2_RESIDUAL_SCALE", 1.0)
    if hidden_dim < 1:
        raise Error("CALIBER158_HIDDEN_DIM must be >= 1 for block2 residual scale")
    return 1.0 / Float32(hidden_dim)


def _env_string(key: String, default: String) -> String:
    return getenv(key, default)


def _env_int(key: String, default: Int) -> Int:
    var raw = getenv(key, String(default))
    try:
        return Int(raw)
    except:
        return default


def _env_float(key: String, default: Float32) -> Float32:
    var raw = getenv(key, String(default))
    try:
        return Float32(Float64(raw))
    except:
        return default


def _env_seed(key: String, default: Int) -> UInt64:
    var raw = getenv(key, String(default))
    try:
        return UInt64(Int(raw))
    except:
        return UInt64(default)
