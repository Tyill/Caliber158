"""Load Caliber158 settings from environment and optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def apply_huggingface_paths() -> Path:
    """Point HuggingFace caches at CALIBER158_MODELS_DIR (project-local by default)."""
    load_dotenv()
    models_dir = Path(_get("CALIBER158_MODELS_DIR", "models"))
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir
    hf_home = models_dir / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    return models_dir


def load_dotenv(path: Path | None = None) -> None:
    """Populate os.environ from .env (does not override existing vars)."""
    env_file = path or ROOT / ".env"
    if not env_file.is_file():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        val = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), val)


def _get(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _get_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _get_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _get_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_teacher_device() -> str:
    """Return ``cuda`` or ``cpu`` for teacher extraction (CALIBER158_TORCH)."""
    import sys

    import torch

    pref = _get("CALIBER158_TORCH", "cuda").strip().lower()
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if pref != "cpu":
        print(
            "warning: CALIBER158_TORCH prefers GPU but CUDA is unavailable; using CPU",
            file=sys.stderr,
        )
    return "cpu"


def resolve_student_device() -> str:
    """Return ``cuda`` or ``cpu`` for Torch student train (CALIBER158_DEVICE)."""
    import sys

    import torch

    pref = _get("CALIBER158_DEVICE", "cuda").strip().lower()
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if pref != "cpu":
        print(
            "warning: CALIBER158_DEVICE prefers GPU but CUDA is unavailable; using CPU",
            file=sys.stderr,
        )
    return "cpu"


@dataclass(frozen=True)
class StudentEnv:
    hidden_dim: int
    dataset_path: Path
    hidden_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    log_every: int
    init_scale: float
    ternary_threshold: float
    smoke_epochs: int
    smoke_batch_size: int
    smoke_samples: int
    model_name: str
    device: str
    holdout_fraction: float
    split_seed: int
    use_ternary: bool
    arch: str

    @property
    def quantize_label(self) -> str:
        return "ternary" if self.use_ternary else "fp32"


def load_student_env() -> StudentEnv:
    """Load student train config from CALIBER158_* (matches src/chain/env.mojo)."""
    load_dotenv()
    dataset = Path(_get("CALIBER158_DATASET", "data/chains/L00_N0000.bin"))
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    arch = _get("CALIBER158_ARCH", "v0").strip().lower()
    if arch not in {"v0", "v1"}:
        arch = "v0"
    quantize_raw = os.environ.get("CALIBER158_QUANTIZE", "1")
    use_ternary = quantize_raw.strip() != "0"
    return StudentEnv(
        hidden_dim=_get_int("CALIBER158_HIDDEN_DIM", 128),
        dataset_path=dataset,
        hidden_size=_get_int("CALIBER158_HIDDEN_SIZE", 896),
        epochs=_get_int("CALIBER158_EPOCHS", 10),
        batch_size=_get_int("CALIBER158_BATCH_SIZE", 64),
        learning_rate=_get_float("CALIBER158_LR", 0.001),
        weight_decay=_get_float("CALIBER158_WEIGHT_DECAY", 0.01),
        beta1=_get_float("CALIBER158_ADAM_BETA1", 0.9),
        beta2=_get_float("CALIBER158_ADAM_BETA2", 0.999),
        eps=_get_float("CALIBER158_ADAM_EPS", 1e-8),
        log_every=_get_int("CALIBER158_LOG_EVERY", 1),
        init_scale=_get_float("CALIBER158_INIT_SCALE", 0.1),
        ternary_threshold=_get_float("CALIBER158_TERNARY_THRESHOLD", 0.0),
        smoke_epochs=_get_int("CALIBER158_SMOKE_EPOCHS", 3),
        smoke_batch_size=_get_int("CALIBER158_SMOKE_BATCH_SIZE", 32),
        smoke_samples=_get_int("CALIBER158_SMOKE_SAMPLES", 128),
        model_name=_get("CALIBER158_MODEL", "Qwen/Qwen2.5-0.5B"),
        device=resolve_student_device(),
        holdout_fraction=_get_float("CALIBER158_HOLDOUT_FRACTION", 0.1),
        split_seed=_get_int("CALIBER158_SEED", 42),
        use_ternary=use_ternary,
        arch=arch,
    )


@dataclass(frozen=True)
class CaliberEnv:
    model: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    layer: int
    neuron: int
    samples: int
    seed: int
    data_dir: Path
    synthetic: bool

    @property
    def chain_filename(self) -> str:
        return f"L{self.layer:02d}_N{self.neuron:04d}.bin"

    @property
    def default_dataset_path(self) -> Path:
        return self.data_dir / self.chain_filename

    @property
    def chain_id(self) -> int:
        return self.layer * self.intermediate_size + self.neuron


def load_env() -> CaliberEnv:
    apply_huggingface_paths()
    data_dir = Path(_get("CALIBER158_DATA_DIR", "data/chains"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    return CaliberEnv(
        model=_get("CALIBER158_MODEL", "Qwen/Qwen2.5-0.5B"),
        hidden_size=_get_int("CALIBER158_HIDDEN_SIZE", 896),
        intermediate_size=_get_int("CALIBER158_INTERMEDIATE_SIZE", 4864),
        num_layers=_get_int("CALIBER158_NUM_LAYERS", 24),
        layer=_get_int("CALIBER158_LAYER", 0),
        neuron=_get_int("CALIBER158_NEURON", 0),
        samples=_get_int("CALIBER158_SAMPLES", 100_000),
        seed=_get_int("CALIBER158_SEED", 42),
        data_dir=data_dir,
        synthetic=_get_bool("CALIBER158_SYNTHETIC", False),
    )
