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
