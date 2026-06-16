#!/usr/bin/env python3
"""Extract one Qwen MLP scalar chain and write a Caliber158 training dataset."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_config import CaliberEnv, load_env, resolve_teacher_device

MAGIC = b"CAL158"
VERSION = 1


@dataclass(frozen=True)
class ChainRef:
    model: str
    layer: int
    neuron: int
    hidden_size: int
    intermediate_size: int

    @property
    def chain_id(self) -> int:
        return self.layer * self.intermediate_size + self.neuron

    def filename(self) -> str:
        return f"L{self.layer:02d}_N{self.neuron:04d}.bin"


def swiglu_chain_numpy(
    gate_w: np.ndarray, up_w: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """Teacher on CPU: SiLU(gate·x) * (up·x) for each row of x."""
    gate = x @ gate_w
    up = x @ up_w
    return np.multiply(np.multiply(gate, 1.0 / (1.0 + np.exp(-gate))), up)


def swiglu_chain_torch(gate_w, up_w, x):
    """Teacher on GPU: SiLU(gate·x) * (up·x) for each row of x."""
    import torch

    gate = x @ gate_w
    up = x @ up_w
    return torch.nn.functional.silu(gate) * up


def write_dataset(path: Path, x: np.ndarray, y: np.ndarray, input_dim: int) -> None:
    """Write CAL158 binary format (see docs/architecture.md)."""
    n, d = x.shape
    if d != input_dim:
        raise ValueError(f"expected input_dim={input_dim}, got {d}")
    if y.shape != (n,):
        raise ValueError(f"expected y shape ({n},), got {y.shape}")

    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, n, d))
        f.write(x.astype(np.float32, copy=False).tobytes(order="C"))
        f.write(y.astype(np.float32, copy=False).tobytes(order="C"))


def load_qwen_config(model_name: str):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name)
    return int(cfg.hidden_size), int(cfg.intermediate_size), int(cfg.num_hidden_layers)


def load_qwen_weights(
    model_name: str,
    layer: int,
    neuron_idx: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Load gate/up rows for one intermediate neuron; return weights + dims."""
    import torch
    from transformers import AutoModelForCausalLM

    hidden, intermediate, n_layers = load_qwen_config(model_name)
    if not (0 <= layer < n_layers):
        raise ValueError(f"layer must be in [0, {n_layers}), got {layer}")
    if not (0 <= neuron_idx < intermediate):
        raise ValueError(f"neuron must be in [0, {intermediate}), got {neuron_idx}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    if device == "cuda":
        model = model.to("cuda")
    mlp = model.model.layers[layer].mlp
    gate_w = mlp.gate_proj.weight[neuron_idx].detach().cpu().numpy()
    up_w = mlp.up_proj.weight[neuron_idx].detach().cpu().numpy()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return (
        gate_w.astype(np.float32),
        up_w.astype(np.float32),
        hidden,
        intermediate,
        n_layers,
    )


def synthetic_teacher(
    gate_w: np.ndarray | None,
    up_w: np.ndarray | None,
    n_samples: int,
    seed: int,
    input_dim: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) using real or random teacher weights."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, input_dim), dtype=np.float32)

    if device == "cuda":
        import torch

        dev = torch.device("cuda")
        x_t = torch.from_numpy(x).to(dev)
        if gate_w is None or up_w is None:
            gen = torch.Generator(device=dev)
            gen.manual_seed(seed)
            gate_t = torch.randn(input_dim, generator=gen, device=dev, dtype=torch.float32) * 0.02
            up_t = torch.randn(input_dim, generator=gen, device=dev, dtype=torch.float32) * 0.02
        else:
            gate_t = torch.from_numpy(gate_w).to(dev)
            up_t = torch.from_numpy(up_w).to(dev)
        y = swiglu_chain_torch(gate_t, up_t, x_t).cpu().numpy()
        return x, y.astype(np.float32)

    if gate_w is None or up_w is None:
        gate_w = rng.standard_normal(input_dim, dtype=np.float32) * 0.02
        up_w = rng.standard_normal(input_dim, dtype=np.float32) * 0.02

    y = swiglu_chain_numpy(gate_w, up_w, x)
    return x, y


def write_metadata(path: Path, ref: ChainRef, n_samples: int, seed: int) -> None:
    meta = {
        **asdict(ref),
        "chain_id": ref.chain_id,
        "n_samples": n_samples,
        "seed": seed,
    }
    path.write_text(json.dumps(meta, indent=2) + "\n")


def apply_env_defaults(env: CaliberEnv, parser: argparse.ArgumentParser) -> None:
    """Set argparse defaults from env (CLI flags still override)."""
    parser.set_defaults(
        output=None,
        samples=env.samples,
        seed=env.seed,
        model=env.model,
        synthetic=env.synthetic,
        layer=env.layer,
        neuron=env.neuron,
        input_dim=env.hidden_size,
        data_dir=env.data_dir,
    )


def main() -> None:
    env = load_env()
    device = resolve_teacher_device()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-n", "--samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--neuron", type=int)
    parser.add_argument("--input-dim", type=int, dest="input_dim")
    parser.add_argument("--data-dir", type=Path, dest="data_dir")
    apply_env_defaults(env, parser)
    args = parser.parse_args()

    gate_w: np.ndarray | None = None
    up_w: np.ndarray | None = None
    hidden = args.input_dim
    intermediate = env.intermediate_size
    model_name = "synthetic" if args.synthetic else args.model

    if not args.synthetic:
        gate_w, up_w, hidden, intermediate, _ = load_qwen_weights(
            args.model, args.layer, args.neuron, device
        )

    ref = ChainRef(
        model=model_name,
        layer=args.layer,
        neuron=args.neuron,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )

    out = args.output or args.data_dir / ref.filename()
    x, y = synthetic_teacher(gate_w, up_w, args.samples, args.seed, hidden, device)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_dataset(out, x, y, hidden)
    write_metadata(out.with_suffix(".json"), ref, args.samples, args.seed)

    print(
        f"wrote {out}\n"
        f"  model={model_name}  layer={args.layer}  neuron={args.neuron}\n"
        f"  chain_id={ref.chain_id}  samples={args.samples}  input_dim={hidden}\n"
        f"  device={device}"
    )


if __name__ == "__main__":
    main()
