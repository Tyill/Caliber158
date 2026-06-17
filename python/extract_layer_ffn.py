#!/usr/bin/env python3
"""Extract one Qwen layer FFN (vector output) and write CAL158L dataset."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_config import LayerExtractEnv, load_layer_extract_env, resolve_teacher_device

MAGIC = b"CAL158L"
VERSION = 1


@dataclass(frozen=True)
class LayerRef:
    model: str
    layer: int
    hidden_size: int
    intermediate_size: int
    target_kind: str

    def filename(self) -> str:
        return f"L{self.layer:02d}_ffn.bin"


def write_layer_dataset(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    """Write CAL158L binary format."""
    n, d = x.shape
    d_out = y.shape[1]
    if y.shape[0] != n:
        raise ValueError(f"x rows {n} != y rows {y.shape[0]}")
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", VERSION, n, d, d_out))
        f.write(x.astype(np.float32, copy=False).tobytes(order="C"))
        f.write(y.astype(np.float32, copy=False).tobytes(order="C"))


def ffn_forward_numpy(
    x: np.ndarray,
    gate_w: np.ndarray,
    up_w: np.ndarray,
    down_w: np.ndarray,
) -> np.ndarray:
    """Teacher FFN output (before residual): down(SiLU(x W_g^T) ⊙ (x W_u^T))."""
    gate = x @ gate_w.T
    up = x @ up_w.T
    h = np.multiply(gate / (1.0 + np.exp(-gate)), up)  # SiLU(gate) * up
    return h @ down_w.T


def ffn_forward_torch(x, gate_w, up_w, down_w):
    """GPU teacher FFN output."""
    import torch

    gate = x @ gate_w.T
    up = x @ up_w.T
    h = torch.nn.functional.silu(gate) * up
    return h @ down_w.T


def load_qwen_config(model_name: str) -> tuple[int, int, int]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name)
    return int(cfg.hidden_size), int(cfg.intermediate_size), int(cfg.num_hidden_layers)


def load_qwen_ffn_weights(
    model_name: str,
    layer: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Load gate/up/down for one transformer layer."""
    import torch
    from transformers import AutoModelForCausalLM

    hidden, intermediate, n_layers = load_qwen_config(model_name)
    if not (0 <= layer < n_layers):
        raise ValueError(f"layer must be in [0, {n_layers}), got {layer}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    if device == "cuda":
        model = model.to("cuda")
    mlp = model.model.layers[layer].mlp
    gate_w = mlp.gate_proj.weight.detach().cpu().numpy()
    up_w = mlp.up_proj.weight.detach().cpu().numpy()
    down_w = mlp.down_proj.weight.detach().cpu().numpy()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return (
        gate_w.astype(np.float32),
        up_w.astype(np.float32),
        down_w.astype(np.float32),
        hidden,
        intermediate,
        n_layers,
    )


def synthetic_ffn_teacher(
    gate_w: np.ndarray | None,
    up_w: np.ndarray | None,
    down_w: np.ndarray | None,
    n_samples: int,
    seed: int,
    input_dim: int,
    intermediate_dim: int,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) using real or random teacher FFN weights."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, input_dim), dtype=np.float32)
    y = np.empty((n_samples, input_dim), dtype=np.float32)

    if device == "cuda":
        import torch

        dev = torch.device("cuda")
        if gate_w is None or up_w is None or down_w is None:
            gen = torch.Generator(device=dev)
            gen.manual_seed(seed)
            gate_t = torch.randn(
                intermediate_dim, input_dim, generator=gen, device=dev, dtype=torch.float32
            ) * 0.02
            up_t = torch.randn(
                intermediate_dim, input_dim, generator=gen, device=dev, dtype=torch.float32
            ) * 0.02
            down_t = torch.randn(
                input_dim, intermediate_dim, generator=gen, device=dev, dtype=torch.float32
            ) * 0.02
        else:
            gate_t = torch.from_numpy(gate_w).to(dev)
            up_t = torch.from_numpy(up_w).to(dev)
            down_t = torch.from_numpy(down_w).to(dev)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_t = torch.from_numpy(x[start:end]).to(dev)
            y[start:end] = ffn_forward_torch(x_t, gate_t, up_t, down_t).cpu().numpy()
        return x, y

    if gate_w is None or up_w is None or down_w is None:
        gate_w = rng.standard_normal((intermediate_dim, input_dim), dtype=np.float32) * 0.02
        up_w = rng.standard_normal((intermediate_dim, input_dim), dtype=np.float32) * 0.02
        down_w = rng.standard_normal((input_dim, intermediate_dim), dtype=np.float32) * 0.02

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        y[start:end] = ffn_forward_numpy(x[start:end], gate_w, up_w, down_w)
    return x, y


def write_metadata(
    path: Path,
    ref: LayerRef,
    n_samples: int,
    seed: int,
) -> None:
    meta = {
        **asdict(ref),
        "n_samples": n_samples,
        "seed": seed,
    }
    path.write_text(json.dumps(meta, indent=2) + "\n")


def apply_env_defaults(env: LayerExtractEnv, parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        output=None,
        samples=env.samples,
        seed=env.seed,
        model=env.model,
        synthetic=env.synthetic,
        layer=env.layer,
        data_dir=env.data_dir,
        batch_size=env.extract_batch_size,
    )


def main() -> None:
    env = load_layer_extract_env()
    device = resolve_teacher_device()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-n", "--samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--data-dir", type=Path, dest="data_dir")
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    apply_env_defaults(env, parser)
    args = parser.parse_args()

    gate_w: np.ndarray | None = None
    up_w: np.ndarray | None = None
    down_w: np.ndarray | None = None
    hidden = env.hidden_size
    intermediate = env.intermediate_size
    model_name = "synthetic" if args.synthetic else args.model

    if not args.synthetic:
        gate_w, up_w, down_w, hidden, intermediate, _ = load_qwen_ffn_weights(
            args.model, args.layer, device
        )

    ref = LayerRef(
        model=model_name,
        layer=args.layer,
        hidden_size=hidden,
        intermediate_size=intermediate,
        target_kind="ffn_out",
    )

    out = args.output or args.data_dir / ref.filename()
    x, y = synthetic_ffn_teacher(
        gate_w,
        up_w,
        down_w,
        args.samples,
        args.seed,
        hidden,
        intermediate,
        device,
        args.batch_size,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    write_layer_dataset(out, x, y)
    write_metadata(out.with_suffix(".json"), ref, args.samples, args.seed)

    print(
        f"wrote {out}\n"
        f"  model={model_name}  layer={args.layer}  target={ref.target_kind}\n"
        f"  samples={args.samples}  input_dim={hidden}  output_dim={hidden}\n"
        f"  device={device}"
    )


if __name__ == "__main__":
    main()
