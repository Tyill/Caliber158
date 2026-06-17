#!/usr/bin/env python3
"""Extract one Qwen3.5/3.6 MoE scalar chain and write a Caliber158 training dataset."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_config import MoeExtractEnv, load_moe_extract_env, resolve_teacher_device
from extract_chain import swiglu_chain_numpy, swiglu_chain_torch, synthetic_teacher, write_dataset
from moe_chain import (
    ExpertKind,
    MoeChainRef,
    extract_moe_gate_up_rows,
    moe_weight_source,
    resolve_text_layers,
    validate_moe_chain_ref,
)


def load_moe_text_config(model_name: str):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name)
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden = int(text_cfg.hidden_size)
    intermediate_raw = getattr(text_cfg, "moe_intermediate_size", None)
    if intermediate_raw is None:
        intermediate_raw = getattr(text_cfg, "intermediate_size", None)
    if intermediate_raw is None:
        raise ValueError(f"{model_name} config has no moe_intermediate_size")
    intermediate = int(intermediate_raw)
    num_layers = int(text_cfg.num_hidden_layers)
    num_experts = int(getattr(text_cfg, "num_experts", 0))
    if num_experts <= 0:
        raise ValueError(f"{model_name} is not a MoE model (num_experts={num_experts})")
    return hidden, intermediate, num_layers, num_experts


def load_moe_model(model_name: str, device: str):
    import torch
    from transformers import AutoConfig, Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration

    cfg = AutoConfig.from_pretrained(model_name)
    architectures = getattr(cfg, "architectures", None) or []
    if "Qwen3_5MoeForConditionalGeneration" in architectures:
        model_cls = Qwen3_5MoeForConditionalGeneration
    else:
        model_cls = Qwen3_5MoeForCausalLM

    model = model_cls.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    if device == "cuda":
        model = model.to("cuda")
    return model


def load_moe_chain_weights(
    model_name: str,
    ref: MoeChainRef,
    device: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    import torch

    hidden, intermediate, num_layers, num_experts = load_moe_text_config(model_name)
    if ref.hidden_size != hidden:
        raise ValueError(f"hidden_size mismatch: env {ref.hidden_size}, config {hidden}")
    if ref.moe_intermediate_size != intermediate:
        raise ValueError(
            f"moe_intermediate_size mismatch: env {ref.moe_intermediate_size}, config {intermediate}"
        )
    if ref.num_experts != num_experts:
        raise ValueError(f"num_experts mismatch: env {ref.num_experts}, config {num_experts}")
    validate_moe_chain_ref(ref, num_layers)

    model = load_moe_model(model_name, device)
    layers = resolve_text_layers(model)
    mlp = layers[ref.layer].mlp
    gate_t, up_t = extract_moe_gate_up_rows(
        mlp,
        ref.expert_kind,
        ref.expert_id,
        ref.neuron,
        ref.moe_intermediate_size,
    )
    gate_w = gate_t.detach().float().cpu().numpy()
    up_w = up_t.detach().float().cpu().numpy()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return gate_w.astype(np.float32), up_w.astype(np.float32), hidden


def write_metadata(path: Path, ref: MoeChainRef, n_samples: int, seed: int) -> None:
    meta = {
        **asdict(ref),
        "expert_kind": ref.expert_kind.value,
        "target_kind": ref.target_kind,
        "chain_id": ref.chain_id,
        "n_samples": n_samples,
        "seed": seed,
        "weight_source": moe_weight_source(ref),
    }
    path.write_text(json.dumps(meta, indent=2) + "\n")


def apply_env_defaults(env: MoeExtractEnv, parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        output=None,
        samples=env.samples,
        seed=env.seed,
        model=env.model,
        synthetic=env.synthetic,
        layer=env.layer,
        neuron=env.neuron,
        expert_kind=env.expert_kind.value,
        expert_id=env.expert_id,
        input_dim=env.hidden_size,
        data_dir=env.data_dir,
    )


def main() -> None:
    env = load_moe_extract_env()
    device = resolve_teacher_device()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-n", "--samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--neuron", type=int)
    parser.add_argument(
        "--expert-kind",
        choices=[ExpertKind.ROUTED.value, ExpertKind.SHARED.value],
        dest="expert_kind",
    )
    parser.add_argument("--expert-id", type=int, dest="expert_id")
    parser.add_argument("--input-dim", type=int, dest="input_dim")
    parser.add_argument("--data-dir", type=Path, dest="data_dir")
    apply_env_defaults(env, parser)
    args = parser.parse_args()

    expert_kind = ExpertKind.parse(args.expert_kind)
    model_name = "synthetic" if args.synthetic else args.model
    ref = MoeChainRef(
        model=model_name,
        layer=args.layer,
        expert_kind=expert_kind,
        expert_id=args.expert_id,
        neuron=args.neuron,
        hidden_size=args.input_dim,
        moe_intermediate_size=env.moe_intermediate_size,
        num_experts=env.num_experts,
    )
    validate_moe_chain_ref(ref, env.num_layers)

    gate_w: np.ndarray | None = None
    up_w: np.ndarray | None = None
    hidden = args.input_dim

    if not args.synthetic:
        gate_w, up_w, hidden = load_moe_chain_weights(args.model, ref, device)

    out = args.output or args.data_dir / ref.filename()
    x, y = synthetic_teacher(gate_w, up_w, args.samples, args.seed, hidden, device)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_dataset(out, x, y, hidden)
    write_metadata(out.with_suffix(".json"), ref, args.samples, args.seed)

    print(
        f"wrote {out}\n"
        f"  model={model_name}  layer={args.layer}  expert_kind={expert_kind.value}  "
        f"expert_id={args.expert_id}  neuron={args.neuron}\n"
        f"  target_kind={ref.target_kind}  chain_id={ref.chain_id}  samples={args.samples}  "
        f"input_dim={hidden}\n"
        f"  device={device}"
    )


if __name__ == "__main__":
    main()
