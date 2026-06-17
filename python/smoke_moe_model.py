#!/usr/bin/env python3
"""Smoke Qwen3.5/3.6 MoE support: config load + optional from_pretrained on GPU."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_config import _get, _get_bool, apply_huggingface_paths, resolve_teacher_device
from extract_moe_chain import load_moe_text_config
from moe_chain import ExpertKind, extract_moe_gate_up_rows, resolve_text_layers


def _require_transformers_version() -> None:
    import transformers

    version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if version < (5, 2):
        raise SystemExit(
            f"transformers {transformers.__version__} too old for Qwen3.5/3.6 MoE; need >=5.2.0"
        )
    print(f"transformers {transformers.__version__} ok")


def _smoke_config(model_name: str) -> None:
    try:
        hidden, intermediate, num_layers, num_experts = load_moe_text_config(model_name)
    except ValueError as exc:
        raise SystemExit(
            f"{exc}\n"
            f"Hint: CALIBER158_MODEL=Qwen/Qwen3.6-35B-A3B make smoke-moe-model"
        ) from exc
    print(
        f"config ok: D={hidden} I_moe={intermediate} L={num_layers} experts={num_experts}"
    )


def _smoke_load(model_name: str, device: str) -> None:
    from extract_moe_chain import load_moe_model

    layer = int(_get("CALIBER158_LAYER", "0"))
    expert_kind = ExpertKind.parse(_get("CALIBER158_EXPERT_KIND", "shared"))
    expert_id = int(_get("CALIBER158_EXPERT_ID", "0"))
    neuron = int(_get("CALIBER158_NEURON", "0"))
    _, intermediate, _, _ = load_moe_text_config(model_name)

    print(f"loading {model_name} on {device} …")
    model = load_moe_model(model_name, device)
    layers = resolve_text_layers(model)
    mlp = layers[layer].mlp
    gate_t, up_t = extract_moe_gate_up_rows(
        mlp, expert_kind, expert_id, neuron, intermediate
    )
    print(
        f"weight smoke ok: layer={layer} kind={expert_kind.value} "
        f"expert_id={expert_id} neuron={neuron} gate={tuple(gate_t.shape)} up={tuple(up_t.shape)}"
    )
    del model


def main() -> None:
    apply_huggingface_paths()
    _require_transformers_version()
    model_name = _get("CALIBER158_MODEL", "Qwen/Qwen3.6-35B-A3B")
    _smoke_config(model_name)
    if _get_bool("CALIBER158_MOE_SMOKE_LOAD", False):
        device = resolve_teacher_device()
        _smoke_load(model_name, device)
    else:
        print("skip from_pretrained (set CALIBER158_MOE_SMOKE_LOAD=1 to load weights)")


if __name__ == "__main__":
    main()
