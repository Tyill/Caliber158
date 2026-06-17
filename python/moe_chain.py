"""MoE scalar chain identity, weight paths, and chain_id for Qwen3.5/3.6 MoE."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class ExpertKind(str, Enum):
    ROUTED = "routed"
    SHARED = "shared"

    @classmethod
    def parse(cls, raw: str) -> ExpertKind:
        value = raw.strip().lower()
        if value not in {cls.ROUTED.value, cls.SHARED.value}:
            raise ValueError(f"expert_kind must be routed or shared, got {raw!r}")
        return cls(value)


@dataclass(frozen=True)
class MoeChainRef:
    model: str
    layer: int
    expert_kind: ExpertKind
    expert_id: int
    neuron: int
    hidden_size: int
    moe_intermediate_size: int
    num_experts: int

    @property
    def target_kind(self) -> str:
        if self.expert_kind == ExpertKind.ROUTED:
            return "moe_routed_ffn"
        return "moe_shared_ffn"

    @property
    def chains_per_layer(self) -> int:
        return self.num_experts * self.moe_intermediate_size + self.moe_intermediate_size

    @property
    def chain_id(self) -> int:
        routed_per_layer = self.num_experts * self.moe_intermediate_size
        layer_base = self.layer * self.chains_per_layer
        if self.expert_kind == ExpertKind.ROUTED:
            return layer_base + self.expert_id * self.moe_intermediate_size + self.neuron
        return layer_base + routed_per_layer + self.neuron

    def filename(self) -> str:
        if self.expert_kind == ExpertKind.SHARED:
            return f"L{self.layer:02d}_S_N{self.neuron:03d}.bin"
        return f"L{self.layer:02d}_E{self.expert_id:03d}_N{self.neuron:03d}.bin"


def routed_chains_per_layer(num_experts: int, moe_intermediate_size: int) -> int:
    return num_experts * moe_intermediate_size


def validate_moe_chain_ref(ref: MoeChainRef, num_layers: int) -> None:
    if not (0 <= ref.layer < num_layers):
        raise ValueError(f"layer must be in [0, {num_layers}), got {ref.layer}")
    if not (0 <= ref.neuron < ref.moe_intermediate_size):
        raise ValueError(
            f"neuron must be in [0, {ref.moe_intermediate_size}), got {ref.neuron}"
        )
    if ref.expert_kind == ExpertKind.ROUTED:
        if not (0 <= ref.expert_id < ref.num_experts):
            raise ValueError(
                f"expert_id must be in [0, {ref.num_experts}), got {ref.expert_id}"
            )
    elif ref.expert_id != 0:
        raise ValueError(f"shared expert must use expert_id=0, got {ref.expert_id}")


def resolve_text_layers(model: object) -> object:
    """Return the decoder layer list for dense or Qwen3.5/3.6 MoE causal models."""
    inner = getattr(model, "model", None)
    if inner is None:
        raise ValueError("model has no .model attribute")

    language_model = getattr(inner, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        return language_model.layers

    layers = getattr(inner, "layers", None)
    if layers is not None:
        return layers

    raise ValueError("could not resolve text decoder layers from model")


def extract_moe_gate_up_rows(
    mlp: object,
    expert_kind: ExpertKind,
    expert_id: int,
    neuron_idx: int,
    moe_intermediate_size: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Read gate/up rows for one scalar chain from a MoE FFN block."""
    if expert_kind == ExpertKind.SHARED:
        shared = getattr(mlp, "shared_expert", None)
        if shared is None:
            raise ValueError("mlp.shared_expert missing — not a Qwen3.5/3.6 MoE block")
        gate_w = shared.gate_proj.weight[neuron_idx]
        up_w = shared.up_proj.weight[neuron_idx]
        return gate_w, up_w

    experts = getattr(mlp, "experts", None)
    if experts is None:
        raise ValueError("mlp.experts missing — not a Qwen3.5/3.6 MoE block")
    gate_up = experts.gate_up_proj[expert_id]
    gate_w = gate_up[neuron_idx]
    up_w = gate_up[moe_intermediate_size + neuron_idx]
    return gate_w, up_w


def moe_weight_source(ref: MoeChainRef) -> str:
    """Human-readable teacher weight path (verified against transformers 5.2+)."""
    layer = ref.layer
    neuron = ref.neuron
    if ref.expert_kind == ExpertKind.SHARED:
        return (
            f"layers[{layer}].mlp.shared_expert.gate_proj.weight[{neuron}] + "
            f"layers[{layer}].mlp.shared_expert.up_proj.weight[{neuron}]"
        )
    expert = ref.expert_id
    i = ref.moe_intermediate_size
    return (
        f"layers[{layer}].mlp.experts.gate_up_proj[{expert}, {neuron}, :] + "
        f"layers[{layer}].mlp.experts.gate_up_proj[{expert}, {i + neuron}, :]"
    )
