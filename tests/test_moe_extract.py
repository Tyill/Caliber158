"""MoE extract helpers: chain identity, weight paths, synthetic roundtrip."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from extract_chain import swiglu_chain_numpy, write_dataset
from extract_moe_chain import load_moe_chain_weights, write_metadata
from moe_chain import (
    ExpertKind,
    MoeChainRef,
    extract_moe_gate_up_rows,
    moe_weight_source,
    validate_moe_chain_ref,
)
from student.dataset import read_dataset


class _FakeSharedExpert(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)


class _FakeMoEExperts(nn.Module):
    def __init__(self, num_experts: int, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate, hidden, dtype=torch.float32)
        )


class _FakeMoEBlock(nn.Module):
    def __init__(self, num_experts: int, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.experts = _FakeMoEExperts(num_experts, hidden, intermediate)
        self.shared_expert = _FakeSharedExpert(hidden, intermediate)


def test_moe_chain_id_routed_and_shared() -> None:
    ref_r = MoeChainRef(
        model="test",
        layer=0,
        expert_kind=ExpertKind.ROUTED,
        expert_id=0,
        neuron=0,
        hidden_size=2048,
        moe_intermediate_size=512,
        num_experts=256,
    )
    ref_s = MoeChainRef(
        model="test",
        layer=0,
        expert_kind=ExpertKind.SHARED,
        expert_id=0,
        neuron=0,
        hidden_size=2048,
        moe_intermediate_size=512,
        num_experts=256,
    )
    assert ref_r.chain_id == 0
    assert ref_s.chain_id == 256 * 512
    assert ref_r.filename() == "L00_E000_N000.bin"
    assert ref_s.filename() == "L00_S_N000.bin"


def test_moe_weight_source_documents_fused_routed() -> None:
    ref = MoeChainRef(
        model="test",
        layer=1,
        expert_kind=ExpertKind.ROUTED,
        expert_id=7,
        neuron=3,
        hidden_size=2048,
        moe_intermediate_size=512,
        num_experts=256,
    )
    source = moe_weight_source(ref)
    assert "gate_up_proj[7, 3" in source
    assert "gate_up_proj[7, 515" in source


def test_extract_moe_gate_up_rows_matches_swiglu() -> None:
    hidden, intermediate, num_experts = 16, 4, 3
    mlp = _FakeMoEBlock(num_experts, hidden, intermediate)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, hidden), dtype=np.float32)

    gate_t, up_t = extract_moe_gate_up_rows(
        mlp, ExpertKind.ROUTED, expert_id=2, neuron_idx=1, moe_intermediate_size=intermediate
    )
    y_ref = swiglu_chain_numpy(gate_t.detach().numpy(), up_t.detach().numpy(), x)

    gate_t, up_t = extract_moe_gate_up_rows(
        mlp, ExpertKind.SHARED, expert_id=0, neuron_idx=2, moe_intermediate_size=intermediate
    )
    y_shared = swiglu_chain_numpy(gate_t.detach().numpy(), up_t.detach().numpy(), x)
    assert y_ref.shape == (8,)
    assert y_shared.shape == (8,)


def test_moe_extract_synthetic_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hidden, intermediate, num_experts, num_layers = 32, 8, 4, 2
    mlp = _FakeMoEBlock(num_experts, hidden, intermediate)
    fake_layers = SimpleNamespace(mlp=mlp)
    fake_model = SimpleNamespace(model=SimpleNamespace(layers=[fake_layers]))

    def fake_load_model(_model_name: str, _device: str):
        return fake_model

    def fake_load_config(_model_name: str):
        return hidden, intermediate, num_layers, num_experts

    monkeypatch.setattr("extract_moe_chain.load_moe_model", fake_load_model)
    monkeypatch.setattr("extract_moe_chain.load_moe_text_config", fake_load_config)

    ref = MoeChainRef(
        model="synthetic-moe",
        layer=0,
        expert_kind=ExpertKind.ROUTED,
        expert_id=1,
        neuron=2,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        num_experts=num_experts,
    )
    gate_w, up_w, dim = load_moe_chain_weights("unused", ref, "cpu")
    assert dim == hidden
    assert gate_w.shape == (hidden,)
    assert up_w.shape == (hidden,)

    n, seed = 64, 7
    x = np.random.default_rng(seed).standard_normal((n, hidden), dtype=np.float32)
    y = swiglu_chain_numpy(gate_w, up_w, x)
    out = tmp_path / ref.filename()
    write_dataset(out, x, y, hidden)
    write_metadata(out.with_suffix(".json"), ref, n, seed)

    loaded = read_dataset(out)
    meta = json.loads(out.with_suffix(".json").read_text())
    np.testing.assert_allclose(loaded.x, x)
    np.testing.assert_allclose(loaded.y, y)
    assert meta["target_kind"] == "moe_routed_ffn"
    assert meta["chain_id"] == ref.chain_id


def test_validate_shared_expert_id() -> None:
    ref = MoeChainRef(
        model="test",
        layer=0,
        expert_kind=ExpertKind.SHARED,
        expert_id=1,
        neuron=0,
        hidden_size=2048,
        moe_intermediate_size=512,
        num_experts=256,
    )
    with pytest.raises(ValueError, match="shared expert must use expert_id=0"):
        validate_moe_chain_ref(ref, num_layers=40)
