# Target: Qwen3.6-35B-A3B

**Status:** **Phase 1 (MoE extract pilot)** — Phase 0 gate **closed (waived)**. See [HANDOFF.md](../HANDOFF.md) § pilot.
MoE + hybrid (Gated DeltaNet + full attention). Text LM path only; vision encoder out of scope v1.

## Production arch (project-wide, 2026-06-17)

**Chosen:** `CALIBER158_ARCH=v0`, `CALIBER158_HIDDEN_DIM=1`, `CALIBER158_CHAIN_GROUP=1` (H=1 solo).

| | H=1 solo | K=16 H=26 (rejected) |
|--|----------|----------------------|
| 0.5B total | **~209M** ✅ | ~343M |
| 35B-A3B active assembled | **~2.6B** ✅ | ~3.1B ❌ (> 3B active) |
| Phase 1 @ 0.5B | ✅ per-chain | ✅ 15/16 groups (N0011 edge) |

**R&D only:** `K=16 H=26` — chain-group experiments on 0.5B (`make extract-group`); not deployed.

## Why this model

| Criterion | Qwen3.6-35B-A3B | Qwen3-Coder-480B-A35B | Qwen3.6-27B dense |
|-----------|-----------------|------------------------|-------------------|
| Teacher active | ~3B | ~35B | ~27B |
| H=1 assembled active (est.) | **~2.6B / ~5 GB bf16** | ~27B / ~54 GB | ~21B / ~43 GB |
| H=1 full student MLP | **~22B < 35B total** | ~312B | ~11B |
| MoE complexity | light (I=512, D=2048) | heavy (I=2560, D=6144) | none |

Best MoE sizing trade-off in the Qwen3.6 line for Caliber158: small experts, active student
beats **3B active** and full student beats **35B total** at H=1 solo.

## Model geometry (HF `text_config`)

Source: `Qwen/Qwen3.6-35B-A3B` → `config.json` → `text_config`.

| Parameter | Value |
|-----------|-------|
| `hidden_size` D | **2048** |
| `num_hidden_layers` L | **40** |
| `num_experts` | **256** |
| `num_experts_per_tok` | **8** (routed) |
| `moe_intermediate_size` I_moe | **512** |
| `shared_expert_intermediate_size` | **512** (always active) |
| `full_attention_interval` | **4** (hybrid stack) |
| Total params | **~35B** |
| Active params / token | **~3B** (8 routed + 1 shared expert) |

One **scalar chain** (unit of work, same semantics as 0.5B):

```
f(x) = SiLU(w_gate · x) · (w_up · x),   x ∈ ℝ²⁰⁴⁸
```

Applied per **(layer, expert, neuron)** on MoE FFN weights (`gate_proj` / `up_proj` rows).
Shared expert is a separate FFN, not folded into the 256 routed experts.

### Out of scope (v1)

- **Gated DeltaNet** / `linear_attention` blocks (~48/40 layers) — separate chain family later.
- **Full attention** Q/K/V/O — separate chain family later.
- **Vision encoder** (`vision_config`) — multimodal path later.
- **Router** weights — keep teacher router at inference v1; record traces for extract only.

## Chain count

Per layer:

| Scope | Formula | Count |
|-------|---------|------:|
| Routed (one expert) | I_moe | 512 |
| Routed (all experts) | num_experts × I_moe | 131 072 |
| Shared expert | I_moe | 512 |
| **Layer total (full)** | (256 + 1) × 512 | **131 584** |

Global:

| Scope | Count |
|-------|------:|
| **Full distil** (all routed + shared, 40 layers) | **5 263 360** |
| **Active footprint** (8 routed + 1 shared / layer) | **184 320** |
| Active / full | **1/28.5** |

Compare 0.5B: **116 736** chains. This target is **~45×** full / **~1.6×** active-only vs 0.5B chain count.

## Chain identity

### Logical key

```
(layer, expert_kind, expert_id, neuron)
```

| Field | Routed | Shared |
|-------|--------|--------|
| `expert_kind` | `routed` | `shared` |
| `expert_id` | `0 … 255` | `0` (fixed) |
| `neuron` | `0 … 511` | `0 … 511` |

### Flat `chain_id` (dense index, full corpus)

```text
ROUTED_CHAINS_PER_LAYER = num_experts × I_moe          # 131_072
SHARED_CHAINS_PER_LAYER   = I_moe                        # 512
CHAINS_PER_LAYER          = ROUTED_CHAINS_PER_LAYER
                          + SHARED_CHAINS_PER_LAYER      # 131_584

# routed
chain_id = layer × CHAINS_PER_LAYER
         + expert_id × I_moe
         + neuron

# shared
chain_id = layer × CHAINS_PER_LAYER
         + ROUTED_CHAINS_PER_LAYER
         + neuron
```

Range: `0 … 5_263_359`.

### Filename convention (proposed)

```text
L{layer:02d}_E{expert:03d}_N{neuron:03d}.bin   # routed
L{layer:02d}_S_N{neuron:03d}.bin                 # shared
```

Legacy 0.5B pattern `L{layer}_N{neuron}` is **ambiguous** for MoE — do not reuse without `E`/`S`.

### Chain-group (K=16)

`I_moe = 512`, `K = 16` → **32 groups per expert** (`512 / 16`).
`base_neuron` must align to multiples of 16 within the same `(layer, expert_id)`.

## Student size budget

Formulas (v0 shared-bottleneck, same as 0.5B R&D):

```text
params/group = 2·H·D + K·(H+1)
total        = (chains / K) × params/group     # when using chain groups
total        = chains × (2·H·D + H + 1)        # H=1 solo, K=1
```

D = 2048.

| Student | Chains scope | Total params | vs 35B total | vs 3B active |
|---------|--------------|-------------:|-------------:|-------------:|
| **H=1 solo** | full | **~21.6B** | **0.62× ✅** | — |
| **H=1 solo** | active | **~0.76B** | — | **0.25× ✅** |
| **K=16 H=26** | full | ~35.2B | 1.01× ❌ | — |
| **K=16 H=26** | active | ~1.23B | — | 0.41× MLP only |

**Assembled inference (H=1, active path, est.):**

```text
student MLP (active)     ~0.76B
+ attn / DeltaNet / embed ~1.9B   # teacher active minus active MoE FFN
≈ 2.6B total                      ~5 GB bf16 weights
```

**Production default for this target:** `arch=v0`, **`H=1` solo**, `CHAIN_GROUP=1` — same as 0.5B.  
`K=16 H=26` **rejected:** 35B-A3B active assembled **~3.1B > 3B** — fails RAM goal.

| | H=1 active assembled | Teacher 3B active |
|--|---------------------:|------------------:|
| bf16 weights | **~5 GB** | ~6 GB |
| int8 | ~3 GB | ~3 GB |

## Distil scope: full vs active path

| Mode | When | Chains | Disk (H=1 bf16 MLP) |
|------|------|-------:|--------------------:|
| **Active path** | default inference target | 184k | **~1.5 GB** |
| **Full** | complete checkpoint replacement | 5.26M | **~43 GB** |

**Active path** = distil/train chains for **8 routed + 1 shared** experts per layer, load
at inference via **MoE router** (same as teacher): only active expert students in RAM.

**Full path** still fits under **35B total** at H=1 (~22B MLP only) — viable if we want
all experts covered without cold-expert fallback.

### Cold experts (active-path mode)

If an expert rarely fires in the extract corpus:

- **Do not** silently skip — log and mark `coverage=cold` in sidecar JSON.
- Inference v1 options (pick explicitly in implementation): teacher FFN fallback, shared-only
  blend, or reject token (hard fail). No magic default.

## Extract contract (proposed extensions)

Extends [extract_chain.py](../python/extract_chain.py) sidecar / `.bin` metadata.

### Required sidecar fields

```json
{
  "model": "Qwen/Qwen3.6-35B-A3B",
  "target_kind": "moe_routed_ffn",
  "layer": 0,
  "expert_kind": "routed",
  "expert_id": 0,
  "neuron": 0,
  "hidden_size": 2048,
  "moe_intermediate_size": 512,
  "chain_id": 0,
  "samples": 100000,
  "seed": 42
}
```

`target_kind`: `moe_routed_ffn` | `moe_shared_ffn`.

### Weight source (teacher)

Verified against `transformers>=5.2.0` (`Qwen3_5MoeSparseMoeBlock` in
`modeling_qwen3_5_moe.py`). Text decoder path:
`model.model.language_model.layers[L]` for `Qwen3_5MoeForConditionalGeneration`.

Routed experts use **fused** `gate_up_proj` — not separate `gate_proj` / `up_proj`:

```text
# routed expert e, neuron n, layer L
gate_w = layers[L].mlp.experts.gate_up_proj[e, n, :]
up_w   = layers[L].mlp.experts.gate_up_proj[e, I_moe + n, :]

# shared expert (separate MLP)
gate_w = layers[L].mlp.shared_expert.gate_proj.weight[n]
up_w   = layers[L].mlp.shared_expert.up_proj.weight[n]
```

Scalar target (same as 0.5B): `SiLU(gate_w · x) * (up_w · x)` — pre-`down_proj`,
without `shared_expert_gate` scaling.

Legacy proposal with per-expert `gate_proj` — **incorrect** for this architecture;
kept here only as a warning not to use:

```text
# WRONG for Qwen3.5/3.6 MoE:
# model.model.layers[L].mlp.experts[e].gate_proj.weight[n]
```

### Router traces (active-path sampling)

Optional sidecar per extract batch (not per chain):

```json
{
  "router_trace": {
    "layer": 0,
    "topk_experts": [3, 17, 42, 88, 120, 155, 201, 240],
    "token_count": 8192,
    "source": "calibration_run"
  }
}
```

Used to prioritize which routed experts get datasets first; not required for shared expert
(always active).

### Env / config

| Variable | Default | Meaning |
|----------|---------|---------|
| `CALIBER158_MODEL` | `Qwen/Qwen3.6-35B-A3B` | HF hub id |
| `CALIBER158_HIDDEN_SIZE` | `2048` | D |
| `CALIBER158_MOE_INTERMEDIATE_SIZE` | `512` | I_moe |
| `CALIBER158_NUM_EXPERTS` | `256` | routed count |
| `CALIBER158_NUM_LAYERS` | `40` | L |
| `CALIBER158_EXPERT_KIND` | `routed` | `routed` \| `shared` |
| `CALIBER158_EXPERT_ID` | `0` | `0 … 255` if routed; must be `0` if shared |
| `CALIBER158_MOE_SMOKE_LOAD` | `0` | `1` = `make smoke-moe-model` loads full weights |
| (reuse) | | `SAMPLES`, `SEED`, `LAYER`, `NEURON`, `DATA_DIR`, `TORCH` |

Commands:

```bash
make test-moe-extract          # synthetic roundtrip (CI-safe)
make smoke-moe-model           # config only
CALIBER158_MOE_SMOKE_LOAD=1 make smoke-moe-model   # GPU weight path smoke

CALIBER158_MODEL=Qwen/Qwen3.6-35B-A3B \
CALIBER158_LAYER=0 CALIBER158_EXPERT_KIND=shared CALIBER158_NEURON=0 \
make extract-moe
```

## Phase 1 success criteria

Same bar as 0.5B unless superseded:

```text
rel_holdout = RMSE / std(Y_holdout) < 0.001
```

Pilot: **FP32 shadow** (`CALIBER158_QUANTIZE=0`), **100k samples**, layer 0.

Order:

1. **Shared expert** — `L00`, `S`, `N0000` (always active, simplest path).
2. **One routed expert** — `L00`, `E000`, `N0000`.
3. **Active octet** — 8 routed ids from one router trace on layer 0 + shared.
4. Scale **layer → all layers → full/full+active** per scope decision.

## Rollout plan

### Phase 0 — gate ✅ closed (2026-06-17, waived)

**Decision locked:** production = **v0 H=1 solo** (`lerning_compare.md` § «Production arch»).

Formal re-run @ `HIDDEN_DIM=1` on `L00_N0000` + `N0001` **не делаем** — достаточно FP32 v0 Phase 1 @ H=16–128 (H=1 ⊂ тот же family) + size math для 35B-A3B. Quality check переносим на **MoE pilot P2** (shared `N0000`).

**Next:** § Phase 1 below.

### Phase 1 — MoE extract pilot (layer 0)

- Implement MoE weight load + new sidecar fields.
- Extract + train shared `N0000`, then routed `E000/N0000`.
- Verify `rel_holdout` and size estimate script for 184k / 5.26M chains.

### Phase 2 — active footprint layer 0

- All **512 × 9 = 4608** chains for layer 0 (8 routed ids from calibration + shared).
- Parallel workers; target same ops model as 0.5B Phase 2.

### Phase 3 — all 40 layers (active scope)

- **184 320** chain jobs.
- Assemble MoE FFN student + teacher router + teacher non-FFN blocks.
- End-to-end inference stub: replace expert `gate/up` scalar paths only.

### Phase 4 — full expert coverage (optional)

- Remaining routed experts (256 per layer).
- **5.26M** jobs; disk ~43 GB bf16 (H=1 MLP student only).
- Required only if cold-expert fallback is unacceptable.

### Phase 5 — non-FFN chain families (future)

- Gated DeltaNet linear-attention blocks.
- Full-attention projections.
- Only after MoE FFN path is proven.

## HuggingFace IDs

- Primary: [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- Instruct / coding variants: evaluate after base pilot (same text config expected).

## Open decisions

1. **Active-only vs full** for first assembled inference demo — recommend **active** (RAM ~5 GB).
2. **Cold expert policy** at inference — needs explicit choice before Phase 3.
3. **Ternary STE** — blocked on 0.5B (~`rel ≈ 1`); FP32/ bf16 student first on this target.
4. **`transformers` version** — config uses `qwen3_5_moe`; pin min version in pilot PR.

## See also

- [Qwen2.5-0.5B](qwen2.5-0.5b.md) — current dev target
- [product-goals.md](product-goals.md) — student must be smaller than teacher
- [architecture.md](architecture.md) — student arch (`v0`, chain groups)
