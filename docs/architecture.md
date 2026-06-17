# Architecture: ternary micro-network

Target model: **Qwen2.5-0.5B** — see [qwen2.5-0.5b.md](qwen2.5-0.5b.md).
Next scale target: **Qwen3.6-35B-A3B** — see [qwen3.6-35b-a3b.md](qwen3.6-35b-a3b.md).

**Production arch (2026-06-17):** **`v0` H=1 solo** — `HIDDEN_DIM=1`, `CHAIN_GROUP=1`, ~**209M** @ 116k chains.
FP32 shadow until ternary STE unblocked. Details: `lerning_compare.md` § «Production arch».

**Product goal:** the assembled student must be **smaller than Qwen** (disk + RAM).
See [product-goals.md](product-goals.md).

For Qwen2.5-0.5B: `hidden = 896`, **116 736** scalar MLP chains (24 layers × 4864 neurons).

---

## Production target: `v0` H=1 solo

**`CALIBER158_ARCH=v0`**, **`CALIBER158_HIDDEN_DIM=1`**, **`CALIBER158_CHAIN_GROUP=1`**.

Minimal bottleneck micro-net per scalar chain — same total params as teacher-shaped `exact`
(~1.8k/chain, ~209M @ 116k) but **Phase 1 trainable** (FP32 shadow; ternary STE still blocked).

```
x [D] → gate/up (1×H linear) → SiLU· → head (H→1) → α
```

| Item | Value |
|------|--------|
| Params / chain | **1 794** (`2·H·D + H + 1`, H=1) |
| Total @ 116k | **~209M** — below Qwen ~0.5B |
| 35B-A3B active assembled | **~2.6B** — below 3B active |

**Train (gate + pilot):** Torch `make train-torch`, `QUANTIZE=0`, `LR_SCHEDULE=rel_decay`.

### Legacy candidate: `exact` (not production)

**`CALIBER158_ARCH=exact`** — teacher-shaped scalar SwiGLU, **no hidden bottleneck H**.
Total size OK (~209M) but **Phase 1 train fails** (rel≈0.19–0.23 @ 100k); closed for train.
See § below for `exact` spec (diagnostic / teacher-init only).

---

## Student v0 (bottleneck H)

v0 micro-net with bottleneck `H`. **Production: H=1.** Wider H and chain groups (K>1) —
R&D only; size fail @ 116k when H>1 without sharing, or @ 35B-A3B when K=16 H=26.

### Original teacher function (one neuron)

For hidden vector `x ∈ ℝ^D` (896 for 0.5B):

```
f(x) = SiLU(w_gate · x) · (w_up · x)
```

### Student v0

```
x [D]
  ├─ TernaryLinear(D → H)  →  gate
  └─ TernaryLinear(D → H)  →  up
  h0 = SiLU(gate) ⊙ up
  TernaryLinear(H → 1)     →  y_tern
  out = α · y_tern
```

| Parameter | Notes |
|-----------|--------|
| `H` | bottleneck width; **production = 1**; H>1 R&D only (size fail @ 116k) |
| Params @ H=16 | 28 689 / chain |
| Mojo | ✅ CPU + GPU (`CALIBER158_ARCH=v0`, default today) |

### Student v1

Second SwiGLU block (H→H) with residual on `h0`; head reads `h1 = h0 + SwiGLU2(h0)`.

```
x [D=896]
  ├─ TernaryLinear(D → H)  →  gate1
  └─ TernaryLinear(D → H)  →  up1
  h0 = SiLU(gate1) ⊙ up1
  ├─ TernaryLinear(H → H)  →  gate2
  └─ TernaryLinear(H → H)  →  up2
  h1 = h0 + SiLU(gate2) ⊙ up2
  TernaryLinear(H → 1)      →  y_tern
  out = α · y_tern
```

Select with `CALIBER158_ARCH=v1`. Block2 shadow weights are zero-init (`h1 ≈ h0` at start).

Torch-only R&D: `v1b` (linear skip) — negative, not for production.

---

## Shared: dataset, train loop, success criteria

### Dataset binary format (`.bin`)

Python (`extract_chain.py`) writes; Mojo (`dataset.mojo`) reads.

```
offset  size        field
0       6           magic = b"CAL158"
6       4           version (uint32 LE) = 1
10      4           n_samples (uint32 LE)
14      4           input_dim (uint32 LE)  # 896 for 0.5B
18      n*d*4       X float32[row-major, n × input_dim]
18+n*d*4 n*4        Y float32[n]
```

### Training loop (per chain)

1. Python: sample random `X`, run teacher chain → `Y`, save `.bin`.
2. Student: load dataset, init micro-net, AdamW + STE, minimize MSE.
3. Export ternary weights + α to checkpoint (format TBD).

### Phase 1 success (one chain, 100k samples)

- Holdout MSE < 1e-4, or
- `rel_holdout = MSE/Var(Y) < 0.001`

Applies to **`exact`** (production target) and legacy v0/v1.

### Future

- Parallel worker pool: one job per chain (116 736 for 0.5B).
- Mojo port of **`exact`** (CPU + GPU, tests, `make test`).
