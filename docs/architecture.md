# Architecture: ternary micro-network

Target model: **Qwen2.5-0.5B** — see [qwen2.5-0.5b.md](qwen2.5-0.5b.md).

**Product goal:** the assembled student must be **smaller than Qwen** (disk + RAM).
See [product-goals.md](product-goals.md).

For Qwen2.5-0.5B: `hidden = 896`, **116 736** scalar MLP chains (24 layers × 4864 neurons).

---

## Production target: `exact`

**`CALIBER158_ARCH=exact`** — teacher-shaped scalar SwiGLU, **no hidden bottleneck H**.

One chain approximates the same function as one Qwen MLP neuron path:

```
f(x) = SiLU(w_gate · x) · (w_up · x)
```

### Student (`exact`)

```
x [D=896]
  ├─ TernaryLinear(1 × D)  →  gate   (dot product → scalar)
  └─ TernaryLinear(1 × D)  →  up     (dot product → scalar)
  out = α · SiLU(gate) · up
```

| Item | Value |
|------|--------|
| Params / chain | **1 793** (2×896 + α) |
| Total @ 116k chains | **~209M** (~0.2–0.8 GB) — **below Qwen ~0.5B / ~1 GB** |
| `HIDDEN_DIM` | ignored (`hidden=exact` in logs) |
| Weights | FP32 shadow → `{-1,0,1}` via STE |
| Loss | MSE vs teacher `Y` |
| Optimizer | AdamW + STE |

### Implementation status

| Layer | `exact` |
|-------|---------|
| Torch (`python/student/model.py`) | ✅ train + holdout |
| Mojo (`BatchMicroNet`, GPU) | ❌ **not ported** — production blocker |
| Phase 1 ternary @ 100k | ❌ `rel≈1` (#100k-k/z) — STE blocker (same as v0) |

**R&D default:** Torch `make train-torch` with `CALIBER158_ARCH=exact`.

**Production default (target):** Mojo `make train-cuda` with `arch=exact` once ported.

---

## Legacy / R&D: v0 and v1 (bottleneck H)

v0/v1 use a **wider bottleneck** `H` (hidden micro-layer). Useful for infra bring-up and
historical experiments; **not** the production architecture — total size **exceeds Qwen**
at full chain count (e.g. v0 H=16 → **~3.35B** params).

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
| `H` | bottleneck width; **deprecated for production** |
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
