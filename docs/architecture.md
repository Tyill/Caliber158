# Architecture: ternary micro-network (v0)

Target model: **Qwen2.5-0.5B** — see [qwen2.5-0.5b.md](qwen2.5-0.5b.md).

Approximate one Qwen MLP scalar chain `hidden → SwiGLU → 1` with a wider ternary
black-box network plus a single real-valued scale factor α.

For Qwen2.5-0.5B: `hidden = 896`, **116 736** chains total (24 layers × 4864 neurons).

## Original function (one neuron)

For hidden vector `x ∈ ℝ^hidden` (896 for 0.5B):

```
f(x) = SiLU(w_gate · x) · (w_up · x)
```

## Student network (v0)

```
x [hidden]   # 896 for Qwen2.5-0.5B
  ├─ TernaryLinear(hidden → H)  →  gate
  └─ TernaryLinear(hidden → H)  →  up
  hidden = SiLU(gate) ⊙ up
  TernaryLinear(H → 1)          →  y_tern
  out = α · y_tern
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `H` | 128 | Increase to 256/512 if MSE plateaus |
| `α` | 1.0 (FP32) | Learnable output scale |
| Weights | FP32 shadow | Quantized to `{-1, 0, 1}` on forward |
| Loss | MSE | vs `Y_qwen` from teacher |
| Optimizer | AdamW + STE | Straight-through on quantize |

## Dataset binary format (`.bin`)

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

## Training loop (per chain)

1. Python: sample random `X`, run teacher chain → `Y_qwen`, save `.bin`.
2. Mojo: load dataset, init `MicroNet`, AdamW + STE, minimize MSE.
3. Export ternary weights + α to `data/checkpoints/chain_<id>.bin`.

## Success criteria (v0)

- Holdout MSE < 1e-4 on 10k samples, or
- Relative error < 0.1% of `Var(Y_qwen)`.

## Future (v1b+)

- Parallel worker pool: one job per chain (116 736 for 0.5B).

---

## Student network (v1)

Second SwiGLU block (H→H) with residual on `h0`; head reads `h1 = h0 + SwiGLU2(h0)`.

```
x [D=896]
  ├─ TernaryLinear(D → H)  →  gate1
  └─ TernaryLinear(D → H)  →  up1
  h0 = SiLU(gate1) ⊙ up1
  ├─ TernaryLinear(H → H)  →  gate2
  └─ TernaryLinear(H → H)  →  up2
  h2 = SiLU(gate2) ⊙ up2
  h1 = h0 + h2
  TernaryLinear(H → 1)      →  y_tern
  out = α · y_tern
```

| vs v0 | v1 |
|-------|-----|
| 1× SwiGLU(D→H) | 2× SwiGLU: (D→H) + (H→H) |
| `h → head` | `h0 + SwiGLU2(h0) → head` |
| params @ H=512 | ~918k → ~1.44M (+2·H²) |

Select with `CALIBER158_ARCH=v1` (default `v0`). Block2 shadow weights are zero-init so training starts near v0 (`h1 ≈ h0`).

Same loss (MSE), STE, AdamW, and single α as v0. Linear skip from `x` is v1b only.
