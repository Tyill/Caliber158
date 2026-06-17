# Product goals

## Primary goal: student smaller than Qwen

The deployed Caliber158 student must use **less storage and less inference RAM** than the
teacher it replaces, at the agreed quality bar (Phase 1 per chain, then full assembly).

**Baseline (teacher):** [Qwen2.5-0.5B](qwen2.5-0.5b.md) — ~**0.5B** shared parameters,
~**1 GB** on disk (typical bf16 checkpoint).

The student is **not** successful if it only distills well but ends up **larger** than Qwen.

### How to measure

Count **total** student parameters across all chains, not per-chain params in isolation:

```
total_params = num_chains × params_per_chain
```

For Qwen2.5-0.5B: **116 736** scalar MLP chains (24 layers × 4864 intermediate neurons).

| Student arch | params / chain | total params | vs Qwen (~0.5B) |
|--------------|----------------|--------------|-----------------|
| `exact` (teacher-shaped) | 1 793 | ~**209M** | **smaller** |
| v0 `H=16` | 28 689 | ~**3.35B** | **~6× larger** |
| v0 `H=128` | 229 505 | ~**26.8B** | much larger |

Weight formats (order-of-magnitude inference storage):

| | Qwen 0.5B | exact @ 116k | v0 H=16 @ 116k |
|--|-----------|--------------|----------------|
| bf16 / fp32-like | ~1 GB | ~0.8 GB | ~13 GB |
| int8 | — | ~0.2 GB | ~3.2 GB |
| 2-bit packed ternary | — | ~50 MB | ~0.8 GB |

Even aggressive ternary packing does not fix **v0 + bottleneck H** if total param count
stays ~3B. **Architecture choice is a size decision**, not only a training convenience.

### Implications

1. **Per-chain distillation** (one micro-net per scalar chain) multiplies parameters by
   `num_chains`. Shared teacher matrices must not be replicated 116k times without a
   hard size budget check.

2. **`arch=exact`** (~1.8k / chain) aligns with the size goal; **v0 + H** (tens of k /
   chain) does **not**, unless `H` or chain count is reduced dramatically or weights are
   shared across chains.

3. Experiments and proposals must include a **total size estimate** (params and MB) vs
   Qwen before expanding H, depth, or chain coverage.

4. Phase 1 quality (`rel_holdout < 0.001` on one chain @ 100k) remains required, but
   **does not override** the smaller-than-Qwen goal for the final assembled model.

### Current status (2026-06-17)

- Quality path explored: FP32 v0 small `H`, ternary blocked by STE (~`rel ≈ 1`).
- **Production arch:** **v0 H=1 solo** (~209M @ 116k) — size ✅ on 0.5B and 35B-A3B active path.
- Legacy: v0 H=16 per-chain (~3.35B) and K=16 H=26 (~343M) — **not production** (size or next-target fail).

See also: [architecture.md](architecture.md), [HANDOFF.md](../HANDOFF.md),
[lerning_compare.md](../lerning_compare.md).
