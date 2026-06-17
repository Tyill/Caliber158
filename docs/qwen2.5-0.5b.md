# Target: Qwen2.5-0.5B

First full-model approximation target before larger Qwen variants.

**Next target (after Phase 1 green):** [Qwen3.6-35B-A3B](qwen3.6-35b-a3b.md).

## Model geometry (from HF config)

| Parameter | Value |
|-----------|-------|
| `hidden_size` | **896** |
| `intermediate_size` | **4864** |
| `num_hidden_layers` | **24** |
| Total params | ~494M |

One MLP scalar chain (our unit of work):

```
f(x) = SiLU(w_gate · x) · (w_up · x),   x ∈ ℝ⁸⁹⁶
```

## Chain count

| Scope | Count |
|-------|-------|
| Per transformer layer | 4864 |
| All MLP chains (24 layers) | **116 736** |

Each chain is independent: random `X ∈ ℝⁿˣ⁸⁹⁶` → teacher `Y` → train ternary `MicroNet` + α.

Chain ID: `layer * 4864 + neuron` (0 … 116735).

## Rollout plan

### Phase 1 — one chain (now)

Configure `.env` (copy from `.env.example`), then:

```bash
pixi run extract
pixi run train
```

Success: holdout MSE < 1e-4 (or < 0.1% of `Var(Y)`).

### Phase 2 — one full MLP block (layer 0, 4864 chains)

- Batch-generate datasets for all neurons in layer 0.
- Parallel Mojo workers (one process per chain or per GPU batch).
- Validate layer-0 MLP replacement end-to-end.

### Phase 3 — all 24 layers

- 116 736 chain fits (embarrassingly parallel).
- Assemble ternary MLP weights + α per layer.
- Wire into inference stub (replace `gate_proj` / `up_proj` paths).

### Phase 4 — attention + rest

MLP is ~2/3 of FFN path; attention projections are separate chain families (future).

## Notes vs larger Qwen

`docs/target.md` mentions ~530k chains and dim 3584 — that matches a **larger** Qwen (e.g. hidden 3584).
For **0.5B** use **896** input dim and **116k** MLP chains; same algorithm, smaller scale.

## HuggingFace IDs

- Base: `Qwen/Qwen2.5-0.5B`
- Instruct (optional later): `Qwen/Qwen2.5-0.5B-Instruct`
