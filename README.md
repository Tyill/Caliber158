# Caliber158

Approximate individual Qwen MLP scalar chains (`3584 → SwiGLU → 1`) with independent
1.58-bit ternary micro-networks. Each of ~530k chains is trained in isolation via
local distillation (random inputs → teacher output → MSE + STE).

See [docs/target.md](docs/target.md) for the problem statement,
[docs/architecture.md](docs/architecture.md) for the v0 network layout, and
[docs/qwen2.5-0.5b.md](docs/qwen2.5-0.5b.md) for the first target model.

## Repository layout

```
Caliber158/
├── main.mojo                 # CLI entry point
├── src/                      # Mojo package
│   └── chain/
│       ├── ternary.mojo      # {-1,0,1} quantize + matvec
│       ├── micro_net.mojo    # student architecture + α scale
│       ├── dataset.mojo      # .bin loader + synthetic data
│       ├── train.mojo        # MSE + STE + AdamW
│       ├── grads.mojo        # gradient buffers
│       ├── adamw.mojo        # AdamW optimizer
│       └── rng.mojo          # deterministic LCG
├── python/
│   └── extract_chain.py      # teacher dataset generator
├── docs/
└── data/                     # generated .bin / checkpoints (gitignored)
```

## Prerequisites

- [pixi](https://pixi.sh/) for Mojo toolchain
- Python 3.10+ (`python3` on PATH, for teacher dataset generation)

## Setup

```bash
cp .env.example .env   # edit CALIBER158_* as needed
pixi install           # loads .env via scripts/load-env.sh on pixi run
pixi run setup-python  # project-local .venv (not global pip)
```

Teacher weights are cached under `models/huggingface/` (see `CALIBER158_MODELS_DIR` in `.env`), not in `~/.cache/huggingface`.

## Configuration

All tunables live in **`.env`** (see `.env.example`). Prefix: `CALIBER158_*`.

| Variable | Purpose |
|----------|---------|
| `CALIBER158_MODEL` | HuggingFace teacher id |
| `CALIBER158_HIDDEN_SIZE` | Teacher input dim (896 for 0.5B) |
| `CALIBER158_LAYER` / `CALIBER158_NEURON` | Which chain to extract |
| `CALIBER158_HIDDEN_DIM` | Student network width |
| `CALIBER158_LR`, `CALIBER158_EPOCHS`, … | Training hyperparams |

CLI flags override env for one-off runs.

## Quick start

Smoke test (uses env defaults):

```bash
pixi run smoke
```

Extract teacher dataset (downloads Qwen on first run; env-driven):

```bash
pixi run extract
# or: bash scripts/run-python.sh python/extract_chain.py
```

Train:

```bash
pixi run train
```

## Status

| Component | State |
|-----------|-------|
| Ternary forward + α | Implemented |
| Dataset I/O (`.bin`) | Implemented |
| MSE + STE backward | Implemented |
| AdamW optimizer | Implemented |
| SIMD matvec | **TODO** |
| Parallel 530k workers | **TODO** |

## License

See [LICENSE](LICENSE).
