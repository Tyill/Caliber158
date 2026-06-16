# Handoff для следующего чата

Обновлено: 2026-06-17 (**pivot → arch v2 shared FFN**; v1 per-chain `exact` ❌ @ Phase 1 ternary)

> **План v1:** per-chain micro-net — `.cursor/plans/architecture_v1_swiglu_7baef79e.plan.md`, `docs/architecture.md` § legacy. **Frozen** (infra остаётся, не production path).
>
> **План v2:** **shared weights, layer-wise FFN** — спека ниже § «Architecture v2». Детальный implementation plan — TBD (`.cursor/plans/` или `docs/architecture-v2.md`).

## Идея проекта

Аппроксимировать Qwen **не прямой заменой весов**, а **дистилляцией** в тернарные `{-1,0,1}` + FP32 scale(s).

**Teacher:** Qwen2.5-0.5B — `hidden=896`, `intermediate=4864`, **24** слоя.

| Поколение | Unit of distill | Статус |
|-----------|-----------------|--------|
| **v1 (per-chain)** | 116 736 scalar chain → micro-net / chain | ❌ ternary Phase 1; legacy infra |
| **v2 (shared layer)** | **1 FFN block / layer** — shared matrices | ✅ **production target** (2026-06-17) |

Подробнее: `docs/target.md`, `docs/product-goals.md`, `docs/qwen2.5-0.5b.md`, `lerning_compare.md` § session 2.

### v1 per-chain — итог (закрыто для production)

**Было (2026-06-17 утро):** production = **`arch=exact`** (1793 params/chain, ~209M total).

**Session 2 (2026-06-17):** ternary `exact` **не тянет** Phase 1:

| Метод | holdout rel @ 100k L00_N0000 |
|-------|------------------------------|
| STE + Adam (best, grad_clip=1.0) | ~**0.76** |
| Oracle CD (best ternary в `{-1,0,1}`) | ~**0.44** |
| Phase 1 target | **< 0.001** |
| FP32 v0 H=128 + rel_decay | **0.00073** ✅ (но **26.8B** total — size fail) |

**Вывод:** representational floor ~0.44 + STE plateau ~0.76 → **менять форму**, не STE sweeps.  
Per-chain × 116k @ rel≈1 **не масштабировать** как path к student.

Детали: `lerning_compare.md` § «exact ternary STE R&D session 2», D6/D6b.

### Production target: **v2 shared FFN layer**

**Решение (2026-06-17):** student = **один тернарный FFN на слой** (shared gate/up/down), как у Qwen — **не** 116k независимых micro-net.

Спека: § «Architecture v2» ниже.

---

## Architecture v2 — shared FFN layer (spec)

### Мотивация

| v1 per-chain | v2 shared layer |
|--------------|-----------------|
| `total = 116736 × params_per_chain` | `total ≈ 24 × params_per_layer` (+ attention позже) |
| Scalar `y` на neuron | Vector **FFN output** `[896]` |
| 116k `.bin`, 116k train jobs | **1 extract + 1 train / layer** |
| exact ternary floor **rel≈0.44** (oracle) | Teacher-shaped **full matmul** — capacity как у Qwen FFN |

### Student forward (один FFN block, layer `L`)

Symbols: `D=896`, `I=4864` (Qwen2.5-0.5B).

```
x [B, D]                           # input to FFN (post-norm hidden)
gate = TernaryLinear(D → I)(x)     # W_gate [I, D] shadow → STE
up   = TernaryLinear(D → I)(x)     # W_up   [I, D]
h    = SiLU(gate) ⊙ up             # [B, I]
y    = TernaryLinear(I → D)(h)     # W_down [D, I]
out  = y                           # Phase 1 target (см. residual ниже)
```

**Optional (Phase 1b+):** `out = x + y` (residual), если target extract включает skip.

**Weights:** FP32 shadow + STE ternary в forward; **без** per-chain `α` (масштаб — через shadow / опц. per-channel FP32 scales, TBD).

### Teacher target (extract)

На `N` samples (default **100k**, holdout 10% @ seed 42):

| Поле | Shape | Как получить |
|------|-------|--------------|
| `X` | `[N, D]` | Random `N(0,1)` (как chain v1) **или** activations из forward teacher (TBD) |
| `Y` | `[N, D]` | `down(SiLU(X W_g^T) ⊙ (X W_u^T))` — **FFN output до residual** |

Primary Phase 1: MSE между student `y` и teacher `Y`, **mean over batch and dims**.

**Не** scalar chain `SiLU(w_g·x)·(w_u·x)` для одного neuron.

### Размер (Qwen2.5-0.5B)

**Full-rank ternary (teacher shapes):**

| Matrix | Shape | Params |
|--------|-------|--------|
| `W_gate` | I×D | 4 358 144 |
| `W_up` | I×D | 4 358 144 |
| `W_down` | D×I | 4 358 144 |
| **Per layer** | | **13 074 432 (~13.1M)** |
| **24 FFN layers** | | **~314M** |

| vs Qwen ~0.5B | ternary packed (~2 bit) |
|---------------|---------------------------|
| **Меньше** bf16 ~1 GB | order **~80 MB** FFN-only (314M×2bit) |

**Low-rank variant (R&D fallback):** `W ≈ A @ B`, rank `r`:

```
params_per_proj ≈ r × (D + I) = r × 5760
params_per_layer ≈ 3 × r × 5760 = 17280 × r
```

| r | / layer | 24 layers |
|---|---------|-----------|
| 64 | ~1.1M | ~**26M** |
| 128 | ~2.2M | ~**53M** |
| 256 | ~4.4M | ~**106M** |

Full-rank v2 **всё ещё < Qwen** по total params; low-rank — запас по RAM.

### Phase 1 (v2 gate)

| | |
|--|--|
| **Unit** | **Layer 0 FFN** (не `L00_N0000` chain) |
| **Data** | `N=100k`, holdout 10%, seed 42 |
| **Criterion** | `rel_holdout < 0.001` где `rel = MSE / mean(Var(Y[:, d]))` по dim **или** scalar MSE/Var(flatten Y) — **зафиксировать в первом Torch PR** (default: mean variance по 896 dims) |
| **Backend** | Torch first → Mojo после breakthrough |
| **Quant** | Ternary STE (reuse `ternary.py` pattern, batched matmul) |

### Dataset wire format (v2, **новый контракт**)

**Не** reuse scalar `CAL158` chain `.bin` (version 1, `Y [N]`).

Proposal **`CAL158L`** (layer FFN, version TBD):

```
magic     = b"CAL158L"   # или CAL158 version=2 + type tag — зафиксировать в PR
n_samples, input_dim D, output_dim D_out   # Phase 1: D_out=D=896
X float32 [N × D]
Y float32 [N × D_out]
sidecar .json: model, layer, seed, target_kind=ffn_out
```

Path example: `data/layers/L00_ffn.bin`.

**v1** `data/chains/L00_N0000.bin` — **legacy / frozen**, не Phase 2 @ scale.

### Env (proposal)

| Var | Meaning |
|-----|---------|
| `CALIBER158_DISTILL_UNIT` | `chain` (v1 legacy) \| `layer_ffn` (v2) |
| `CALIBER158_LAYER` | 0…23 |
| `CALIBER158_FFN_RANK` | `0` = full rank; `>0` = low-rank r |
| (reuse) | `SAMPLES`, `SEED`, `HOLDOUT_FRACTION`, `LR`, `DEVICE`, STE knobs |

### Implementation roadmap

| Step | Deliverable | Layer |
|------|-------------|-------|
| **V2-1** | `extract_layer_ffn.py` + `CAL158L` write/read | Python host |
| **V2-2** | `TernaryFFNLayer` + `train_layer.py` (Torch) | Python R&D |
| **V2-3** | Phase 1 @ L0, 100k, ternary STE | Torch |
| **V2-4** | FP32 diagnostic control (optional) | Torch |
| **V2-5** | Mojo: ternary matmul `D→I`, `I→D`, GPU batch | kernel |
| **V2-6** | Layers 1…23, checkpoint, assembly sketch | host |
| **V2-7** | Attention / full block | Phase 3+ |

**Не приоритет:** порт v1 `exact` в Mojo; Phase 2 batch extract 4864 chains; train 116k per-chain @ rel≈1.

### Reuse from v1

| Component | v2 |
|-----------|-----|
| STE / shadow / AdamW | ✅ pattern |
| Holdout LCG split | ✅ |
| pixi, Makefile discipline | ✅ |
| Mojo GPU train loop | ⚠️ новые kernel sizes (I=4864) |
| `micro_net_batch`, chain `.bin` | legacy only |

### Open decisions (resolve in V2-1 PR)

1. Target: FFN `y` only vs `x + y` residual.
2. `X` sampling: Gaussian vs real activations.
3. `rel` for vector `Y`: per-dim mean vs flatten.
4. Per-channel FP32 scales (BitNet-style) — yes/no Phase 1.
5. Full-rank vs low-rank default for first experiment.

---

## Что уже сделано

### Репозиторий и toolchain

- Mojo-проект на **pixi** (`pixi.toml`, Mojo 1.0.0b1)
- Зависимость **`max >=26.3.0,<27`** (GPU API, `std.gpu.host`)
- Пакет в `src/chain/`
- CLI: `main.mojo` → `info | smoke | train | test-grad | test-grad-gpu | parity-export`
- **Makefile** в корне — `make test-grad-v1`, `make test-grad-gpu-v1`, gate: **`make test`**
- Правило для агентов: `.cursor/rules/makefile-commands.mdc`

### Конфигурация через `.env`

- Шаблон: `.env.example`, рабочий: `.env` (gitignored)
- Префикс: `CALIBER158_*`
- Загрузка: `scripts/load-env.sh` + `python/env_config.py`
- Mojo: `src/chain/env.mojo` (`TrainEnv.load()`)

Ключевые env (student): `HIDDEN_DIM`, `DATASET`, `EPOCHS`, `BATCH_SIZE`, `LR`, **`DEVICE`** (`cuda`|`cpu`), **`TRAIN_BACKEND=mojo`**.

Teacher: `TORCH=cuda|cpu`, `MODEL`, `SAMPLES`, `LAYER`, `NEURON`, …

### Python (teacher) — без изменений

- `python/extract_chain.py` — Qwen → `(X, Y)` → `.bin` + `.json`
- `CALIBER158_TORCH=cuda` — extract на GPU ок
- Кэш: `models/huggingface/`

### Mojo (student) — батч + полный GPU train (v2)

**Архитектура train (один путь, sample-by-sample удалён):**

| Модуль | Роль |
|--------|------|
| `buffer.mojo` | `ChainData` — dense `X`/`Y` на host; один upload в GPU при старте train |
| `micro_net_batch.mojo` | `BatchMicroNet` — `train_step_cpu()` (CPU path) |
| `gpu/buffer_pool.mojo` | **`GpuTrainState`** — persistent device buffers (~75–85 MB при B=64, H=128, D=896) |
| `gpu/device.mojo` | upload/download, zero, pointer offset, Float64 reduce |
| `gpu/quantize.mojo` | STE quantize shadow → ternary на device |
| `gpu/ternary_matmul.mojo` | CUDA: ternary matmul, SwiGLU, head reduce, scale(`alpha_dev`) |
| `gpu/backward.mojo` | STE backward: partial `[B×H×D]` / `[B×H]` + reduce по B (без atomics) |
| `gpu/adamw.mojo` | AdamW на device (shadow + `alpha_dev` + `m`/`v`) |
| `gpu/batch_step.mojo` | `train_step_gpu()` — quantize → forward → backward → AdamW |
| `train.mojo` | `_train_epochs_cpu` / `_train_epochs_gpu` (`GpuTrainState` один раз на train) |
| `adamw.mojo` | AdamW на host — **только CPU path** |
| `device.mojo` | `DeviceKind`, `resolve_device_from_env()` |
| `holdout.mojo` | train/holdout split (LCG shuffle, `CALIBER158_SEED`) |
| `metrics.mojo` | `Var(Y)`, `relative_mse` |
| `test_batch_grad.mojo` | CPU regression + `run_gpu_backward_regression_test()` |

Удалено: `micro_net.mojo` (`accumulate_grad`, sample-by-sample).

**GPU v2 hot loop (`device=cuda`):**

```
quantize → forward (x_dev offset, alpha_dev) → backward → AdamW
```

- Shadow weights + optimizer state — **source of truth на GPU**
- `BatchMicroNet` на host — только `init_random_weights`; опц. `download_shadow()` позже для checkpoint
- В hot loop host не участвует в backward/AdamW; только scalar loss download для лога `mse=`

**Проверено:**

- `make test` — green (`build` 0 warnings + `test-grad` + `smoke` на CUDA path)
- `make test-grad` — CPU batch vs reference, `max|grad_diff| < 1e-5`
- `make test-grad-gpu` — CPU vs GPU backward, loss `< 1e-5`, grads `< 1e-4` (float32 reorder в parallel matmul)
- `make smoke` / `make smoke-cuda` — MSE падает на synthetic
- **`make train-cuda`** на `L00_N0000.bin` (4096, RTX 3050 Ti): 10 epochs **~5 с** (было ~59 с в v1); MSE `44.6M → 9.9M` (≈ v1: `44.6M → 9.3M`)
- Compile + run требуют видимый NVIDIA GPU (`sm_86` ок для 3050 Ti)

### Архитектура v1 — dual SwiGLU + residual (2026-06-16) ✅

Реализовано по плану `.cursor/plans/architecture_v1_swiglu_7baef79e.plan.md`:

| Компонент | Статус |
|-----------|--------|
| `src/chain/arch.mojo` — `ArchKind`, `arch_from_env()`, param counts | ✅ |
| CPU: `BatchMicroNet` v0/v1, block2 zero-init, `eval_mse`, AdamW block2 | ✅ |
| `make test-grad-v1`, `make test-grad-gpu-v1` | ✅ |
| GPU: `vector_add_kernel`, forward v1 (h0/hidden2/h1), quantize ×5 | ✅ |
| GPU backward: head→`dL_dh`, block2 input-grad, skip, block1 | ✅ |
| `GpuTrainState` block2 buffers (~+130 MB @ B=64, H=512), AdamW gate2/up2 | ✅ |
| `download_shadow` gate2/up2 для holdout | ✅ |
| `CALIBER158_ARCH`, `.env.example`, `docs/architecture.md` | ✅ |
| `scripts/load-env.sh` — shell env не перезаписывается `.env` | ✅ |
| Holdout прогон #6 (30 ep, H=512) | ✅ см. `lerning_compare.md` |
| Holdout **#6b FP32 v1** (30 ep) | ✅ `rel≈1.004` — ≈ FP32 v0, ≈ ternary v1 |

**Результат holdout #6:** `rel_holdout ≈ 1.03` — **≈ v0** (1.04); Phase 1 не достигнута.

CLI: `info | smoke | train | test-grad | test-grad-gpu` — v1 при `CALIBER158_ARCH=v1`.

---

### Holdout + тюнинг качества (2026-06-16)

- **Holdout split** в train: `CALIBER158_HOLDOUT_FRACTION=0.1`, seed=`CALIBER158_SEED=42` → 3687 train / 409 holdout
- Лог каждой эпохи: `train_mse`, `holdout_mse`, `rel_holdout = MSE/Var(Y)`
- **Fix:** `download_shadow()` перед holdout eval на GPU path (иначе holdout_mse застывал)
- `micro_net_batch.eval_mse()` — inference-only MSE на host
- Прогоны и выводы: **`lerning_compare.md`**

| Прогон | H | Epochs | Train MSE | Holdout MSE | rel_holdout | Время |
|--------|---|--------|-----------|-------------|-------------|-------|
| GPU v1 | 128 | 10 | 9.3×10⁶ | — | — | ~59 с |
| Тюнинг | 256 | 50 | 0.0329 | — | ~0.99* | ~17 с |
| Тюнинг | 512 | 100 | 0.0325 | — | ~0.98* | ~50 с |
| **Holdout v0** | **512** | **30** | **0.0321** | **0.0384** | **1.04** | **~45 с** |
| **Holdout v1** | **512** | **30** | **0.0321** | **0.0382** | **1.03** | **~63 с** |

\* train-only, до holdout

**Вывод Phase 1 (2026-06-17, финал после 100k + rel_decay):**

- **4k ternary:** `rel_holdout ≈ 1` (#5, #T1) — плато @ ep ~30.
- **100k re-extract** ✅ (`342 MB`, 90k/10k holdout).
- **100k ternary v0 H=128, LR=3e-4, 20 ep:** `rel ≈ 1.002` → дрейф до 1.24 (#100k-c); LR=1e-4 (#100k-f) — то же.
- **100k FP32 v0 constant LR:** `rel ≈ 0.0049` (#100k-e).
- **100k FP32 v0 + rel_decay (3e-4→1e-4 @ rel<0.01):** **`rel = 0.00073`** (#100k-h) — **Phase 1 ✅** (Torch FP32 shadow).
- **Вывод:** capacity + data **достаточны** (FP32); **STE ternary — блокер** production path.

Детали — `lerning_compare.md` § «100k re-extract», § «rel_decay», § «Диагностика Phase 1».

### Уже прогнано пользователем

- `data/chains/L00_N0000.bin` — **100k samples** (342 MB, re-extract 2026-06-17); ранее 4096 (14 MB)
- Старый CPU sample-by-sample train (4096): ~4+ мин/epoch 0, прервано
- GPU v1 train (4096): ~59 с / 10 epochs (до v2)

### Важно: PyTorch — teacher + R&D student

- **Teacher extract** — PyTorch (`python/extract_chain.py`), без изменений.
- **Production student** — **Mojo** (`CALIBER158_TRAIN_BACKEND=mojo`), gate `make test`.
- **Torch-prototype student** — ✅ реализован (`python/student/`), gate `make test-torch-parity` (не в `make test`); Mojo **не выбрасываем**, результаты сверяем по одним метрикам и `.bin`.

### Torch student prototype (2026-06-16) ✅

Параллельный R&D path для быстрых arch-экспериментов без новых Mojo kernels.

| Компонент | Статус |
|-----------|--------|
| `python/student/` — dataset, holdout, rng, ternary, model, metrics, train_chain | ✅ |
| `StudentEnv` / `load_student_env()` в `python/env_config.py` | ✅ |
| `MicroNet` v0/v1, LCG init, block2 zero, STE ternary | ✅ |
| Holdout eval + лог epoch как Mojo | ✅ |
| `make train-torch`, `smoke-torch`, `test-torch-parity` | ✅ |
| Mojo `parity-export` для golden-тестов | ✅ |
| Parity tests: holdout indices + 1-batch loss vs Mojo CPU | ✅ (`make test-torch-parity`) |
| Holdout прогоны **#T1** (v0), **#T2** (v1), **#T3** (FP32 v1) | ✅ см. `lerning_compare.md` |

**Результат #T1–#T3:** `rel_holdout ≈ 1.033 / 1.033 / 1.004` — **≈ Mojo #5/#6/#6b**. Wall time ~6–7 с vs ~45–63 с Mojo GPU.

**Torch R&D (2026-06-17):** #T4–#T15 sweep; диагностика D1–D5; **100k** #100k-c…#100k-l; **rel_decay** schedule.

**Torch-only infra:** `CALIBER158_ARCH=v1b|exact`, `BLOCK2_INIT`, `LR_SCHEDULE=none|cosine|rel_decay`, `LR_REL_THRESHOLD` — defaults = Mojo parity.

**Ключевой результат 100k:** FP32 v0 H=128 + **rel_decay** → **`rel=0.00073`** (Phase 1 ✅); ternary → `rel≈1` @ те же hyperparams.

---

## Текущая фаза

**Phase 1 (v2):** **layer 0 FFN**, shared ternary weights, **100k** samples.

Критерий успеха: `rel_holdout < 0.001` на vector FFN output `[896]` (см. § Architecture v2).

| Path | Статус | Роль |
|------|--------|------|
| **v2 ternary FFN** (production target) | ❌ не реализован | **следующий R&D + код** |
| **v1 ternary per-chain** (`exact`/v0) | ❌ floor ~0.44 / STE ~0.76 | **legacy**, не масштабировать |
| **v1 FP32 v0 H=128** | ✅ `rel=0.00073` | capacity diagnostic only |
| **Mojo v1** (v0/v1 micro-net) | ✅ infra | legacy train path |
| **Mojo v2 FFN** | ❌ | после Torch Phase 1 |

Детали v1: `lerning_compare.md` § session 2. Спека v2: § «Architecture v2» выше.

---

## Архитектура v1 — статус (реализовано 2026-06-16)

Спецификация: `docs/architecture.md`. Детальный план (historical): `.cursor/plans/architecture_v1_swiglu_7baef79e.plan.md`.

**Сделано:** CPU + GPU v1, tests, holdout #6, docs — см. секцию «Архитектура v1» выше.

**Не сделано / вне scope (v1):**

| Задача | Примечание |
|--------|------------|
| Phase 1 ternary per-chain | ❌ **closed** — pivot v2 |
| ~~Порт `exact` в Mojo~~ | **отменён** как priority |
| Phase 2 batch extract 4864 chains | **отменён** @ v1; v2 = 1 bin/layer |
| v0/v1 Mojo stack | ✅ legacy |
| Checkpoint export (chain) | superseded by layer checkpoint (v2 TBD) |

**v2 — не сделано:**

| Задача | Step |
|--------|------|
| `extract_layer_ffn.py` + `CAL158L` | V2-1 |
| Torch `TernaryFFNLayer` + train | V2-2, V2-3 |
| Phase 1 ternary @ L0 100k | V2-3 |
| Mojo v2 FFN kernels | V2-5 |

**Команды v1:**

```bash
make test-grad-v1 test-grad-gpu-v1
CALIBER158_ARCH=v1 CALIBER158_HIDDEN_DIM=512 CALIBER158_EPOCHS=30 \
  CALIBER158_LR=0.003 CALIBER158_WEIGHT_DECAY=0.001 make train-cuda
```

---

## Что делать дальше (приоритет)

### 1. **v2 Phase 1** — shared ternary FFN layer 0 (Torch)

| Step | Действие |
|------|----------|
| **V2-1** | `extract_layer_ffn.py` + read/write `CAL158L`; `data/layers/L00_ffn.bin` @ 100k |
| **V2-2** | `TernaryFFNLayer` (full-rank D→I→D), STE, AdamW, holdout |
| **V2-3** | Train @ 100k; gate `rel_holdout < 0.001` |
| **V2-4** | (опц.) FP32 control; (опц.) low-rank `FFN_RANK=128` |

**Не делаем:** v1 STE sweeps; Mojo exact port; 4864 chain extract @ v1.

**Целевая команда (после V2-1…2):**

```bash
# TBD — example
CALIBER158_DISTILL_UNIT=layer_ffn CALIBER158_LAYER=0 CALIBER158_SAMPLES=100000 \
  make extract-layer
CALIBER158_EPOCHS=20 CALIBER158_LR=0.0003 CALIBER158_WEIGHT_DECAY=0.001 \
  make train-torch-layer
```

### 2. Mojo v2 FFN (после Torch Phase 1 ✅)

- Ternary matmul `896→4864`, SwiGLU, `4864→896`
- `make test-grad-ffn`, GPU train layer 0 @ 100k

### 3. Layers 1…23 + checkpoint + assembly (Phase 2+)

### 4. Attention / full transformer block (Phase 3+)

### 5. Документация

- **README** sync (v1 legacy + v2 target)
- `docs/architecture.md` — pointer на v2 (или `docs/architecture-v2.md`)

---

## PyTorch student prototype (parallel, не замена Mojo)

**Статус v1:** per-chain R&D **закрыт** (#100k-m…t, D6 — `lerning_compare.md`).

**Статус v2:** **следующий Torch path** — `TernaryFFNLayer` + layer extract (§ Architecture v2).

**v1 Torch (legacy, не удалять):** `python/student/` — `MicroNet` v0/v1/exact, `train_chain.py`, `make train-torch`.

**Сделано:**

- 100k dataset, holdout, `rel_decay`, grad_clip env
- FP32 diagnostic (#100k-h, #100k-n) — **закрыто**, код Torch без FP32 path
- Warm-start — **удалён** (не помог)
- Session 2: `INIT=teacher|cd`, `STE=masked`, `diag_ternary_fit.py`, `coord_desc_init.py`
- Прогоны #100k-m…t, D6/D6b — `lerning_compare.md`

**v2 Torch (planned):** `extract_layer_ffn.py`, `TernaryFFNLayer`, `train_layer.py`.

**Следующий код:** V2-1 extract + V2-2 train layer 0.

### Контракт паритета с Mojo (обязательно)

| Область | Mojo (эталон) | Torch (должен совпасть) |
|---------|---------------|-------------------------|
| Dataset | `dataset.mojo` / `ChainData` | `read_dataset()` — тот же layout `.bin` |
| Holdout split | `holdout.mojo`: LCG `seed=CALIBER158_SEED`, Fisher–Yates, **первые** `holdout_count` индексов → holdout | **Бит-в-бит** те же `train`/`holdout` индексы (константы LCG: `6364136223846793005`, `unit_float = (bits>>11)/(2^53)`) |
| Loss | MSE, mean over batch | `F.mse_loss(..., reduction='mean')` |
| `Var(Y)` | population variance holdout | то же на holdout split |
| Init weights | LCG seed `0xC158_C158`, scale `INIT_SCALE`, block2 **zero** | тот же алгоритм для сравнимого старта |
| Ternary | STE: forward `sign/threshold`, backward через shadow | `torch.autograd.Function` (always ternary) |
| Arch | Mojo: `v0\|v1` (legacy stack) | Torch: **`exact`** (production target) + `v0/v1/v1b` (parity/legacy) |
| Optimizer | AdamW, те же β, eps, weight_decay | `torch.optim.AdamW` |

**Sanity после первого train:** на одном batch с фиксированным init — forward loss Torch vs Mojo CPU в пределах ~1e-4 (float reorder ok).

### Структура файлов (реализовано)

```
python/
├── extract_chain.py
├── env_config.py             # StudentEnv + load_student_env()
└── student/
    ├── dataset.py            # read/write CAL158 .bin
    ├── holdout.py            # split = holdout.mojo
    ├── rng.py                # LCG + init_random_weights
    ├── ternary.py            # quantize + STE autograd
    ├── model.py              # MicroNet v0 / v1 / v1b / exact (Torch R&D)
    ├── metrics.py
    └── train_chain.py        # CLI: train | smoke
tests/
├── test_holdout_golden.py
├── test_forward_loss_mojo.py
└── test_gradcheck_tiny.py    # optional, не в gate
scripts/run-test-torch-parity.sh
```

Точка входа: `python/student/train_chain.py`.

### Этапы реализации

#### Этап 0 — Scaffold ✅

- [x] `python/student/dataset.py` — read/write `.bin`
- [x] `StudentEnv` в `python/env_config.py`
- [x] `make train-torch` / `smoke-torch` / `test-torch-parity` — **не** в `make test`
- [x] `.env.example`: `CALIBER158_TRAIN_BACKEND` — torch только для CLI

#### Этап 1 — v0 train + holdout ✅

- [x] `MicroNet` v0: gate/up/head + α
- [x] STE ternary на matmul
- [x] AdamW train loop, batched, partial last batch
- [x] Holdout eval каждую epoch — лог как Mojo
- [x] Smoke: synthetic LCG, `make smoke-torch`
- [x] #T1: `rel_holdout ≈ 1.033` ≈ Mojo #5

#### Этап 2 — v1 + FP32 diagnostic ✅

- [x] `MicroNet` v1: block2 H→H, residual, block2 zero-init
- [x] `CALIBER158_QUANTIZE=0` — FP32 path
- [x] **#T1**, **#T2**, **#T3** в `lerning_compare.md`

#### Этап 3 — v1b + sweep (#T4–#T15) ✅ (negative)

- [x] v1b linear skip в Torch (`CALIBER158_ARCH=v1b`) — **не помог** (rel 4–7300)
- [x] 50 ep v1, H↑768 (#T4, #T7) — rel ≈ 1
- [x] block2 lcg, cosine LR (#T11–#T15) — lcg **взрыв**; cosine ≈ baseline
- [x] **Диагностика D1–D5** — teacher replay OK; bottleneck ≠ teacher form

#### Этап 4 — 100k + FP32 + rel_decay ✅ (2026-06-17)

- [x] Re-extract 100k (`make extract`)
- [x] Torch `arch=exact` — **production target** (ternary `rel≈1` ≈ v0; wins on size)
- [x] #100k-c…#100k-l (ternary, FP32, rel_decay, exact)
- [x] **#100k-h FP32 v0 + rel_decay** — **Phase 1 ✅** `rel=0.00073`
- [x] LR schedule `rel_decay` в Torch (`train_chain.py`, env)

#### Этап 5 — Ternary `exact` per-chain ❌ (2026-06-17, closed)

- [x] STE / init / grad_clip / masked STE / CD / CD→STE — **negative** (lerning_compare § session 2)
- [x] Oracle CD floor ~0.44 — representational limit
- [ ] ~~Порт exact Mojo~~ — **cancelled** (pivot v2)

#### Этап 6 — **v2 shared FFN layer** (следующий)

- [ ] V2-1 extract `CAL158L` + layer 0 dataset
- [ ] V2-2 Torch `TernaryFFNLayer` + train loop
- [ ] V2-3 Phase 1 @ 100k ternary
- [ ] V2-5 Mojo FFN kernels

#### Этап 6 — Optional hardening (частично)

- [x] `test_gradcheck_tiny.py` — файл есть
- [x] Unit test holdout + batch loss vs Mojo (`make test-torch-parity`)
- [ ] Подключить gradcheck к gate (optional)
- [x] Init weights — LCG only, без `torch.manual_seed` (N/A для SEED sync)

### Makefile / CLI (целевое)

```bash
# не в make test
make train-torch          # always ternary STE
make smoke-torch

# 100k ternary baseline (production arch)
CALIBER158_ARCH=exact CALIBER158_EPOCHS=20 CALIBER158_LR=0.0003 \
  CALIBER158_WEIGHT_DECAY=0.001 make train-torch
```

`make test` — **только Mojo**, без изменений.

### Критерии «prototype готов»

| Критерий | |
|----------|--|
| Читает `data/chains/L00_N0000.bin` | ✅ |
| Holdout split = Mojo @ seed 42 | ✅ |
| Лог epoch совпадает по форме с Mojo | ✅ |
| v0 + v1 train на CUDA/CPU | ✅ |
| Прогон записан в `lerning_compare.md` (#T*) | ✅ |
| Mojo `make test` не тронут | ✅ |

### Порядок работ vs Mojo

1. **v2 Torch** — layer 0 FFN Phase 1
2. **v2 Mojo** — FFN kernels + train
3. v1 chain code — **frozen**, `make test` без регрессии
4. Layers 1…23, assembly, attention — позже

---

## GPU v2.1 и позже (не блокер)

Оптимизации и инфраструктура — **не перенос** (hot loop уже на GPU):

| Задача | Зачем |
|--------|--------|
| Atomics вместо temp `[B×H×D]` / `[B×H]` для grads | быстрее; риск float reorder vs CPU |
| Fused gate+up ternary matmul (один launch) | меньше kernel launches |
| Pinned host buffers для initial dataset upload | быстрее один upload при старте |
| CI без NVIDIA GPU | отдельный compile target без `gpu/*.mojo` |
| `make test-grad-gpu` в CI job с GPU runner | регрессия backward на CI |
| Tensor-core path для dense shadow matmul | только если уйдём от ternary в backward inner loop |

**Намеренно на host (не баг):**

- CPU path (`CALIBER158_DEVICE=cpu`) — `train_step_cpu` + host `adamw.mojo`
- Загрузка `.bin` → `ChainData` на host → один upload в `GpuTrainState`
- `download_grads` / `backward_only_gpu` — только тесты
- Scalar loss download per batch — лог epoch MSE

---

## Быстрые команды

```bash
cp .env.example .env
make install
make setup
make info
make extract           # teacher dataset (100k: CALIBER158_SAMPLES=100000)
make smoke             # synthetic student
make smoke-cuda        # synthetic на CUDA path
make train-cuda        # student на CALIBER158_DATASET
make test              # gate перед commit
make test-grad         # CPU v0 grad regression
make test-grad-v1      # CPU v1 grad regression
make test-grad-gpu     # CPU vs GPU backward v0
make test-grad-gpu-v1  # CPU vs GPU backward v1
make test-torch-parity # Torch vs Mojo parity (не в make test)
make train-torch       # Torch student on CALIBER158_DATASET
make smoke-torch       # Torch synthetic smoke
```

Эквивалент через pixi: `pixi run <task>` (см. `pixi.toml`).

---

## Известные ограничения / техдолг

- **v1 per-chain ternary** — floor ~0.44 (CD), STE ~0.76; **не production path**
- **v2** — spec only; extract/train **не реализованы**
- v1 `L00_N0000.bin` @ 100k — legacy dataset for chain R&D
- README не синхронизирован (v1 + v2)
- `mojo build` требует **видимый NVIDIA GPU** при compile (CI workaround — v2.1)
- `make test` не включает `test-grad-gpu` (runtime GPU); build всё равно тянет GPU-модули
- GPU grad regression: допуск **1e-4** (не 1e-5) из‑за float32 reorder в parallel matmul
- `down_proj` (4864→896) не покрыт micro-сетью
- `dataset.mojo`: `sample_input()` остаётся для совместимости, не используется в train hot path

---

## Критерии GPU train — статус

| Критерий | Статус |
|----------|--------|
| Батчевый CPU train + grad regression | ✅ `make test-grad` |
| `CALIBER158_DEVICE`, `DeviceKind`, лог device | ✅ |
| GPU forward (ternary matmul + SwiGLU) | ✅ |
| GPU backward + AdamW + persistent buffers | ✅ GPU v2 |
| `alpha_dev` в forward/backward/AdamW | ✅ |
| `make test` green, 0 warnings | ✅ |
| `make test-grad-gpu` | ✅ (grad `< 1e-4`, loss `< 1e-5`) |
| `make train-cuda` 10 epochs на `L00_N0000.bin` | ✅ **~5 с** (3050 Ti) |
| MSE curve ≈ GPU v1 | ✅ 44.6M → 9.9M |
| Нет per-batch upload весов/`X` | ✅ один upload at start |
| Checkpoint export | ❌ |
| Holdout + relative MSE | ✅ `CALIBER158_HOLDOUT_FRACTION`, `lerning_compare.md` |
| Phase 1 v2 ternary FFN L0 | ❌ not started |
| Phase 1 v1 ternary per-chain | ❌ closed (~0.44 floor) |
| Phase 1 v1 FP32 @ 100k (#100k-h) | ✅ diagnostic `rel=0.00073` |
| Production target | **v2 shared FFN** (HANDOFF § Architecture v2) |
| Legacy v0/v1 Mojo | ✅ |
| v1 Torch (chain) | ✅ frozen |
| **`exact` Mojo port** | ❌ **cancelled** |
| `make test-grad-v1` / `test-grad-gpu-v1` | ✅ |
| Teacher `make extract` без регрессии | ✅ (код не менялся) |
| Torch student prototype (v0/v1, parity) | ✅ `make test-torch-parity` |
| Torch holdout #T1–#T3 ≈ Mojo | ✅ `lerning_compare.md` |

---

## Структура файлов (актуальная)

```
Caliber158/
├── Makefile              # make help | test | test-grad-gpu | train-cuda | …
├── .env / .env.example
├── main.mojo
├── pixi.toml             # mojo + max
├── src/chain/
│   ├── arch.mojo
│   ├── buffer.mojo
│   ├── micro_net_batch.mojo
│   ├── train.mojo, adamw.mojo, dataset.mojo, env.mojo, device.mojo
│   ├── holdout.mojo, metrics.mojo
│   ├── ternary.mojo, grads.mojo, rng.mojo
│   ├── test_batch_grad.mojo
│   └── gpu/
│       ├── device.mojo
│       ├── ternary_matmul.mojo
│       ├── batch_step.mojo      # full GPU pipeline
│       ├── buffer_pool.mojo     # GpuTrainState
│       ├── quantize.mojo
│       ├── backward.mojo
│       └── adamw.mojo
├── python/
│   ├── extract_chain.py
│   ├── env_config.py
│   └── student/              # Torch R&D prototype (не в make test)
│       ├── dataset.py, holdout.py, rng.py, ternary.py
│       ├── model.py, metrics.py, train_chain.py
├── tests/                    # test_holdout_golden, test_forward_loss_mojo, …
├── scripts/
│   └── run-test-torch-parity.sh
├── docs/
├── lerning_compare.md  # сравнение прогонов train/holdout
├── data/chains/          # L00_N0000.bin + .json (generated)
└── models/huggingface/
```
