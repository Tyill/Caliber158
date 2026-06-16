# Handoff для следующего чата

Обновлено: 2026-06-16 (v1 GPU + holdout; v1 train нестабилен)

## Идея проекта

Аппроксимировать Qwen **не прямой заменой весов**, а локальной дистилляцией: каждая скалярная MLP-цепочка `hidden → SwiGLU → 1` заменяется отдельной тернарной micro-сетью `{-1,0,1}` + один обучаемый масштаб **α** (FP32).

Первый target: **Qwen2.5-0.5B** (`hidden=896`, `intermediate=4864`, `24` слоя → **116 736** MLP-цепочек).

Подробнее: `docs/target.md`, `docs/architecture.md`, `docs/qwen2.5-0.5b.md`.

---

## Что уже сделано

### Репозиторий и toolchain

- Mojo-проект на **pixi** (`pixi.toml`, Mojo 1.0.0b1)
- Зависимость **`max >=26.3.0,<27`** (GPU API, `std.gpu.host`)
- Пакет в `src/chain/`
- CLI: `main.mojo` → `info | smoke | train | test-grad | test-grad-gpu`
- **Makefile** в корне — основной интерфейс команд (`make help`); gate перед commit: **`make test`**
- Правило для агентов: `.cursor/rules/makefile-commands.mdc`

### Конфигурация через `.env`

- Шаблон: `.env.example`, рабочий: `.env` (gitignored)
- Префикс: `CALIBER158_*`
- Загрузка: `scripts/load-env.sh` + `python/env_config.py`
- Mojo: `src/chain/env.mojo` (`TrainEnv.load()`)

Ключевые env (student): `ARCH` (`v0`|`v1`), `HIDDEN_DIM`, `DATASET`, `EPOCHS`, `BATCH_SIZE`, `LR`, `HOLDOUT_FRACTION`, `INIT_SCALE`, `INIT_SCALE_BLOCK2` (опц.), `BLOCK2_RESIDUAL_SCALE` (опц.), **`DEVICE`** (`cuda`|`cpu`), **`TRAIN_BACKEND=mojo`**.

Teacher: `TORCH=cuda|cpu`, `MODEL`, `SAMPLES`, `LAYER`, `NEURON`, …

### Python (teacher) — без изменений

- `python/extract_chain.py` — Qwen → `(X, Y)` → `.bin` + `.json`
- `CALIBER158_TORCH=cuda` — extract на GPU ок
- Кэш: `models/huggingface/`

### Mojo (student) — батч + полный GPU train (v2) + arch v1

**Архитектуры (`CALIBER158_ARCH`, `src/chain/arch.mojo`):**

| Arch | Топология |
|------|-----------|
| **v0** | один SwiGLU `D→H` → head → α |
| **v1** | block1 `D→H` → h1; block2 `H→H` → h2; **h = h1 + scale·h2** → head → α |

**Архитектура train (один путь, sample-by-sample удалён):**

| Модуль | Роль |
|--------|------|
| `arch.mojo` | `ArchKind` v0/v1 |
| `holdout.mojo`, `metrics.mojo` | 90/10 split (LCG seed), `Var(Y)`, `rel_holdout` |
| `buffer.mojo` | `ChainData` — dense `X`/`Y` на host; один upload в GPU при старте train |
| `micro_net_batch.mojo` | `BatchMicroNet` — v0/v1 CPU forward/backward + `eval_mse()` |
| `gpu/buffer_pool.mojo` | **`GpuTrainState`** — v0 + **v1** (block2 buffers, scaled residual) |
| `gpu/device.mojo` | upload/download, zero, pointer offset, Float64 reduce |
| `gpu/quantize.mojo` | STE quantize shadow → ternary на device |
| `gpu/ternary_matmul.mojo` | CUDA: ternary matmul, SwiGLU, **scaled_residual_add**, head, scale(α) |
| `gpu/backward.mojo` | v0 `enqueue_backward`; **v1 `enqueue_backward_v1`** (residual + block2) |
| `gpu/adamw.mojo` | AdamW на device; **gate2/up2** через `enqueue_adamw_weight_list` |
| `gpu/batch_step.mojo` | `train_step_gpu()` — quantize → forward → backward → AdamW |
| `train.mojo` | holdout в логе; `download_shadow` перед holdout eval на GPU |
| `adamw.mojo` | AdamW на host — CPU path (+ gate2/up2 для v1) |
| `device.mojo` | `DeviceKind`, `resolve_device_from_env()` |
| `test_batch_grad.mojo` | CPU regression + `run_gpu_backward_regression_test()` (v0) |

**v1 block2 scaling (ternary STE):**

- `CALIBER158_INIT_SCALE_BLOCK2` — shadow init gate2/up2; default `INIT_SCALE / HIDDEN_DIM`
- `CALIBER158_BLOCK2_RESIDUAL_SCALE` — множитель **h2** в `h = h1 + scale·h2`; default `1 / HIDDEN_DIM`
- Без residual scale малый shadow init **не помогает**: после STE все ненулевые веса → ±1

Удалено: `micro_net.mojo` (`accumulate_grad`, sample-by-sample).

**GPU v2 hot loop (`device=cuda`):**

```
quantize → forward (v0 или v1) → backward → AdamW
```

- Shadow weights + optimizer state — **source of truth на GPU**
- `BatchMicroNet` на host — `init_random_weights`; `download_shadow()` перед holdout eval
- В hot loop host не участвует в backward/AdamW; scalar loss download для лога

**Проверено:**

- `make test` — green (`build` 0 warnings + `test-grad` + `smoke` на CUDA path)
- `make test-grad` — CPU batch vs reference, grads `< 2e-4`
- `make test-grad-gpu` — CPU vs GPU backward v0, loss `< 1e-5`, grads `< 1e-4`
- **v0 `make train-cuda`**, `L00_N0000.bin`, H=512, **30 epochs** (~45 с, 3050 Ti):
  - train_mse → **~0.032**, holdout_mse → **~0.038**, **rel_holdout ≈ 1.04** (underfit, ≈ `Var(Y)`)
  - **10 epochs недостаточно** — на ep 9 MSE ещё ~10⁶; плато к ~ep 27
- **v1 GPU train** — компилируется и бежит, но **не сходится** (см. `lerning_comprare.md`)
- v1 CPU vs GPU один батч — loss совпадает (kernels ок)

### Уже прогнано

- `data/chains/L00_N0000.bin` (14 MB) + `L00_N0000.json`, 4096 samples
- Holdout 90/10: **3687 train / 409 holdout** (`CALIBER158_SEED=42`)
- **v0 GPU**, H=512/128, 30 ep: плато `rel_holdout ≈ 1` (underfit)
- **v1 GPU**, H=512/128, 30 ep: loss **взрывается** (~10¹²–10²⁵) — см. **`lerning_comprare.md`**
- Сравнение v0/v1 на H=128 задокументировано в `lerning_comprare.md`

### Важно: PyTorch ≠ student

- **PyTorch только для teacher.**
- Student — **Mojo**; `CALIBER158_TRAIN_BACKEND=mojo`.

---

## Текущая фаза

**Phase 1**: одна цепочка `L00_N0000` — стабилизировать **v1** и улучшить качество **v0**.

Критерий успеха: holdout `rel_holdout < 0.001` (или MSE < 1e-4).

- **v0 GPU**: сходится к `Var(Y)` за ~**30 epochs** (`rel_holdout ≈ 1`) — underfit
- **v1 GPU**: hot loop готов, но train **не сходится** (H=128 и H=512)
- На **10 epochs** v0 выглядит «плохо» (MSE ~10⁶) — это норма до плато

---

## Что делать дальше (приоритет)

### 1. Обучение v1 — что попробовать

| # | Эксперимент | Зачем |
|---|-------------|--------|
| 1 | Sweep `BLOCK2_RESIDUAL_SCALE` (`1/H²`, `1e-5`, …) | STE → ±1; нужен меньший вклад h2 |
| 2 | Отдельный LR / weight decay для block2 | block2 может разгонять AdamW |
| 3 | `test-grad-gpu-v1` в gate | CPU vs GPU v1, один батч |
| 4 | `BLOCK2_RESIDUAL_SCALE=0` | sanity: должен ≈ v0 |
| 5 | Float shadow для block2 (без STE) | проверить, учится ли block2; **меняет контракт** |
| 6 | Pre-norm на h1 перед block2 | стабилизация активаций |
| 7 | Bottleneck H→H/k→H вместо H→H | меньше fan-in |

Сравнение прогонов: **`lerning_comprare.md`**.

### 2. Качество v0 (baseline)

- `EPOCHS=30` в `.env` (минимум для плато)
- Re-extract `SAMPLES=100000`
- Тюнинг `HIDDEN_DIM` 256/512
- **Checkpoint export** (см. ниже)

### 3. Checkpoint export (нет в коде)

После успешного train — gate/up/head (+ gate2/up2 для v1), shadow, α → `data/checkpoints/`. `GpuTrainState.download_shadow()` уже есть.

### 4. Phase 2 — batch extract layer 0

- `batch_extract.py`: 4864 цепочек → `L00_N####.bin`
- Параллельный train worker pool
- `data/chains/manifest.jsonl`

### 5. Phase 3+

- Сборка тернарного MLP слоя, 24 слоя FFN, attention — отдельно

### 6. Документация

- **README** устарел: всё ещё `micro_net.mojo`, 3584/~530k chains, нет `DEVICE`/Makefile/GPU/`test-grad-gpu`
- Обновить layout в README под `micro_net_batch`, `gpu/`, `Makefile`

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
make extract           # teacher dataset
make smoke             # synthetic student
make smoke-cuda        # synthetic на CUDA path
make train-cuda        # student на CALIBER158_DATASET
make test              # gate перед commit
make test-grad         # CPU сверка градиентов
make test-grad-gpu     # CPU vs GPU backward (нужен CUDA runtime)
```

Эквивалент через pixi: `pixi run <task>` (см. `pixi.toml`).

---

## Известные ограничения / техдолг

- README не синхронизирован с кодом (см. выше)
- v1 train нестабилен; нет `test-grad-gpu-v1` в gate
- Нет checkpoint I/O, нет batch extract pipeline
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
| `make train-cuda` v0, 30 ep, `L00_N0000.bin` | ✅ rel_holdout ≈ 1 |
| v1 GPU forward/backward/AdamW | ✅ |
| v1 train сходится | ❌ |
| Holdout + `rel_holdout` в логе | ✅ |
| `lerning_comprare.md` (v0 vs v1 H=128) | ✅ |
| Teacher `make extract` без регрессии | ✅ (код не менялся) |

---

## Структура файлов (актуальная)

```
Caliber158/
├── Makefile              # make help | test | test-grad-gpu | train-cuda | …
├── .env / .env.example
├── main.mojo
├── pixi.toml             # mojo + max
├── src/chain/
│   ├── arch.mojo, holdout.mojo, metrics.mojo
│   ├── buffer.mojo
│   ├── micro_net_batch.mojo
│   ├── train.mojo, adamw.mojo, dataset.mojo, env.mojo, device.mojo
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
├── scripts/
├── docs/
├── data/chains/          # L00_N0000.bin + .json (generated)
├── lerning_comprare.md   # v0 vs v1 train curves
└── models/huggingface/
```
