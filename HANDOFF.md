# Handoff для следующего чата

Обновлено: 2026-06-16 (holdout + Phase 1 quality runs)

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
| **Holdout** | **512** | **30** | **0.0321** | **0.0384** | **1.04** | **~45 с** |

\* train-only, до holdout

**Вывод Phase 1:** train ≈ holdout ≈ `Var(Y)` → **underfit** (предсказание среднего), не overfit. Цель `rel < 0.001` — в ~1000× дальше. Следующий шаг — **архитектура v1**, не гиперпараметры.

### Уже прогнано пользователем

- `data/chains/L00_N0000.bin` (14 MB) + `L00_N0000.json`, 4096 samples
- Старый CPU sample-by-sample train: ~4+ мин/epoch 0, прервано (`mse≈4.46e7`)
- GPU v1 train: ~59 с / 10 epochs (до v2)

### Важно: PyTorch ≠ student

- **PyTorch только для teacher.**
- Student — **Mojo**; `CALIBER158_TRAIN_BACKEND=mojo`.

---

## Текущая фаза

**Phase 1**: одна цепочка `L00_N0000` — качество **не достигнуто** (holdout проверен).

Критерий успеха: holdout MSE < 1e-4 (или `rel_holdout < 0.001`).

GPU v2 и holdout закрыты; **следующий приоритет — архитектура v1** (второй SwiGLU-блок + residual) или FP32 diagnostic run.

---

## Что делать дальше (приоритет)

### 1. Архитектура v1 / диагностика — **следующий шаг**

- Holdout показал underfit (`rel_holdout ≈ 1.0` при H=512) — см. `lerning_compare.md`
- v1: второй SwiGLU-блок + residual (`docs/architecture.md`)
- Опц.: один прогон FP32 shadow без ternary — хватает ли ёмкости без квантизации

### 2. Checkpoint export (нет в коде)

После успешного train — сохранять gate/up/head ternary, shadow (опц.), α, chain_id → `data/checkpoints/L00_N0000.bin` (формат не спроектирован). На GPU path: `GpuTrainState.download_shadow()` уже есть для чтения весов с device.

### 3. Phase 2 — batch extract layer 0

- `batch_extract.py`: 4864 цепочек → `L00_N####.bin`
- Параллельный train worker pool
- `data/chains/manifest.jsonl`

### 4. Phase 3+

- Сборка тернарного MLP слоя, 24 слоя FFN, attention — отдельно

### 5. Документация

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
| `make train-cuda` 10 epochs на `L00_N0000.bin` | ✅ **~5 с** (3050 Ti) |
| MSE curve ≈ GPU v1 | ✅ 44.6M → 9.9M |
| Нет per-batch upload весов/`X` | ✅ один upload at start |
| Checkpoint export | ❌ |
| Holdout + relative MSE | ✅ `CALIBER158_HOLDOUT_FRACTION`, `lerning_compare.md` |
| Phase 1 quality (`rel_holdout < 0.001`) | ❌ сейчас ~1.04 |
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
├── scripts/
├── docs/
├── lerning_compare.md  # сравнение прогонов train/holdout
├── data/chains/          # L00_N0000.bin + .json (generated)
└── models/huggingface/
```
