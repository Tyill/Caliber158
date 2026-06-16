# Handoff для следующего чата

Обновлено: 2026-06-16 (GPU v1 прогнан; план GPU v2)

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
- CLI: `main.mojo` → `info | smoke | train | test-grad`
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

### Mojo (student) — батч + GPU (новое)

**Архитектура train (один путь, sample-by-sample удалён):**

| Модуль | Роль |
|--------|------|
| `buffer.mojo` | `ChainData` — dense `X`/`Y`, доступ по индексу без `sample_input()` в hot path |
| `micro_net_batch.mojo` | `BatchMicroNet` — `train_step_cpu()` (батч STE + MSE) |
| `gpu/device.mojo` | `GpuDevice` — host/device буферы, upload/download |
| `gpu/ternary_matmul.mojo` | CUDA kernels: ternary matmul, SwiGLU, head reduce, scale |
| `gpu/batch_step.mojo` | `train_step_gpu()` — **forward на GPU**, backward STE на CPU |
| `train.mojo` | `_train_epochs_cpu` / `_train_epochs_gpu`, лог `device=` |
| `adamw.mojo` | AdamW на `BatchMicroNet` (host lists) |
| `device.mojo` | `DeviceKind`, `resolve_device_from_env()` |
| `test_batch_grad.mojo` | регрессия градиентов vs старый per-sample path |

Удалено: `micro_net.mojo` (`accumulate_grad`, sample-by-sample).

**Проверено:**

- `make test` — green (`build` 0 warnings + `test-grad` + `smoke`)
- `make test-grad` — `max|grad_diff| < 1e-5`
- `make smoke` / `make smoke-cuda` — MSE падает на synthetic (128 samples)
- RTX 3050 Ti (sm_86): GPU path компилируется и работает при видимом NVIDIA GPU **на этапе compile и run**

**Ограничение GPU v1:** backward и AdamW остаются на **host**; на device только forward (matmul + SwiGLU + head). Полный backward/AdamW на GPU — не сделан.

### Уже прогнано пользователем

- `data/chains/L00_N0000.bin` (14 MB) + `L00_N0000.json`, 4096 samples
- Старый CPU sample-by-sample train: ~4+ мин/epoch 0, прервано (`mse≈4.46e7`)
- **GPU v1 train** (`make train-cuda`, RTX 3050 Ti): 10 epochs за **~59 с**; MSE `44.6M → 9.3M` (падает, до Phase 1 цели далеко)

### Важно: PyTorch ≠ student

- **PyTorch только для teacher.**
- Student — **Mojo**; `CALIBER158_TRAIN_BACKEND=mojo`.

---

## Текущая фаза

**Phase 1**: одна цепочка `L00_N0000` — **дообучить до конца** и оценить качество.

Критерий успеха: holdout MSE < 1e-4 (или < 0.1% от `Var(Y_qwen)`).

---

## Что делать дальше (приоритет)

### 1. Качество Phase 1 — **следующий шаг**

GPU v1 train прогнан (см. выше). Дальше по качеству, не по скорости:

- **Holdout split** + relative MSE vs `Var(Y)`
- Тюнинг: `HIDDEN_DIM` 256/512, больше `EPOCHS`, re-extract с `SAMPLES=100000`
- **Checkpoint export** (см. ниже)

### 2. Checkpoint export (нет в коде)

После успешного train — сохранять gate/up/head ternary, shadow (опц.), α, chain_id → `data/checkpoints/L00_N0000.bin` (формат не спроектирован).

### 3. Метрики качества

- Relative MSE vs `Var(Y)`
- `pred` vs teacher на holdout
- Сравнить CPU-batch vs GPU forward numerics на одном батче (опц.)

### 4. GPU v2 — перенос остатка на device

См. раздел **«План: GPU v2»** ниже. Имеет смысл **после** holdout/checkpoint или параллельно, если упираемся в wall time на 100k samples.

### 5. Phase 2 — batch extract layer 0

- `batch_extract.py`: 4864 цепочек → `L00_N####.bin`
- Параллельный train worker pool
- `data/chains/manifest.jsonl`

### 6. Phase 3+

- Сборка тернарного MLP слоя, 24 слоя FFN, attention — отдельно

### 7. Документация

- **README** устарел: всё ещё `micro_net.mojo`, 3584/~530k chains, нет `DEVICE`/Makefile/GPU
- Обновить layout в README под `micro_net_batch`, `gpu/`, `Makefile`

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
make test-grad         # только сверка градиентов
```

Эквивалент через pixi: `pixi run <task>` (см. `pixi.toml`).

---

## Известные ограничения / техдолг

- README не синхронизирован с кодом (см. выше)
- Нет holdout, нет checkpoint I/O, нет batch extract pipeline
- GPU v1: backward/AdamW на CPU; per-batch upload весов/данных — **план исправления: GPU v2**
- Compile student с GPU kernels требует **видимый NVIDIA GPU** при `mojo build` (sm_86 ок для 3050 Ti)
- `down_proj` (4864→896) не покрыт micro-сетью
- `dataset.mojo`: `sample_input()` остаётся для совместимости, не используется в train hot path

---

## Критерии GPU train — статус

| Критерий | Статус |
|----------|--------|
| Батчевый CPU train + grad regression | ✅ `make test-grad` |
| `CALIBER158_DEVICE`, `DeviceKind`, лог device | ✅ |
| GPU forward (ternary matmul + SwiGLU) | ✅ GPU v1 |
| `make test` green, 0 warnings | ✅ |
| `make train-cuda` 10 epochs на `L00_N0000.bin` | ✅ ~59 с |
| Wall time < 1 мин (4096, 3050 Ti) | ✅ |
| MSE падает на реальном `.bin` | ✅ 44.6M → 9.3M (качество Phase 1 — нет) |
| GPU backward + AdamW + persistent buffers | ❌ см. GPU v2 |
| Checkpoint export | ❌ |
| Holdout | ❌ |
| Teacher `make extract` без регрессии | ✅ (код не менялся) |

---

## План: GPU v2 — backward, AdamW, persistent buffers

Цель: убрать host из hot loop train; тот же numerics/STE, что CPU `train_step_cpu` / `make test-grad`.

**Не делать:** PyTorch для student; второй train path sample-by-sample.

### Что осталось на CPU (GPU v1)

Источник: `gpu/batch_step.mojo` + `adamw.mojo` + `train.mojo`.

| Операция | Где сейчас | Проблема |
|----------|------------|----------|
| Upload `X` batch | `train_step_gpu` каждый батч | PCIe + sync |
| Upload ternary весов | каждый батч | gate/up/head не меняются между quantize внутри step, но shadow меняется после AdamW — нужен device-resident shadow + on-device quantize |
| Forward | GPU ✅ | — |
| Download gate, up, pred | после forward | ломает pipeline |
| MSE + STE backward | CPU loops `b × H × D` | ~14M итераций/батч при B=64, H=128, D=896 |
| `ModelGrads` accumulate | host `List[Float32]` | — |
| AdamW update | `adamw.mojo` host lists | мелко, но копии shadow туда-сюда |
| `BatchMicroNet` weights | host `List` | единственный source of truth на host |

### Целевая архитектура (GPU v2)

```
epoch:
  quantize_ste_kernel(shadow_dev → ternary_dev)   # на device
  for each batch offset:
    forward_kernels(X_dev, ternary_dev) → pred_dev
    loss_backward_kernels(pred_dev, Y_dev, activations) → grads_dev
    adamw_kernel(shadow_dev, grads_dev, m_dev, v_dev)
  # опц.: один download shadow в конце epoch для checkpoint
```

Веса, optimizer state, dataset — **persistent `DeviceBuffer`** на протяжении epoch (или всего train).

### Порядок работ (по impact)

#### Шаг 1. `gpu/buffer_pool.mojo` — persistent device state

Новый модуль; владеет всеми GPU-буферами одного train run:

| Буфер | Shape | Когда alloc |
|-------|-------|-------------|
| `x_dev` | `[n_samples × input_dim]` | load dataset, один upload |
| `y_dev` | `[n_samples]` | load dataset |
| `gate_shadow_dev`, `up_shadow_dev` | `[H × D]` each | init weights |
| `head_shadow_dev` | `[H]` | init |
| `alpha_dev` | scalar | init |
| `gate_tern_dev`, `up_tern_dev`, `head_tern_dev` | ternary | после quantize |
| `gate_act_dev`, `up_act_dev`, `hidden_dev` | `[B × H]` | per batch или ring buffer |
| `pred_dev` | `[B]` | per batch |
| `grad_gate_dev`, `grad_up_dev`, `grad_head_dev` | same as shadow | zero per batch |
| `grad_alpha_dev` | scalar | zero per batch |
| `adam_*_m_dev`, `adam_*_v_dev` | same as params | init once |

API sketch:

- `GpuTrainState.from_chain_data(data, model) -> GpuTrainState`
- `upload_once()` при старте train; `download_shadow()` только для checkpoint/debug

VRAM оценка (4096, D=896, H=128): X ~14 MB + weights/grads/adam ~3×229k×4 B ≈ 2.7 MB × несколько буферов — **< 100 MB**, влезает в 3050 Ti 4 GB.

#### Шаг 2. `gpu/quantize.mojo` — STE quantize на device

Kernel: `shadow[i] → ternary[i] ∈ {-1,0,1}` по `CALIBER158_TERNARY_THRESHOLD`.

- Вызывать **перед forward каждого batch** (как `sync_ternary()` сейчас)
- Backward по-прежнему пишет в **shadow grads** (STE: градиент не идёт в ternary)

#### Шаг 3. `gpu/backward.mojo` — STE backward kernels

Математика 1:1 с `batch_step.mojo:99–131` / `micro_net_batch.train_step_cpu`:

```
err[b] = pred[b] - y[b]
loss += err[b]² / B
y_tern[b] = pred[b] / alpha
dL_dout = 2 * err[b] / B
grad_alpha += dL_dout * y_tern[b]

dL_dy_tern = dL_dout * alpha
h[j] = silu(gate[j]) * up[j]
grad_head[j] += dL_dy_tern * h[j]
dL_dh = dL_dy_tern * head_tern[j]
dL_dgate = dL_dh * up[j] * silu'(gate[j])
dL_dup = dL_dh * silu(gate[j])
grad_gate[j,i] += dL_dgate * x[i]
grad_up[j,i]   += dL_dup * x[i]
```

Предлагаемые kernels:

| Kernel | Параллелизм | Заметки |
|--------|-------------|---------|
| `mse_loss_kernel` | 1 thread / sample | пишет `err_dev`, reducе loss (atomic или host после small download) |
| `backward_head_kernel` | 1 thread / (b,j) | grad_head[j] atomic add по batch |
| `backward_gate_up_kernel` | 1 thread / (b,j,i) или tile | outer product; **atomic add** в `grad_gate`, `grad_up` по строке j |

Альтернатива без atomics на gate/up: сначала per-sample grad в temp `[B,H,D]`, потом `reduce_sum` по B — больше VRAM, проще корректность.

**Сверка:** расширить `test_batch_grad.mojo` — CPU batch vs GPU v2 backward на одном батче, `max|diff| < 1e-5`.

#### Шаг 4. `gpu/adamw.mojo` — update на device

Порт `_adamw_update_list` / `_adamw_update_scalar` из `adamw.mojo`:

- один element-wise kernel на `gate_shadow`, `up_shadow`, `head_shadow`
- scalar kernel для `alpha`
- `timestep` на host (int), bias_corr1/2 передавать как uniform

Host `adamw.mojo` оставить для **CPU path** (`train_step_cpu`); GPU path вызывает только `gpu/adamw.mojo`.

#### Шаг 5. `gpu/batch_step.mojo` v2 — собрать pipeline

Заменить `train_step_gpu`:

1. ~~upload x, weights~~ → slice `x_dev[offset:offset+B×D]`, ternary уже на device
2. forward kernels (как сейчас, без download промежуточных)
3. backward kernels → `grad_*_dev`
4. `adamw_apply(dev state)`
5. **без** `download_f32` gate/up/pred в hot path

Опционально: fused `gate+up` matmul (один launch, два выхода) — шаг 6.

#### Шаг 6. `train.mojo` — lifecycle

- `_train_epochs_gpu`: создать `GpuTrainState` один раз, переиспользовать
- `BatchMicroNet` на host: либо mirror для checkpoint, либо shadow **только на GPU** + download в конце epoch
- Лог: `device=cuda backend=mojo v2` (или флаг env `CALIBER158_GPU_FULL=1` до стабилизации)

#### Шаг 7. Убрать лишние sync

- Сейчас: `synchronize()` после каждого kernel и upload — минимизировать до границ batch/epoch
- Loss для лога: reducе на GPU или download scalar loss per epoch

#### Шаг 8. Опционально (v2.1, после корректности)

- Fused gate+up ternary matmul
- Tensor-core path для dense shadow matmul (если уйдём от ternary в backward inner loop)
- Pinned host buffers для initial dataset upload
- CI без GPU: compile-time guard / отдельный target без `gpu/*.mojo` (сейчас `mojo build` требует NVIDIA)

### Файлы — чеклист

| Действие | Файл |
|----------|------|
| persistent device buffers | `src/chain/gpu/buffer_pool.mojo` **новый** |
| on-device quantize | `src/chain/gpu/quantize.mojo` **новый** |
| backward kernels | `src/chain/gpu/backward.mojo` **новый** |
| AdamW on device | `src/chain/gpu/adamw.mojo` **новый** |
| v2 train step | `src/chain/gpu/batch_step.mojo` **переписать** |
| epoch lifecycle | `src/chain/train.mojo` **изменить** |
| расширить grad regression | `src/chain/test_batch_grad.mojo` **изменить** |
| upload/download helpers | `src/chain/gpu/device.mojo` **расширить** (slice, zero, reduce) |
| host AdamW | `src/chain/adamw.mojo` — оставить для CPU path |

### Критерии готовности GPU v2

- [ ] `make test-grad` green; GPU backward сверка с CPU batch
- [ ] `make train-cuda` на `L00_N0000.bin`: **10 epochs < 15 с** (3050 Ti, ориентир)
- [ ] MSE curve ≈ GPU v1 (допуск на float reorder: relative diff loss < 1e-4 per epoch)
- [ ] Нет per-batch upload весов/`X` (профилировать: один upload dataset at start)
- [ ] `make test` green, 0 warnings
- [ ] Host не участвует в backward/AdamW hot loop

### Что не входит в GPU v2

- Holdout, checkpoint I/O, batch extract
- PyTorch student
- Полная autodiff / MAX Graph — только hand-written kernels как в v1

---

## Структура файлов (актуальная)

```
Caliber158/
├── Makefile              # make help | test | train-cuda | …
├── .env / .env.example
├── main.mojo
├── pixi.toml             # mojo + max
├── src/chain/
│   ├── buffer.mojo
│   ├── micro_net_batch.mojo
│   ├── train.mojo, adamw.mojo, dataset.mojo, env.mojo, device.mojo
│   ├── ternary.mojo, grads.mojo, rng.mojo
│   ├── test_batch_grad.mojo
│   └── gpu/
│       ├── device.mojo
│       ├── ternary_matmul.mojo
│       ├── batch_step.mojo      # v1 forward GPU; v2 = full pipeline
│       ├── buffer_pool.mojo     # v2 TODO
│       ├── quantize.mojo        # v2 TODO
│       ├── backward.mojo        # v2 TODO
│       └── adamw.mojo           # v2 TODO
├── python/
├── scripts/
├── docs/
├── data/chains/          # L00_N0000.bin + .json (generated)
└── models/huggingface/
```
