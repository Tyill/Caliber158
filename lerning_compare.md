# Сравнение прогонов обучения — L00_N0000

Обновлено: 2026-06-17 (100k; FP32 v0 + rel_decay → Phase 1 ✅; ternary ❌)

**Датасет (актуальный):** `data/chains/L00_N0000.bin` — **100 000 samples** (342 MB), re-extract 2026-06-17  
**Датасет (история):** прогоны #1–#T15 и §4k-holdout — **4096 samples** (14 MB)  
**Teacher:** Qwen2.5-0.5B, layer=0, neuron=0  
**Student:** Mojo v0/v1 / Torch v0/v1/v1b/exact  
**GPU:** NVIDIA RTX 3050 Ti  

## Масштаб teacher (Y) — 100k

| Метрика | Значение |
|---------|----------|
| `Y` min / max | ≈ −1.81 / 1.57 |
| `Var(Y)` all | **0.0343** |
| `Var(Y)` train (90k) | 0.0342 |
| `Var(Y)` holdout (10k) | 0.0343 |
| Split @ seed 42 | **90 000 train / 10 000 holdout** |

| **Критерий Phase 1:** holdout MSE < 1e-4 **или** `rel_holdout = MSE / Var(Y) < 0.001` (0.1% дисперсии).  
**Статус (100k, Torch):** ✅ **FP32 v0 H=128 + rel_decay** (#100k-h) `rel=0.00073`; ❌ **ternary** ≈1.0.

---

## Прогоны (хронология)

### 1. CPU sample-by-sample (до GPU refactor)

| Параметр | Значение |
|----------|----------|
| `HIDDEN_DIM` | 128 |
| `EPOCHS` | 10 |
| `LR` | 0.001 |
| Device | CPU (старый loop) |

| Результат | |
|-----------|--|
| Epoch 0 MSE | **4.46×10⁷** |
| Статус | прервано (~4+ мин на epoch 0) |
| Holdout | нет |

---

### 2. GPU, H=128, 10 epochs

| Параметр | Значение |
|----------|----------|
| `HIDDEN_DIM` | 128 |
| `EPOCHS` | 10 |
| `LR` | 0.001 |
| `WEIGHT_DECAY` | 0.01 |
| Params | 229k |

| Epoch | MSE |
|-------|-----|
| 0 | 4.46×10⁷ |
| 9 | **9.28×10⁶** |

| | |
|--|--|
| Wall time | ~59 с |
| `rel = MSE/Var(Y)` | ~2.8×10⁸ |
| Holdout | нет |

---

### 3. GPU, H=256, 50 epochs (тюнинг гиперпараметров)

| Параметр | Значение |
|----------|----------|
| `HIDDEN_DIM` | 256 |
| `EPOCHS` | 50 |
| `LR` | 0.003 |
| `WEIGHT_DECAY` | 0.001 |
| Params | 459k |

| Epoch | MSE |
|-------|-----|
| 0 | 8.31×10⁷ |
| 23 | 0.0334 |
| 49 | **0.0329** |

| | |
|--|--|
| Wall time | ~17 с |
| `rel = MSE/Var(Y)` | **~0.99** (плато ≈ предсказание среднего) |
| Holdout | нет |

---

### 4. GPU, H=512, 100 epochs

| Параметр | Значение |
|----------|----------|
| `HIDDEN_DIM` | 512 |
| `EPOCHS` | 100 |
| `LR` | 0.003 |
| `WEIGHT_DECAY` | 0.001 |
| Params | 918k |

| Epoch | MSE |
|-------|-----|
| 0 | 1.92×10⁸ |
| 27 | 0.0334 |
| 58 (лучший) | **0.0324** |
| 99 | 0.0325 |

| | |
|--|--|
| Wall time | ~50 с |
| `rel = MSE/Var(Y)` | **~0.98** |
| Holdout | нет |
| Вывод | 512 hidden **не пробил** потолок vs 256 |

---

### 5. GPU, H=512, 30 epochs + holdout ✅ (актуальный)

| Параметр | Значение |
|----------|----------|
| `HIDDEN_DIM` | 512 |
| `EPOCHS` | 30 |
| `LR` | 0.003 |
| `WEIGHT_DECAY` | 0.001 |
| `HOLDOUT_FRACTION` | 0.1 (`SEED=42`) |
| Split | **3687 train / 409 holdout** |

| Epoch | train_mse | holdout_mse | rel_holdout |
|-------|-----------|-------------|-------------|
| 0 | 1.94×10⁸ | 1.41×10⁸ | 3.81×10⁹ |
| 15 | 9.83×10³ | 5.58×10³ | 1.50×10⁵ |
| 23 | 0.573 | 0.316 | 8.51 |
| 27 | 0.0334 | 0.0400 | 1.08 |
| 29 | **0.0321** | **0.0384** | **1.04** |

| | |
|--|--|
| Wall time | ~43–47 с |
| Финал train `rel` | ~0.97 |
| Финал holdout `rel` | **1.04** |
| Phase 1 target `rel` | 0.001 |
| Отставание от цели | **~1000×** |

---

### 5b. GPU, H=512, 30 epochs + holdout — FP32 v0 diagnostic ✅

| Параметр | Значение |
|----------|----------|
| `HIDDEN_DIM` | 512 |
| `EPOCHS` | 30 |
| `LR` | 0.003 |
| `WEIGHT_DECAY` | 0.001 |
| `HOLDOUT_FRACTION` | 0.1 (`SEED=42`) |
| `CALIBER158_QUANTIZE` | **0** (shadow FP32, без STE ternary) |
| Split | **3687 train / 409 holdout** |

| Epoch | train_mse | holdout_mse | rel_holdout |
|-------|-----------|-------------|-------------|
| 0 | 3.19 | 0.470 | 12.7 |
| 5 | 3.14 | 0.140 | 3.76 |
| 16 | 11.3 | 0.0467 | 1.26 |
| 18 | 0.158 | 0.0372 | 1.00 |
| 29 | **0.0327** | **0.0372** | **1.004** |

| | |
|--|--|
| Wall time | ~36 с |
| Финал holdout `rel` | **1.004** |
| vs ternary #5 | **≈ то же плато** (1.04 vs 1.004) |

**Вывод:** без ternary та же shallow v0 arch → тот же `rel` на ep ~30. FP32 исключил STE; позже — **bottleneck ≠ teacher** (§ Диагностика).

---

### 6. GPU, H=512, 30 epochs + holdout — v1 ternary ✅

| Параметр | Значение |
|----------|----------|
| `CALIBER158_ARCH` | **v1** (dual SwiGLU + residual) |
| `HIDDEN_DIM` | 512 |
| `EPOCHS` | 30 |
| `LR` | 0.003 |
| `WEIGHT_DECAY` | 0.001 |
| `HOLDOUT_FRACTION` | 0.1 (`SEED=42`) |
| Params | **1 442 305** (+524k block2) |
| Split | **3687 train / 409 holdout** |

| Epoch | train_mse | holdout_mse | rel_holdout |
|-------|-----------|-------------|-------------|
| 0 | 1.94×10⁸ | 1.41×10⁸ | 3.81×10⁹ |
| 23 | 0.765 | 0.353 | 9.51 |
| 27 | 0.0346 | 0.0404 | 1.09 |
| 29 | **0.0321** | **0.0382** | **1.03** |

| | |
|--|--|
| Wall time | ~63 с |
| Финал holdout `rel` | **1.03** |
| vs v0 ternary #5 | **≈ то же плато** (1.04 vs 1.03) |

**Вывод:** v1 @ 30 ep не улучшил holdout vs v0 — позже выяснилось: bottleneck H ≠ teacher form (§ Диагностика).

---

### 6b. GPU, H=512, 30 epochs + holdout — FP32 v1 diagnostic ✅

| Параметр | Значение |
|----------|----------|
| `CALIBER158_ARCH` | **v1** |
| `CALIBER158_QUANTIZE` | **0** (shadow FP32, без STE ternary) |
| `HIDDEN_DIM` | 512 |
| `EPOCHS` | 30 |
| `LR` | 0.003 |
| `WEIGHT_DECAY` | 0.001 |
| `HOLDOUT_FRACTION` | 0.1 (`SEED=42`) |
| Params | **1 442 305** |
| Split | **3687 train / 409 holdout** |

| Epoch | train_mse | holdout_mse | rel_holdout |
|-------|-----------|-------------|-------------|
| 0 | 3.19 | 0.470 | 12.7 |
| 16 | 7.06 | 0.0405 | 1.09 |
| 18 | 0.070 | 0.0371 | 1.00 |
| 29 | **0.0327** | **0.0372** | **1.004** |

| | |
|--|--|
| Wall time | ~56 с |
| Финал holdout `rel` | **1.004** |
| vs ternary v1 #6 | **≈ то же плато** (1.03 vs 1.004) |
| vs FP32 v0 #5b | **≈ то же** (1.004 vs 1.004) |

**Заметка:** эпохи 9–15 — нестабильность (train_mse до ~2000), затем сходимость к плато. LR/WD те же, что v0/v1 ternary.

**Вывод:** FP32 v1 не лучше FP32 v0 → второй SwiGLU-block @ H=512 **не даёт** выигрыша. Узкое место не STE; позже выяснилось — **bottleneck H vs teacher form** (см. § Диагностика).

---

## Сводная таблица (финальные метрики)

| Прогон | H | Epochs | Train MSE | Holdout MSE | rel_holdout | Время |
|--------|---|--------|-----------|-------------|-------------|-------|
| CPU v0 | 128 | 0 (прерван) | 4.46×10⁷ | — | — | >4 мин/ep |
| GPU | 128 | 10 | 9.28×10⁶ | — | — | ~59 с |
| GPU | 256 | 50 | 0.0329 | — | ~0.99* | ~17 с |
| GPU | 512 | 100 | 0.0325 | — | ~0.98* | ~50 с |
| **GPU + holdout** | **512** | **30** | **0.0321** | **0.0384** | **1.04** | **~45 с** |
| **FP32 v0 + holdout** | **512** | **30** | **0.0327** | **0.0372** | **1.004** | **~36 с** |
| **v1 ternary + holdout** | **512** | **30** | **0.0321** | **0.0382** | **1.03** | **~63 с** |
| **FP32 v1 + holdout** | **512** | **30** | **0.0327** | **0.0372** | **1.004** | **~56 с** |
| **Torch #T1 v0** | **512** | **30** | **0.0321** | **0.0383** | **1.033** | **~7 с** |
| **Torch #T2 v1** | **512** | **30** | **0.0321** | **0.0383** | **1.033** | **~7 с** |
| **Torch #T3 FP32 v1** | **512** | **30** | **0.0328** | **0.0373** | **1.004** | **~6 с** |

\* train MSE / Var(Y), holdout не измерялся

---

## Выводы

1. **GPU train** дал ускорение с минут до ~45 с на 30 epochs (512 hidden).
2. **Рост `HIDDEN_DIM`** 128 → 256 дал огромный скачок; 256 → 512 почти не помог.
3. **Holdout ≈ train на ep ~30** (`rel ~ 1.0`) — **ошибочно** трактовалось как underfit (см. § «Диагностика Phase 1» ниже). При длинном train wide-модель: **train→0, holdout≈1** → memorization без generalization.
4. **FP32 diagnostic (#5b):** `rel_holdout ≈ 1.004` — как ternary → узкое место **не STE**, а **постановка student vs teacher** (bottleneck H).
5. **v1 ternary (#6):** `rel_holdout ≈ 1.03` @ 1.44M params — **≈ v0**; block2 не помог за 30 ep.
6. **FP32 v1 (#6b):** `rel_holdout ≈ 1.004` — **≈ FP32 v0 и ≈ ternary v1**; глубина v1 не даёт gain.
7. **Torch sweep #T4–#T10:** 50 ep, H↑768, v1b skip — **без gain** или хуже (skip rel 4–7300).
8. **100k (2026-06-17):** FP32 v0 H=128 + **rel_decay** → **rel=0.00073** (Phase 1 ✅); ternary → **rel≈1**. STE — блокер production path. См. § «100k re-extract», § «rel_decay».

---

## Команды для воспроизведения

```bash
make train-cuda
# env: CALIBER158_HIDDEN_DIM=512, EPOCHS=30, LR=0.003,
#      WEIGHT_DECAY=0.001, HOLDOUT_FRACTION=0.1, SEED=42

# FP32 v0 diagnostic:
CALIBER158_QUANTIZE=0 make train-cuda

# FP32 v1 diagnostic:
CALIBER158_ARCH=v1 CALIBER158_QUANTIZE=0 make train-fp32-v1-cuda

# v1 ternary:
CALIBER158_ARCH=v1 CALIBER158_HIDDEN_DIM=512 CALIBER158_LR=0.003 \
  CALIBER158_WEIGHT_DECAY=0.001 CALIBER158_EPOCHS=30 make train-cuda
make test-grad-v1 test-grad-gpu-v1
```

---

## Torch student prototype (#T1–#T3)

**Backend:** PyTorch (`make train-torch`), parity gate: `make test-torch-parity`  
**Датасет / split:** тот же `L00_N0000.bin`, holdout @ `SEED=42` (3687 / 409)  
**Паритет:** holdout indices + 1-batch loss vs Mojo CPU (`tests/test_holdout_golden.py`, `tests/test_forward_loss_mojo.py`)

### #T1 — v0 ternary + holdout ✅

| Параметр | Значение |
|----------|----------|
| `CALIBER158_ARCH` | v0 |
| `HIDDEN_DIM` | 512 |
| `EPOCHS` | 30 |
| `LR` / `WEIGHT_DECAY` | 0.003 / 0.001 |
| Params | 918 017 |
| Device | CUDA (Torch) |

| Epoch 29 | train_mse | holdout_mse | rel_holdout |
|----------|-----------|-------------|-------------|
| | 0.0321 | 0.0383 | **1.033** |

| | |
|--|--|
| Wall time | ~7 с |
| vs Mojo #5 holdout | **≈ то же** (rel 1.04 vs 1.033) |

### #T2 — v1 ternary + holdout ✅

| Параметр | Значение |
|----------|----------|
| `CALIBER158_ARCH` | v1 |
| Params | 1 442 305 |

| Epoch 29 | train_mse | holdout_mse | rel_holdout |
|----------|-----------|-------------|-------------|
| | 0.0321 | 0.0383 | **1.033** |

| | |
|--|--|
| Wall time | ~7 с |
| vs Mojo #6 | **≈ то же** (rel 1.03) |

### #T3 — FP32 v1 diagnostic + holdout ✅

| Параметр | Значение |
|----------|----------|
| `CALIBER158_ARCH` | v1 |
| `CALIBER158_QUANTIZE` | 0 |

| Epoch 29 | train_mse | holdout_mse | rel_holdout |
|----------|-----------|-------------|-------------|
| | 0.0328 | 0.0373 | **1.004** |

| | |
|--|--|
| Wall time | ~6 с |
| vs Mojo #6b | **≈ то же** (rel 1.004) |

**Вывод:** Torch-prototype воспроизводит Mojo holdout/split и `rel` на ep ~30. Диагностика 2026-06-17 уточнила: это не underfit, а arch mismatch + memorization при длинном train (§ Диагностика).

```bash
make test-torch-parity   # gate: holdout + 1-batch loss vs Mojo
make train-torch         # env как train-cuda
CALIBER158_ARCH=v1 CALIBER158_HIDDEN_DIM=512 CALIBER158_EPOCHS=30 \
  CALIBER158_LR=0.003 CALIBER158_WEIGHT_DECAY=0.001 make train-torch
```

---

## Torch Phase 1 experiments (#T4–#T10, 2026-06-17)

**Цель:** v1b (linear skip), 50 ep v1, H↑ — быстрый A/B в Torch до порта в Mojo.

**v1b spec (Torch):** `out = α·head(h1) + β·(w_res·x)` поверх v1; block2 zero-init как v1; skip `w_res` — **FP32 dense** (ternary skip давал rel ~7000+); `w_res=0`, `β=1` at init (forward = v1, градиенты в skip идут).

### #T4 — v1, 50 ep, H=512 ✅

| Параметр | Значение |
|----------|----------|
| `CALIBER158_ARCH` | v1 |
| `HIDDEN_DIM` / `EPOCHS` | 512 / **50** |
| Params | 1 442 305 |

| Epoch 49 | train_mse | holdout_mse | rel_holdout |
|----------|-----------|-------------|-------------|
| | 0.0321 | 0.0384 | **1.034** |

| | |
|--|--|
| Wall time | ~10 с |
| vs #T2 (30 ep) | **≈ то же плато** (1.033 vs 1.034) |

**Вывод:** 50 ep не улучшают holdout vs 30 ep — модель уже на плато к ep ~27.

### #T5 — v1b ternary skip, β=0 init (dead path) ✅

| Init | `w_res=0`, `β=0` |
|------|------------------|
| Epoch 49 rel | **1.034** (бит-в-бит #T4) |

**Вывод:** при `β=0` градиенты в skip не текут — v1b ≡ v1. Init исправлен на `β=1`.

### #T6 — v1b ternary skip, β=1, H=512 ❌

| Epoch 49 | train_mse | holdout_mse | rel_holdout |
|----------|-----------|-------------|-------------|
| | 276 | 271 | **7302** |

**Вывод:** ternary linear skip **ломает** обучение — rel >> 1.

### #T7 — v1, 50 ep, H=768 ✅

| Параметр | Значение |
|----------|----------|
| `HIDDEN_DIM` | **768** |
| `BATCH_SIZE` | 32 |
| Params | 2 556 673 |

| Epoch 49 | train_mse | holdout_mse | rel_holdout |
|----------|-----------|-------------|-------------|
| | 0.0328 | 0.0368 | **0.993** |

| | |
|--|--|
| Wall time | ~19 с |
| vs H=512 #T4 | **чуть лучше**, но всё ещё плато ~1.0 |

### #T8 — v1b ternary skip, β=1, H=768 ❌

| Epoch 49 rel | **1054** |

### #T9 — v1b FP32 skip, H=512 ✅

| Skip | FP32 `F.linear(x, w_res)`, `β=1` init |
|------|---------------------------------------|
| Epoch 49 rel | **11.94** |

### #T10 — v1b FP32 skip, H=768 ✅

| Epoch 49 | train_mse | holdout_mse | rel_holdout |
|----------|-----------|-------------|-------------|
| | 0.147 | 0.145 | **3.90** |

**Вывод v1b:** linear skip (ternary или FP32) **ухудшает** holdout vs v1 @ H=512/768. Skip конкурирует с nonlinear path при том же LR/WD; Phase 1 quality не улучшен.

---

## Сводная таблица (#T4–#T10, дополнение)

| Прогон | Arch | H | Epochs | rel_holdout | Время |
|--------|------|---|--------|-------------|-------|
| **#T4 v1 50ep** | v1 | 512 | 50 | **1.034** | ~10 с |
| #T5 v1b dead skip | v1b | 512 | 50 | 1.034 | ~10 с |
| #T6 v1b tern skip | v1b | 512 | 50 | 7302 | ~10 с |
| **#T7 v1 H↑** | v1 | 768 | 50 | **0.993** | ~19 с |
| #T8 v1b tern H↑ | v1b | 768 | 50 | 1054 | ~21 с |
| #T9 v1b fp32 skip | v1b | 512 | 50 | 11.94 | ~11 с |
| #T10 v1b fp32 H↑ | v1b | 768 | 50 | 3.90 | ~20 с |

---

## Выводы Phase 1 Torch sweep (2026-06-17)

1. **50 ep v1 @ H=512** — без gain vs 30 ep; сходимость к плато к ep ~27.
2. **H↑ 768** — marginal (`rel ≈ 0.99` vs 1.03); underfit сохраняется.
3. **v1b linear skip** — не помогает (ternary catastrophic, FP32 skip rel 12–4); **не портируем в Mojo** без новой гипотезы (отдельный LR skip, init block2 ≠ zero, …).
4. **Следующий R&D (до диагностики):** init block2, LR schedule — **отменено** после § Диагностика; v1b **не портируем** в Mojo.

```bash
# 50 ep v1 baseline
CALIBER158_ARCH=v1 CALIBER158_HIDDEN_DIM=512 CALIBER158_EPOCHS=50 \
  CALIBER158_LR=0.003 CALIBER158_WEIGHT_DECAY=0.001 make train-torch

# H↑ v1
CALIBER158_ARCH=v1 CALIBER158_HIDDEN_DIM=768 CALIBER158_BATCH_SIZE=32 \
  CALIBER158_EPOCHS=50 make train-torch
```

---

## Torch block2 init / LR schedule (#T11–#T15, 2026-06-17, остановлено)

**Env (Torch-only):** `CALIBER158_BLOCK2_INIT=lcg|zero`, `CALIBER158_LR_SCHEDULE=cosine|none`, `CALIBER158_BLOCK2_INIT_SCALE` (опц.).

| Прогон | block2 | schedule | H | rel_holdout | Итог |
|--------|--------|----------|---|-------------|------|
| #T11 | lcg @ 0.1 | none | 512 | **~10²¹** | взрыв |
| #T12 | zero | cosine | 512 | **1.033** | ≈ baseline |
| #T13 | lcg @ 0.1 | cosine | 512 | **~10²¹** | взрыв |
| #T14 | lcg @ 0.1 | none | 768 | **~10²²** | взрыв |
| #T15 | lcg @ 0.1 | cosine | 768 | **~10²²** | взрыв |

**Вывод:** block2 LCG @ `INIT_SCALE=0.1` ломает residual (`h1 = h0 + SwiGLU2(h0)`); cosine LR alone — без эффекта. Прогоны с `BLOCK2_INIT_SCALE=0.01` **не завершены** (остановлено).

---

## Диагностика Phase 1 (2026-06-17) — ключевой раздел

**Метод:** Torch ad-hoc scripts на `L00_N0000.bin`, holdout @ seed 42 (3687 / 409). Teacher weights из Qwen2.5-0.5B L0/N0.

### D1. Данные и split ✅

| Проверка | Результат |
|----------|-----------|
| `Var(Y)` | 0.0333 (2553 unique Y @ 1e-4) |
| Teacher replay `Y' = SiLU(wg·x)·(wu·x)` | `max|Y'−Y| ≈ 2.4×10⁻⁷`, MSE ≈ 3×10⁻¹⁶ |
| X | `N(0,1)`, std≈1 по dim |

**Вывод:** `.bin` корректен; проблема не в extract/split.

### D2. Baselines (holdout)

| Модель | rel_holdout |
|--------|-------------|
| predict mean(train Y) | **1.001** |
| linear OLS (896+1) | **1.244** (хуже mean) |

### D3. Teacher-shaped vs student v0 (FP32, AdamW)

**Teacher (один neuron):** `y = SiLU(w_gate·x) · (w_up·x)` — **2×896** params, **без** hidden bottleneck H.

**Student v0/v1:** `D → H → … → 1` — другая parameterization.

| Эксперимент | train_mse | holdout rel | Комментарий |
|-------------|-----------|-------------|-------------|
| Exact SwiGLU + **teacher weights**, 0 train | — | **≈ 0** | target **достижим** |
| Exact SwiGLU, random init, 500 ep, **train split** | ↓ 0.005 | **6.22** | holdout **ухудшается** |
| Wide SwiGLU H=896, 500 ep, train split | **≈ 0** | **1.07** | **memorization** train, holdout ≈ mean |
| Wide SwiGLU H=896, 500 ep, train split | — | — | full-dataset rel ≈ **0.12** |
| FP32 Wide H=512–1024, 100 ep | — | **~1.0** | upper bound внутри v0 arch |

### D4. Исправление диагноза (уточнено после 100k)

| Вывод | Контекст | Статус |
|-------|----------|--------|
| «Underfit @ ep ~30» | train≈holdout≈Var(Y) на **4k** ternary | Частично: ранняя стадия / STE plateau |
| «Arch mismatch — bottleneck ≠ teacher» | **4k**, FP32 wide @ 500 ep → holdout≈1 | Верно для **мало данных** |
| «Ternary не bottleneck» | FP32 #5b @ **4k** rel≈1.004 | Верно для **4k** |
| **100k + FP32 v0 H=128** | **rel≈0.005** @ 20 ep constant LR | STE off — bottleneck **достаточен** |
| **100k + FP32 v0 + rel_decay** | **rel≈0.00073** (#100k-h) | **Phase 1 ✅** (Torch FP32) |
| **100k + ternary v0** | **rel≈1.002** | **STE — блокер** production path |
| **arch exact FP32** | **rel≈0.23** (#100k-l) | teacher-form **хуже** v0 bottleneck @ random init |

**Критерий Phase 1 (`rel_holdout < 0.001`):**  
- **FP32 v0 H=128 + rel_decay @ 100k:** ✅ **rel=0.00073** (#100k-h).  
- **Ternary (v0 / exact):** ❌ rel≈1.0 — rel_decay **не включается** (rel не < 0.01).

### D5. Следующие шаги

1. **Ternary production:** warm-start от FP32 v0 → quantize → fine-tune; grad clip; Mojo @ 100k.
2. **Порт rel_decay** в Mojo train (Torch-only: `CALIBER158_LR_SCHEDULE=rel_decay`).
3. **Не приоритет:** arch `exact` (FP32 rel≈0.23, ternary ≈0.95 — хуже v0).

```bash
# teacher replay sanity (из python/, нужен Qwen в cache)
pixi run python -c "
import sys; sys.path.insert(0,'python')
from student.dataset import read_dataset
from extract_chain import load_qwen_weights, swiglu_chain_numpy
from env_config import resolve_teacher_device
import numpy as np
d = read_dataset('data/chains/L00_N0000.bin')
gw, uw, *_ = load_qwen_weights('Qwen/Qwen2.5-0.5B', 0, 0, resolve_teacher_device())
y = swiglu_chain_numpy(gw, uw, d.x)
print('max err', np.max(np.abs(y - d.y)))
"
```

---

## 100k re-extract + v0 sweep (Torch, 2026-06-17)

**Re-extract:** `CALIBER158_SAMPLES=100000 make extract` → `L00_N0000.bin` (342 MB), `n_samples=100000` в `.json`.

**Общие env (если не указано иное):** `arch=v0`, `H=128`, `EPOCHS=20`, `LR=0.0003`, holdout 10% @ seed 42, Torch CUDA.

### #100k-a — v0 ternary, H=512, LR=0.003 (прервано ~ep 20)

| | |
|--|--|
| Params | 918 017 |
| ep 6–8 rel | ~1.002 |
| ep 20 | train_mse **570**, rel **1.003** (train разнос) |

**Вывод:** LR=0.003 + 100k → нестабильность после плато.

### #100k-b — exact ternary (partial, прервано)

| ep 1–7 rel | ~1.003 |
| ep 8+ | осцилляции, ep 28 rel **35** (partial log) |

**Вывод:** exact ternary на 100k — то же плато ~1, без gain vs v0.

### #100k-c — v0 ternary baseline ✅

| Параметр | Значение |
|----------|----------|
| `HIDDEN_DIM` | 128 |
| `LR` / `WEIGHT_DECAY` | 0.0003 / **0.001** |
| Params | 229 505 |

| Epoch | train_mse | holdout_mse | rel_holdout |
|-------|-----------|-------------|-------------|
| 6–8 | ~0.0343 | ~0.0343 | **~1.002** ← best |
| 15 | 0.0364 | 0.0378 | 1.103 |
| 19 | 0.0469 | 0.0426 | **1.242** |

| | |
|--|--|
| Wall time | ~37 с |
| vs 4k ternary #T1 | **то же плато** ~1.0 |

### #100k-d — v0 ternary, WD=0.01 ✅

| `WEIGHT_DECAY` | **0.01** |

| Epoch | rel_holdout |
|-------|-------------|
| 6–8 | **~1.002** |
| 17 | **2.01** (всплеск) |
| 19 | **1.440** |

**Вывод:** WD×10 **не улучшает** плато; после ep 10 **хуже** baseline.

### #100k-e — FP32 v0 control ✅

| Параметр | Значение |
|----------|----------|
| `CALIBER158_QUANTIZE` | **0** (FP32 shadow) |
| `LR` / `WEIGHT_DECAY` | 0.0003 / 0.001 |
| Params | 229 505 |

| Epoch | train_mse | holdout_mse | rel_holdout |
|-------|-----------|-------------|-------------|
| 0 | 0.240 | 0.0346 | 1.009 |
| 4 | 0.0027 | 0.0018 | **0.054** |
| 8 | 0.00020 | 0.00024 | **0.0069** |
| 17 | 0.00013 | 0.00013 | **0.0039** ← best |
| 19 | 0.00012 | 0.00017 | **0.0049** |

| | |
|--|--|
| Wall time | ~27 с |
| vs #100k-c ternary | **~200× лучше** holdout @ ep 19 |
| vs Phase 1 target 0.001 | **~5×** до цели |

**Вывод:** на 100k **STE ternary — главный блокер** при LR=0.0003; FP32 v0 H=128 **стабильно** учится. **Phase 1 ✅** с `rel_decay` (#100k-h, `rel=0.00073`).

### #100k-f — v0 ternary, LR=1e-4 ✅

| `LR` | **0.0001** |

| Epoch | rel_holdout |
|-------|-------------|
| 11–14 | **~1.0015** (плато) |
| 19 | **1.009** |

**Вывод:** LR↓ **не пробивает** ternary platо ~1.0.

### #100k-g — FP32 v0, H=256 ✅

| `HIDDEN_DIM` | **256** |
| Params | 459 009 |

| | H=128 (#100k-e) | H=256 |
|--|-----------------|-------|
| Best rel | **0.0039** ep 17 | 0.0046 ep 18 |
| Final ep 19 | 0.0049 | **0.0041** |

**Вывод:** H↑ marginal; constant LR=3e-4 **не закрывает** 0.001 без rel_decay.

---

## LR schedule `rel_decay` (Torch-only, 2026-06-17)

**Env:** `CALIBER158_LR_SCHEDULE=rel_decay`  
**Логика:** старт `CALIBER158_LR` (phase 1); когда `rel_holdout < CALIBER158_LR_REL_THRESHOLD` (default **0.01**) — один раз LR → `CALIBER158_LR_MIN` (phase 2).

```bash
CALIBER158_LR_SCHEDULE=rel_decay \
  CALIBER158_LR=0.0003 CALIBER158_LR_MIN=0.0001 \
  CALIBER158_LR_REL_THRESHOLD=0.01 make train-torch
```

### #100k-h — FP32 v0 H=128 + rel_decay ✅ **Phase 1**

| Параметр | Значение |
|----------|----------|
| `LR` → `LR_MIN` | **0.0003 → 0.0001** |
| threshold | 0.01 |
| Params | 229 505 |

| Epoch | rel_holdout | LR |
|-------|-------------|-----|
| 6 | **0.0085** | 0.0003 → **decay** |
| 8 | **0.00122** | 0.0001 |
| 9 | **0.00094** | 0.0001 ← **< 0.001** |
| **19** | **0.00073** | 0.0001 |

| | |
|--|--|
| Wall time | ~27 с |
| vs #100k-e constant LR | **6.7× лучше** (0.0049 → 0.00073) |
| **Phase 1 target** | **✅ достигнут** (FP32 shadow) |

**Вывод:** не «LR=0.003 сразу», а **3e-4 до rel<0.01**, потом **1e-4** fine-tune.

### #100k-i — FP32 v0 + rel_decay, LR=0.003 ❌

| Phase1 LR | **0.003** (слишком высокий) |

| Best rel | ~0.034 ep 10 |
| Final | **0.048** |
| lr_decay | **не сработал** (rel не < 0.01) |

### #100k-j — ternary v0 H=128 + rel_decay ❌

| Best rel | ~1.002 ep 6–8 |
| Final | **1.242** |
| lr_decay | **не сработал** |

### #100k-k — ternary `arch=exact` + rel_decay ❌

| Params | **1 793** (2×896 + α) |

| Best rel | **~0.947** ep 11 (лучше v0 ternary ~1.0) |
| Final | **1.025** |
| lr_decay | **не сработал** |

### #100k-l — FP32 `arch=exact` + rel_decay ❌

| Best rel | **~0.234** ep 13+ (плато) |
| Final | **0.233** |
| lr_decay | **не сработал** |

**Вывод exact:** teacher-shaped form **не лучше** v0 bottleneck при random init — FP32 v0 H=128 (**229k** params) учится, exact (**1.8k**) застревает @ rel≈0.23. Teacher weights → rel≈0, но Adam не находит их.

---

## Сводная 100k (финальные метрики)

| ID | Arch | Q | H | LR schedule | Best rel | Final rel | Phase 1 |
|----|------|---|-----|-------------|----------|-----------|---------|
| #100k-c | v0 | tern | 128 | const 3e-4 | 1.002 | 1.242 | ❌ |
| #100k-d | v0 | tern | 128 | const, WD=0.01 | 1.002 | 1.440 | ❌ |
| #100k-e | v0 | **fp32** | 128 | const 3e-4 | **0.0039** | 0.0049 | ⚠️ |
| #100k-f | v0 | tern | 128 | const 1e-4 | 1.002 | 1.009 | ❌ |
| #100k-g | v0 | fp32 | 256 | const 3e-4 | 0.0046 | 0.0041 | ⚠️ |
| **#100k-h** | v0 | **fp32** | 128 | **rel_decay 3e-4→1e-4** | **0.00094** | **0.00073** | **✅** |
| #100k-i | v0 | fp32 | 128 | rel_decay, LR=0.003 | 0.034 | 0.048 | ❌ |
| #100k-j | v0 | tern | 128 | rel_decay | 1.002 | 1.242 | ❌ |
| #100k-k | exact | tern | — | rel_decay | 0.947 | 1.025 | ❌ |
| #100k-l | exact | fp32 | — | rel_decay | 0.234 | 0.233 | ❌ |

---

## Выводы 100k (2026-06-17, финал)

1. **Re-extract 100k** — обязателен; на 4k FP32 тоже ≈1.
2. **FP32 v0 H=128 + rel_decay (3e-4→1e-4 @ rel<0.01)** — **Phase 1 ✅** `rel=0.00073` (#100k-h).
3. **Constant LR=3e-4 FP32** — `rel≈0.005` (#100k-e); **ternary** — `rel≈1` при любых LR/WD.
4. **rel_decay** помогает только если phase1 **достигает rel<0.01** (FP32 v0 да; ternary/exact — нет).
5. **arch exact** — хуже v0 bottleneck (FP32 rel≈0.23, ternary ≈0.95); не приоритет.
6. **Production (ternary):** warm-start FP32 → quantize; порт rel_decay в Mojo; Mojo train @ 100k.

```bash
# Re-extract
CALIBER158_SAMPLES=100000 make extract

# Phase 1 winner (Torch FP32, не production ternary)
CALIBER158_ARCH=v0 CALIBER158_QUANTIZE=0 CALIBER158_HIDDEN_DIM=128 \
  CALIBER158_EPOCHS=20 CALIBER158_LR=0.0003 CALIBER158_LR_MIN=0.0001 \
  CALIBER158_LR_REL_THRESHOLD=0.01 CALIBER158_LR_SCHEDULE=rel_decay \
  CALIBER158_WEIGHT_DECAY=0.001 make train-torch

# Ternary baseline (production path — platо ~1)
CALIBER158_ARCH=v0 CALIBER158_HIDDEN_DIM=128 CALIBER158_EPOCHS=20 \
  CALIBER158_LR=0.0003 CALIBER158_WEIGHT_DECAY=0.001 make train-torch
```
