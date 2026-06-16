# Сравнение прогонов обучения — L00_N0000

Обновлено: 2026-06-16

**Датасет:** `data/chains/L00_N0000.bin`  
**Teacher:** Qwen2.5-0.5B, layer=0, neuron=0, **4096 samples** (не 100k из `.env`)  
**Student:** Mojo v0/v1 (`CALIBER158_ARCH`), ternary `{-1,0,1}` + α, **полный GPU train** (forward + backward + AdamW on device)  
**GPU:** NVIDIA RTX 3050 Ti  

## Масштаб teacher (Y)

| Метрика | Значение |
|---------|----------|
| `Y` min / max | −1.81 / 1.57 |
| `Y` std | ~0.18 |
| `Var(Y)` (весь датасет) | **0.0333** |
| `Var(Y)` train (3687) | 0.0329 |
| `Var(Y)` holdout (409) | 0.0371 |

**Критерий Phase 1:** holdout MSE < 1e-4 **или** `rel_holdout = MSE / Var(Y) < 0.001` (0.1% дисперсии).

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

**Вывод:** без ternary та же shallow v0 arch → underfit сохраняется → bottleneck в **ёмкости архитектуры**, не в quantize/STE. Обоснование для v1.

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

**Вывод:** v1 @ 30 ep не улучшил holdout vs v0 — block2 zero-init + те же гиперпараметры; нужны 50 ep, v1b (linear skip), или H↑.

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

\* train MSE / Var(Y), holdout не измерялся

---

## Выводы

1. **GPU train** дал ускорение с минут до ~45 с на 30 epochs (512 hidden).
2. **Рост `HIDDEN_DIM`** 128 → 256 дал огромный скачок; 256 → 512 почти не помог.
3. **Holdout ≈ train** на плато (`rel ~ 1.0`) → **underfit**, не overfit. Модель не выучивает teacher, а предсказывает ~среднее `Y`.
4. **FP32 diagnostic (#5b):** `rel_holdout ≈ 1.004` — как ternary → узкое место **ёмкость v0**, не ternary.
5. **v1 ternary (#6):** `rel_holdout ≈ 1.03` @ 1.44M params — **≈ v0**; block2 не помог за 30 ep.
6. **Дальше:** 50 ep v1, v1b (linear skip от x), H↑, или FP32 v1 diagnostic.

---

## Команды для воспроизведения

```bash
make train-cuda
# env: CALIBER158_HIDDEN_DIM=512, EPOCHS=30, LR=0.003,
#      WEIGHT_DECAY=0.001, HOLDOUT_FRACTION=0.1, SEED=42

# FP32 diagnostic (v0, no ternary quantize):
CALIBER158_QUANTIZE=0 make train-cuda

# v1 ternary (shell env overrides .env):
CALIBER158_ARCH=v1 CALIBER158_HIDDEN_DIM=512 CALIBER158_LR=0.003 \
  CALIBER158_WEIGHT_DECAY=0.001 CALIBER158_EPOCHS=30 make train-cuda
make test-grad-v1 test-grad-gpu-v1
```
