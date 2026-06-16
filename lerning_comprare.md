# Learning compare — v0 vs v1 (`L00_N0000.bin`)

Dataset: `data/chains/L00_N0000.bin` (4096 samples → **3687 train / 409 holdout**, seed 42).

GPU train (`CALIBER158_DEVICE=cuda`), `LR=0.003`, `WEIGHT_DECAY=0.001`, batch 64.

v1 defaults (code): `INIT_SCALE_BLOCK2 = INIT_SCALE/H`, `BLOCK2_RESIDUAL_SCALE = 1/H`.

---

## HIDDEN_DIM=128, 30 epochs

| | v0 | v1 |
|---|---|---|
| arch | v0 | v1 |
| params | 229 377 | 262 273 |
| `Var(Y)` train / holdout | 0.0329 / 0.0371 | same |
| epoch 0 train_mse | 192 072 460 | **23 390 115 000 000** |
| epoch 29 train_mse | **0.0326** | **14 172 599 000 000** |
| final holdout_mse | **0.0384** | **11 841 315 000 000** |
| rel_holdout | **~1.03** | **~3.19×10¹⁴** |
| wall time | ~14 s | ~16 s |

### v1 — полная кривая (2026-06-16)

```
epoch  0  train_mse= 2.339e+13  holdout_mse= 7.899e+12  rel_holdout= 2.130e+14
epoch  1  train_mse= 2.225e+13  holdout_mse= 7.512e+12  rel_holdout= 2.025e+14
epoch  2  train_mse= 3.255e+16  holdout_mse= 5.505e+12  rel_holdout= 1.484e+14
epoch  3  train_mse= 1.917e+17  holdout_mse= 6.404e+12  rel_holdout= 1.727e+14
epoch  4  train_mse= 5.965e+17  holdout_mse= 1.126e+13  rel_holdout= 3.036e+14
epoch  5  train_mse= 1.157e+18  holdout_mse= 9.326e+12  rel_holdout= 2.515e+14
epoch  6  train_mse= 5.655e+17  holdout_mse= 1.785e+13  rel_holdout= 4.815e+14
epoch  7  train_mse= 4.322e+18  holdout_mse= 2.411e+13  rel_holdout= 6.502e+14
epoch  8  train_mse= 2.026e+19  holdout_mse= 5.300e+13  rel_holdout= 1.429e+15
epoch  9  train_mse= 7.063e+19  holdout_mse= 2.238e+13  rel_holdout= 6.034e+14
epoch 10  train_mse= 2.348e+18  holdout_mse= 1.634e+13  rel_holdout= 4.407e+14
epoch 11  train_mse= 6.929e+15  holdout_mse= 1.593e+13  rel_holdout= 4.296e+14
epoch 12  train_mse= 5.019e+15  holdout_mse= 1.593e+13  rel_holdout= 4.295e+14
epoch 13  train_mse= 1.497e+15  holdout_mse= 1.482e+13  rel_holdout= 3.995e+14
epoch 14  train_mse= 6.454e+14  holdout_mse= 1.288e+13  rel_holdout= 3.474e+14
epoch 15  train_mse= 7.696e+14  holdout_mse= 1.219e+13  rel_holdout= 3.287e+14
epoch 16  train_mse= 1.342e+14  holdout_mse= 1.161e+13  rel_holdout= 3.129e+14
epoch 17  train_mse= 1.311e+14  holdout_mse= 1.183e+13  rel_holdout= 3.189e+14
epoch 18  train_mse= 8.196e+13  holdout_mse= 1.174e+13  rel_holdout= 3.167e+14
epoch 19  train_mse= 1.687e+14  holdout_mse= 1.102e+13  rel_holdout= 2.971e+14
epoch 20  train_mse= 6.237e+13  holdout_mse= 1.223e+13  rel_holdout= 3.298e+14
epoch 21  train_mse= 7.146e+13  holdout_mse= 1.161e+13  rel_holdout= 3.130e+14
epoch 22  train_mse= 1.677e+14  holdout_mse= 1.250e+13  rel_holdout= 3.371e+14
epoch 23  train_mse= 1.212e+14  holdout_mse= 1.283e+13  rel_holdout= 3.460e+14
epoch 24  train_mse= 2.248e+13  holdout_mse= 1.291e+13  rel_holdout= 3.482e+14
epoch 25  train_mse= 1.488e+13  holdout_mse= 1.233e+13  rel_holdout= 3.326e+14
epoch 26  train_mse= 6.608e+12  holdout_mse= 1.240e+13  rel_holdout= 3.344e+14
epoch 27  train_mse= 5.205e+12  holdout_mse= 1.276e+13  rel_holdout= 3.441e+14
epoch 28  train_mse= 1.930e+13  holdout_mse= 1.237e+13  rel_holdout= 3.336e+14
epoch 29  train_mse= 1.417e+13  holdout_mse= 1.184e+13  rel_holdout= 3.193e+14
```

Команда:

```bash
pixi run bash -c 'source scripts/load-env.sh && \
  export CALIBER158_ARCH=v1 CALIBER158_DEVICE=cuda \
         CALIBER158_HIDDEN_DIM=128 CALIBER158_EPOCHS=30 && \
  mojo main.mojo train'
```

### Вывод (H=128)

- **v0** сходится к плато `MSE ≈ Var(Y)` (`rel_holdout ≈ 1`) — underfit, но стабильно.
- **v1** на GPU не обучается: loss остаётся ~10¹²–10¹⁹, `rel_holdout` ~10¹⁴ (цель Phase 1: `0.001`).
- Уменьшение H с 512 до 128 **не спасает** v1.
