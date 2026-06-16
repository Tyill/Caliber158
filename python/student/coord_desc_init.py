"""Coordinate descent ternary init for exact arch (Torch R&D)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .ternary import quantize_ternary

_TERNARY_VALUES = (-1.0, 0.0, 1.0)


def forward_base(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiLU(gate·x) * (up·x) without alpha; gate/up shape [D]."""
    g = x @ gate
    u = x @ up
    return F.silu(g) * u


def fit_alpha(base: torch.Tensor, y: torch.Tensor) -> float:
    """Closed-form alpha minimizing mean((alpha*base - y)^2)."""
    num = torch.dot(base, y)
    den = torch.dot(base, base)
    if float(den.item()) <= 0.0:
        return 1.0
    return float((num / den).item())


def ternary_from_teacher(
    gate_fp: np.ndarray,
    up_fp: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    gate_t = torch.tensor(gate_fp, dtype=torch.float32)
    up_t = torch.tensor(up_fp, dtype=torch.float32)
    return (
        quantize_ternary(gate_t, threshold).cpu().numpy(),
        quantize_ternary(up_t, threshold).cpu().numpy(),
    )


def _train_mse(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    alpha: float,
) -> float:
    pred = alpha * forward_base(x, gate, up)
    return float(F.mse_loss(pred, y, reduction="mean").item())


def coordinate_descent(
    gate: torch.Tensor,
    up: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    max_sweeps: int,
    label: str = "cd",
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    alpha = fit_alpha(forward_base(x_train, gate, up), y_train)
    best_mse = _train_mse(x_train, y_train, gate, up, alpha)
    if verbose:
        print(f"{label}: start train_mse={best_mse} alpha={alpha}")

    for sweep in range(max_sweeps):
        improved = False
        for vec in (gate, up):
            for i in range(vec.shape[0]):
                old = float(vec[i].item())
                for val in _TERNARY_VALUES:
                    if val == old:
                        continue
                    vec[i] = val
                    trial_alpha = fit_alpha(forward_base(x_train, gate, up), y_train)
                    trial_mse = _train_mse(x_train, y_train, gate, up, trial_alpha)
                    if trial_mse < best_mse:
                        best_mse = trial_mse
                        alpha = trial_alpha
                        old = val
                        improved = True
                    else:
                        vec[i] = old
                vec[i] = old
        if verbose:
            print(
                f"{label}: sweep {sweep + 1}/{max_sweeps} "
                f"train_mse={best_mse} alpha={alpha} improved={improved}"
            )
        if not improved:
            break
    return gate, up, alpha


def find_cd_init_from_teacher_quant(
    gate_fp: np.ndarray,
    up_fp: np.ndarray,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    max_sweeps: int,
    quant_threshold: float = 0.0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """CD on train from ternary-quantized teacher gate/up; returns CPU numpy + alpha."""
    gate_np, up_np = ternary_from_teacher(gate_fp, up_fp, quant_threshold)
    device = x_train.device
    gate = torch.tensor(gate_np, dtype=torch.float32, device=device)
    up = torch.tensor(up_np, dtype=torch.float32, device=device)
    gate, up, alpha = coordinate_descent(
        gate,
        up,
        x_train,
        y_train,
        max_sweeps=max_sweeps,
        label="cd_init",
        verbose=verbose,
    )
    return gate.detach().cpu().numpy(), up.detach().cpu().numpy(), alpha
