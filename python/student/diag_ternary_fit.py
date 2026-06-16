#!/usr/bin/env python3
"""Diagnostic: best ternary gate/up for exact arch (not FP32 teacher quant).

Searches {-1,0,1}^D gate/up via coordinate descent on train; reports holdout rel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env_config import StudentEnv, load_student_env
from student.coord_desc_init import (
    coordinate_descent,
    find_cd_init_from_teacher_quant,
    fit_alpha,
    forward_base,
    ternary_from_teacher,
)
from student.dataset import read_dataset
from student.holdout import split_holdout
from student.metrics import relative_mse
from student.rng import init_model_weights
from student.teacher_init import load_exact_teacher_vectors
from student.ternary import quantize_ternary


def _eval_split(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    alpha: float,
    var_y: float,
) -> tuple[float, float]:
    pred = alpha * forward_base(x, gate, up)
    mse = float(torch.nn.functional.mse_loss(pred, y, reduction="mean").item())
    return mse, relative_mse(mse, var_y)


def _vectors_from_numpy(gate_np, up_np, device):
    gate = torch.tensor(gate_np, dtype=torch.float32, device=device)
    up = torch.tensor(up_np, dtype=torch.float32, device=device)
    return gate, up


def _ternary_from_lcg(input_dim: int, scale: float):
    w = init_model_weights(input_dim, 0, 0, scale)
    gate_t = torch.tensor(w["gate"], dtype=torch.float32)
    up_t = torch.tensor(w["up"], dtype=torch.float32)
    return quantize_ternary(gate_t, 0.0).cpu().numpy(), quantize_ternary(up_t, 0.0).cpu().numpy()


def _report_case(
    name: str,
    gate: torch.Tensor,
    up: torch.Tensor,
    alpha: float,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_hold: torch.Tensor,
    y_hold: torch.Tensor,
    var_y_train: float,
    var_y_hold: float,
) -> None:
    train_mse, train_rel = _eval_split(x_train, y_train, gate, up, alpha, var_y_train)
    hold_mse, hold_rel = _eval_split(x_hold, y_hold, gate, up, alpha, var_y_hold)
    nz_gate = int((gate != 0).sum().item())
    nz_up = int((up != 0).sum().item())
    print(
        f"{name}: alpha={alpha:.6g} nz_gate={nz_gate} nz_up={nz_up} "
        f"train_mse={train_mse:.6g} train_rel={train_rel:.6g} "
        f"holdout_mse={hold_mse:.6g} holdout_rel={hold_rel:.6g}"
    )


def run_diag(env: StudentEnv, *, max_sweeps: int) -> None:
    device = torch.device(env.device)
    print("loading dataset:", env.dataset_path)
    data = read_dataset(env.dataset_path)
    split = split_holdout(data, env.holdout_fraction, env.split_seed)
    x_train = torch.from_numpy(split.train.x).to(device)
    y_train = torch.from_numpy(split.train.y).to(device)
    x_hold = torch.from_numpy(split.holdout.holdout.x).to(device)
    y_hold = torch.from_numpy(split.holdout.holdout.y).to(device)
    var_y_train = split.holdout.var_y_train
    var_y_hold = split.holdout.var_y_holdout

    print(
        f"train={split.train.n_samples} holdout={split.holdout.holdout.n_samples} "
        f"var_y_holdout={var_y_hold}"
    )

    gate_fp, up_fp = load_exact_teacher_vectors(env.model_name, env.layer, env.neuron)

    gate_fp_t, up_fp_t = _vectors_from_numpy(gate_fp, up_fp, device)
    alpha_fp = fit_alpha(forward_base(x_train, gate_fp_t, up_fp_t), y_train)
    _report_case(
        "teacher_fp32",
        gate_fp_t,
        up_fp_t,
        alpha_fp,
        x_train,
        y_train,
        x_hold,
        y_hold,
        var_y_train,
        var_y_hold,
    )

    for thresh in (0.0, 0.01):
        g_np, u_np = ternary_from_teacher(gate_fp, up_fp, thresh)
        gate_t, up_t = _vectors_from_numpy(g_np, u_np, device)
        alpha = fit_alpha(forward_base(x_train, gate_t, up_t), y_train)
        _report_case(
            f"teacher_quant(thresh={thresh})",
            gate_t,
            up_t,
            alpha,
            x_train,
            y_train,
            x_hold,
            y_hold,
            var_y_train,
            var_y_hold,
        )

    g_lcg, u_lcg = _ternary_from_lcg(data.input_dim, env.init_scale)
    gate_lcg, up_lcg = _vectors_from_numpy(g_lcg, u_lcg, device)
    alpha_lcg = fit_alpha(forward_base(x_train, gate_lcg, up_lcg), y_train)
    _report_case(
        "lcg_ternary_quant",
        gate_lcg,
        up_lcg,
        alpha_lcg,
        x_train,
        y_train,
        x_hold,
        y_hold,
        var_y_train,
        var_y_hold,
    )

    print(f"\ncoordinate descent (max_sweeps={max_sweeps}):")
    for init_name, g_np, u_np in (
        ("from_teacher_quant0", *_ternary_from_teacher(gate_fp, up_fp, 0.0)),
        ("from_lcg_ternary", g_lcg, u_lcg),
    ):
        gate, up = _vectors_from_numpy(g_np.copy(), u_np.copy(), device)
        gate, up, alpha = coordinate_descent(
            gate,
            up,
            x_train,
            y_train,
            max_sweeps=max_sweeps,
            label=init_name,
        )
        _report_case(
            f"coord_desc_{init_name}",
            gate,
            up,
            alpha,
            x_train,
            y_train,
            x_hold,
            y_hold,
            var_y_train,
            var_y_hold,
        )

    g_cd, u_cd, alpha_cd = find_cd_init_from_teacher_quant(
        gate_fp,
        up_fp,
        x_train,
        y_train,
        max_sweeps=max_sweeps,
        verbose=False,
    )
    gate_cd, up_cd = _vectors_from_numpy(g_cd, u_cd, device)
    _report_case(
        "find_cd_init_from_teacher_quant",
        gate_cd,
        up_cd,
        alpha_cd,
        x_train,
        y_train,
        x_hold,
        y_hold,
        var_y_train,
        var_y_hold,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Best ternary fit diagnostic for exact arch")
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=3,
        help="coordinate descent sweeps per init (default 3)",
    )
    args = parser.parse_args(argv)
    env = load_student_env()
    if env.arch != "exact":
        print(f"note: CALIBER158_ARCH={env.arch}; diagnostic uses exact forward regardless")
    run_diag(env, max_sweeps=args.max_sweeps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
