#!/usr/bin/env python3
"""Train one chain with PyTorch student (parallel R&D path to Mojo)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env_config import StudentEnv, load_student_env
from student.dataset import ChainDataset, read_dataset, synthetic
from student.holdout import HoldoutSplit, split_holdout
from student.metrics import batch_count, relative_mse
from student.model import MicroNet
from student.teacher_init import load_exact_teacher_vectors
from student.coord_desc_init import find_cd_init_from_teacher_quant


def _model_params(model: MicroNet, arch: str) -> list[torch.nn.Parameter]:
    if arch == "exact":
        return [model.gate, model.up, model.alpha]
    base = [model.gate, model.up, model.head, model.alpha]
    if arch in {"v1", "v1b"}:
        base.extend([model.gate2, model.up2])
    if arch == "v1b":
        base.extend([model.w_res, model.beta])
    return base


def _make_optimizer(model: MicroNet, env: StudentEnv) -> torch.optim.AdamW:
    params = [{"params": _model_params(model, env.arch), "weight_decay": env.weight_decay}]
    return torch.optim.AdamW(
        params,
        lr=env.learning_rate,
        betas=(env.beta1, env.beta2),
        eps=env.eps,
    )


def _set_optimizer_lr(optimizer: torch.optim.AdamW, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _optimizer_lr(optimizer: torch.optim.AdamW) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _eval_holdout(
    model: MicroNet,
    holdout: HoldoutSplit,
    device: torch.device,
) -> tuple[float, float]:
    hx = torch.from_numpy(holdout.holdout.x).to(device)
    hy = torch.from_numpy(holdout.holdout.y).to(device)
    holdout_mse = model.eval_mse(hx, hy)
    rel = relative_mse(holdout_mse, holdout.var_y_holdout)
    return holdout_mse, rel


def _maybe_decay_lr_on_rel(
    optimizer: torch.optim.AdamW,
    env: StudentEnv,
    rel: float,
    *,
    lr_decay_stage: int,
) -> int:
    """Two-stage rel_decay: threshold -> lr_min, then threshold2 -> lr_min2."""
    if env.lr_schedule != "rel_decay":
        return lr_decay_stage
    if lr_decay_stage < 1 and rel < env.lr_rel_threshold:
        old_lr = _optimizer_lr(optimizer)
        _set_optimizer_lr(optimizer, env.lr_min)
        print(
            f"lr_decay stage1: rel_holdout={rel} < {env.lr_rel_threshold}, "
            f"lr {old_lr} -> {env.lr_min}"
        )
        return 1
    if (
        lr_decay_stage < 2
        and env.lr_rel_threshold2 > 0.0
        and rel < env.lr_rel_threshold2
    ):
        old_lr = _optimizer_lr(optimizer)
        _set_optimizer_lr(optimizer, env.lr_min2)
        print(
            f"lr_decay stage2: rel_holdout={rel} < {env.lr_rel_threshold2}, "
            f"lr {old_lr} -> {env.lr_min2}"
        )
        return 2
    return lr_decay_stage


def _initial_lr_decay_stage(env: StudentEnv, lr: float) -> int:
    if env.lr_schedule != "rel_decay":
        return 0
    if env.lr_rel_threshold2 > 0.0 and lr <= env.lr_min2:
        return 2
    if lr <= env.lr_min:
        return 1
    return 0


def _make_scheduler(
    optimizer: torch.optim.AdamW,
    env: StudentEnv,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if env.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=env.lr_min,
        )
    return None


def _log_startup(
    env: StudentEnv,
    model: MicroNet,
    train: ChainDataset,
    holdout: HoldoutSplit,
) -> None:
    print(
        "training chain: train_samples=",
        train.n_samples,
        " holdout_samples=",
        holdout.holdout.n_samples,
        " hidden=",
        env.hidden_dim if env.arch != "exact" else "exact",
        " params=",
        model.param_count_total(),
        " lr=",
        env.learning_rate,
        " device=",
        env.device,
        " backend=torch",
        " quantize=",
        env.quantize_label,
        " arch=",
        env.arch,
        " init=",
        env.weight_init,
        " ste=",
        env.ste_mode,
        " block2_init=",
        env.block2_init,
        " lr_schedule=",
        env.lr_schedule,
        (
            f" lr_min={env.lr_min} rel_threshold={env.lr_rel_threshold}"
            f" lr_min2={env.lr_min2} rel_threshold2={env.lr_rel_threshold2}"
            if env.lr_schedule == "rel_decay"
            else ""
        ),
        (
            f" cd_sweeps={env.cd_sweeps}"
            if env.weight_init == "cd"
            else ""
        ),
        (
            f" grad_clip={env.grad_clip_max_norm}"
            if env.grad_clip_max_norm > 0.0
            else ""
        ),
        sep="",
    )
    if holdout.holdout.n_samples > 0:
        print(
            "holdout metrics: var_y_train=",
            holdout.var_y_train,
            " var_y_holdout=",
            holdout.var_y_holdout,
            sep="",
        )


def _log_epoch(
    epoch: int,
    train_mse: float,
    holdout_mse: float,
    rel: float,
    *,
    lr: float | None = None,
) -> None:
    lr_suffix = f" lr={lr}" if lr is not None else ""
    print(
        f"epoch {epoch} train_mse={train_mse} "
        f"holdout_mse={holdout_mse} rel_holdout={rel}{lr_suffix}"
    )


def _clip_grads(model: MicroNet, env: StudentEnv) -> None:
    if env.grad_clip_max_norm <= 0.0:
        return
    torch.nn.utils.clip_grad_norm_(_model_params(model, env.arch), env.grad_clip_max_norm)


def _run_epochs(
    model: MicroNet,
    optimizer: torch.optim.AdamW,
    env: StudentEnv,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    holdout_meta: HoldoutSplit,
) -> None:
    n = train_x.shape[0]
    has_holdout = holdout_meta.holdout.n_samples > 0
    scheduler = _make_scheduler(optimizer, env, env.epochs)
    lr_decay_stage = _initial_lr_decay_stage(env, _optimizer_lr(optimizer))

    for epoch in range(env.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        start = 0
        while start < n:
            count = batch_count(n, start, env.batch_size)
            if count == 0:
                break
            xb = train_x[start : start + count]
            yb = train_y[start : start + count]
            optimizer.zero_grad()
            loss = model.train_batch_loss(xb, yb)
            loss.backward()
            _clip_grads(model, env)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
            start += env.batch_size

        if scheduler is not None:
            scheduler.step()

        if epoch % env.log_every == 0 or epoch == env.epochs - 1:
            avg = total_loss / batches if batches else 0.0
            if has_holdout:
                model.eval()
                holdout_mse, rel = _eval_holdout(model, holdout_meta, model.gate.device)
                _log_epoch(
                    epoch,
                    avg,
                    holdout_mse,
                    rel,
                    lr=_optimizer_lr(optimizer) if env.lr_schedule == "rel_decay" else None,
                )
                lr_decay_stage = _maybe_decay_lr_on_rel(
                    optimizer, env, rel, lr_decay_stage=lr_decay_stage
                )
            else:
                _log_epoch(epoch, avg, 0.0, 0.0)


def train_chain(env: StudentEnv, data: ChainDataset, *, use_holdout: bool = True) -> None:
    device = torch.device(env.device)
    if use_holdout:
        split = split_holdout(data, env.holdout_fraction, env.split_seed)
        train_data = split.train
        holdout_meta = split.holdout
    else:
        train_data = data
        holdout_meta = HoldoutSplit(
            holdout=ChainDataset(0, data.input_dim, data.x[:0], data.y[:0]),
            var_y_train=0.0,
            var_y_holdout=0.0,
        )

    train_x = torch.from_numpy(train_data.x).to(device)
    train_y = torch.from_numpy(train_data.y).to(device)

    teacher_gate = None
    teacher_up = None
    cd_gate = None
    cd_up = None
    cd_alpha = None
    if env.weight_init == "teacher":
        print(
            "loading teacher init: model=",
            env.model_name,
            " layer=",
            env.layer,
            " neuron=",
            env.neuron,
            sep="",
        )
        teacher_gate, teacher_up = load_exact_teacher_vectors(
            env.model_name,
            env.layer,
            env.neuron,
        )
    elif env.weight_init == "cd":
        print(
            "cd init: model=",
            env.model_name,
            " layer=",
            env.layer,
            " neuron=",
            env.neuron,
            " sweeps=",
            env.cd_sweeps,
            sep="",
        )
        gate_fp, up_fp = load_exact_teacher_vectors(
            env.model_name,
            env.layer,
            env.neuron,
        )
        cd_gate, cd_up, cd_alpha = find_cd_init_from_teacher_quant(
            gate_fp,
            up_fp,
            train_x,
            train_y,
            max_sweeps=env.cd_sweeps,
        )

    model = MicroNet(
        data.input_dim,
        env.hidden_dim,
        arch=env.arch,
        use_ternary=env.use_ternary,
        ternary_threshold=env.ternary_threshold,
        ste_mode=env.ste_mode,
        init_scale=env.init_scale,
        block2_init=env.block2_init,
        block2_init_scale=env.block2_init_scale,
        weight_init=env.weight_init,
        teacher_gate=teacher_gate,
        teacher_up=teacher_up,
        cd_gate=cd_gate,
        cd_up=cd_up,
        cd_alpha=cd_alpha,
    ).to(device)
    _log_startup(env, model, train_data, holdout_meta)

    has_holdout = holdout_meta.holdout.n_samples > 0
    if has_holdout and env.weight_init == "cd":
        model.eval()
        pre_holdout, pre_rel = _eval_holdout(model, holdout_meta, device)
        print(
            f"pre_train holdout_mse={pre_holdout} rel_holdout={pre_rel} (cd init, before STE)"
        )

    optimizer = _make_optimizer(model, env)
    _run_epochs(model, optimizer, env, train_x, train_y, holdout_meta)

    if has_holdout:
        model.eval()
        final_holdout, rel = _eval_holdout(model, holdout_meta, device)
        print(
            f"final holdout_mse={final_holdout} rel_holdout={rel} "
            "(phase1 rel target 0.001)"
        )


def cmd_train(env: StudentEnv) -> None:
    print("loading dataset:", env.dataset_path)
    data = read_dataset(env.dataset_path)
    if data.input_dim != env.hidden_size:
        print(
            f"note: dataset input_dim={data.input_dim} "
            f"env CALIBER158_HIDDEN_SIZE={env.hidden_size}"
        )
    train_chain(env, data, use_holdout=True)
    print("done")


def cmd_smoke(env: StudentEnv) -> None:
    data = synthetic(env.smoke_samples, env.hidden_size)
    smoke_env = StudentEnv(
        hidden_dim=env.hidden_dim,
        dataset_path=env.dataset_path,
        hidden_size=env.hidden_size,
        epochs=env.smoke_epochs,
        batch_size=env.smoke_batch_size,
        learning_rate=env.learning_rate,
        weight_decay=env.weight_decay,
        beta1=env.beta1,
        beta2=env.beta2,
        eps=env.eps,
        log_every=env.log_every,
        init_scale=env.init_scale,
        ternary_threshold=env.ternary_threshold,
        smoke_epochs=env.smoke_epochs,
        smoke_batch_size=env.smoke_batch_size,
        smoke_samples=env.smoke_samples,
        model_name=env.model_name,
        device=env.device,
        holdout_fraction=env.holdout_fraction,
        split_seed=env.split_seed,
        use_ternary=env.use_ternary,
        arch=env.arch,
        block2_init=env.block2_init,
        block2_init_scale=env.block2_init_scale,
        lr_schedule=env.lr_schedule,
        lr_min=env.lr_min,
        lr_rel_threshold=env.lr_rel_threshold,
        lr_min2=env.lr_min2,
        lr_rel_threshold2=env.lr_rel_threshold2,
        grad_clip_max_norm=env.grad_clip_max_norm,
        weight_init=env.weight_init,
        layer=env.layer,
        neuron=env.neuron,
        ste_mode=env.ste_mode,
        cd_sweeps=env.cd_sweeps,
    )
    train_chain(smoke_env, data, use_holdout=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caliber158 Torch student prototype")
    parser.add_argument("command", choices=["train", "smoke"])
    args = parser.parse_args(argv)
    env = load_student_env()
    if args.command == "train":
        cmd_train(env)
    else:
        cmd_smoke(env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
