#!/usr/bin/env python3
"""Train one shared FFN layer with PyTorch (v2 R&D path)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env_config import LayerStudentEnv, load_layer_student_env
from student.ffn_layer import TernaryFFNLayer
from student.layer_dataset import LayerDataset, read_layer_dataset, synthetic_layer
from student.layer_holdout import LayerHoldoutSplit, split_layer_holdout
from student.metrics import batch_count, relative_mse
from student.teacher_init import load_layer_teacher_weights


def _model_params(model: TernaryFFNLayer) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    if model.rank <= 0:
        params.extend([model.gate, model.up, model.down])
    else:
        params.extend([
            model.gate_a,
            model.gate_b,
            model.up_a,
            model.up_b,
            model.down_a,
            model.down_b,
        ])
    if model.output_scale == "global" and model.alpha.numel() > 0:
        params.append(model.alpha)
    if model.output_scale == "channel" and model.channel_scale.numel() > 0:
        params.append(model.channel_scale)
    params.extend(model.projection_scale_params())
    return params


def _effective_rank(env: LayerStudentEnv, input_dim: int, intermediate_dim: int) -> int:
    if env.ffn_rank <= 0:
        return 0
    return min(env.ffn_rank, input_dim, intermediate_dim)


def _make_optimizer(model: TernaryFFNLayer, env: LayerStudentEnv) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [{"params": _model_params(model), "weight_decay": env.weight_decay}],
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
    model: TernaryFFNLayer,
    holdout: LayerHoldoutSplit,
    device: torch.device,
) -> tuple[float, float]:
    hx = torch.from_numpy(holdout.holdout.x).to(device)
    hy = torch.from_numpy(holdout.holdout.y).to(device)
    holdout_mse = model.eval_mse(hx, hy)
    rel = relative_mse(holdout_mse, holdout.var_y_holdout)
    return holdout_mse, rel


def _maybe_decay_lr_on_rel(
    optimizer: torch.optim.AdamW,
    env: LayerStudentEnv,
    rel: float,
    *,
    lr_decay_stage: int,
) -> int:
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


def _initial_lr_decay_stage(env: LayerStudentEnv, lr: float) -> int:
    if env.lr_schedule != "rel_decay":
        return 0
    if env.lr_rel_threshold2 > 0.0 and lr <= env.lr_min2:
        return 2
    if lr <= env.lr_min:
        return 1
    return 0


def _make_scheduler(
    optimizer: torch.optim.AdamW,
    env: LayerStudentEnv,
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
    env: LayerStudentEnv,
    model: TernaryFFNLayer,
    train: LayerDataset,
    holdout: LayerHoldoutSplit,
) -> None:
    rank_label = "full" if env.ffn_rank <= 0 else str(env.ffn_rank)
    print(
        "training layer_ffn: train_samples=",
        train.n_samples,
        " holdout_samples=",
        holdout.holdout.n_samples,
        " D=",
        env.hidden_size,
        " I=",
        env.intermediate_size,
        " rank=",
        rank_label,
        " params=",
        model.param_count_total(),
        " lr=",
        env.learning_rate,
        " device=",
        env.device,
        " backend=torch",
        " quantize=",
        env.quantize_label,
        " layer=",
        env.layer,
        " init=",
        env.weight_init,
        " ffn_scale=",
        env.ffn_scale,
        (
            f" scale_init={env.ffn_scale_init}"
            if env.ffn_scale != "none"
            else ""
        ),
        " ste=",
        env.ste_mode,
        " lr_schedule=",
        env.lr_schedule,
        (
            f" lr_min={env.lr_min} rel_threshold={env.lr_rel_threshold}"
            f" lr_min2={env.lr_min2} rel_threshold2={env.lr_rel_threshold2}"
            if env.lr_schedule == "rel_decay"
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
            "holdout metrics: mean_dim_var_y_train=",
            holdout.var_y_train,
            " mean_dim_var_y_holdout=",
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


def _clip_grads(model: TernaryFFNLayer, env: LayerStudentEnv) -> None:
    if env.grad_clip_max_norm <= 0.0:
        return
    torch.nn.utils.clip_grad_norm_(_model_params(model), env.grad_clip_max_norm)


def _run_epochs(
    model: TernaryFFNLayer,
    optimizer: torch.optim.AdamW,
    env: LayerStudentEnv,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    holdout_meta: LayerHoldoutSplit,
) -> None:
    n = train_x.shape[0]
    has_holdout = holdout_meta.holdout.n_samples > 0
    scheduler = _make_scheduler(optimizer, env, env.epochs)
    lr_decay_stage = _initial_lr_decay_stage(env, _optimizer_lr(optimizer))
    device = train_x.device

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
                holdout_mse, rel = _eval_holdout(model, holdout_meta, device)
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


def train_layer(env: LayerStudentEnv, data: LayerDataset, *, use_holdout: bool = True) -> None:
    device = torch.device(env.device)
    if use_holdout:
        split = split_layer_holdout(data, env.holdout_fraction, env.split_seed)
        train_data = split.train
        holdout_meta = split.holdout
    else:
        train_data = data
        holdout_meta = LayerHoldoutSplit(
            holdout=LayerDataset(0, data.input_dim, data.output_dim, data.x[:0], data.y[:0]),
            var_y_train=0.0,
            var_y_holdout=0.0,
        )

    train_x = torch.from_numpy(train_data.x).to(device)
    train_y = torch.from_numpy(train_data.y).to(device)

    teacher_gate = None
    teacher_up = None
    teacher_down = None
    if env.weight_init == "teacher":
        print(
            "loading teacher FFN init: model=",
            env.model_name,
            " layer=",
            env.layer,
            sep="",
        )
        teacher_gate, teacher_up, teacher_down, _, _ = load_layer_teacher_weights(
            env.model_name,
            env.layer,
        )

    model = TernaryFFNLayer(
        data.input_dim,
        env.intermediate_size,
        rank=_effective_rank(env, data.input_dim, env.intermediate_size),
        use_ternary=env.use_ternary,
        ternary_threshold=env.ternary_threshold,
        ste_mode=env.ste_mode,
        init_scale=env.init_scale,
        weight_init=env.weight_init,
        teacher_gate=teacher_gate,
        teacher_up=teacher_up,
        teacher_down=teacher_down,
        output_scale=env.ffn_scale,
    ).to(device)
    _log_startup(env, model, train_data, holdout_meta)

    has_holdout = holdout_meta.holdout.n_samples > 0
    if env.ffn_scale_init == "fit" and env.ffn_scale == "projection":
        fit_n = min(train_x.shape[0], 4096)
        if fit_n > 0:
            model.fit_projection_scales_from_samples(train_x, max_samples=fit_n)
            scales = torch.cat([p.detach().flatten() for p in model.projection_scale_params()])
            print(
                f"projection_scale_fit: count={scales.numel()} "
                f"mean={float(scales.mean())} min={float(scales.min())} max={float(scales.max())}"
            )
    elif env.ffn_scale_init == "fit" and env.ffn_scale != "none":
        fit_n = min(train_x.shape[0], 4096)
        if fit_n > 0:
            model.fit_output_scale_from_samples(train_x, train_y, max_samples=fit_n)
            if env.ffn_scale == "global":
                print(f"scale_fit: alpha={float(model.alpha.item())}")
            else:
                cs = model.channel_scale.detach()
                print(
                    f"scale_fit: channel_scale mean={float(cs.mean())} "
                    f"min={float(cs.min())} max={float(cs.max())}"
                )

    if has_holdout and (
        env.weight_init == "teacher"
        or (env.ffn_scale_init == "fit" and env.ffn_scale != "none")
    ):
        model.eval()
        pre_holdout, pre_rel = _eval_holdout(model, holdout_meta, device)
        print(
            f"pre_train holdout_mse={pre_holdout} rel_holdout={pre_rel} "
            "(after init / scale fit, before STE train)"
        )

    optimizer = _make_optimizer(model, env)
    _run_epochs(model, optimizer, env, train_x, train_y, holdout_meta)

    if holdout_meta.holdout.n_samples > 0:
        model.eval()
        final_holdout, rel = _eval_holdout(model, holdout_meta, device)
        print(
            f"final holdout_mse={final_holdout} rel_holdout={rel} "
            "(phase1 rel target 0.001)"
        )


def cmd_train(env: LayerStudentEnv) -> None:
    print("loading layer dataset:", env.dataset_path)
    data = read_layer_dataset(env.dataset_path)
    if data.input_dim != env.hidden_size or data.output_dim != env.hidden_size:
        print(
            f"note: dataset dims in={data.input_dim} out={data.output_dim} "
            f"env CALIBER158_HIDDEN_SIZE={env.hidden_size}"
        )
    train_layer(env, data, use_holdout=True)
    print("done")


def cmd_smoke(env: LayerStudentEnv) -> None:
    d = env.smoke_input_dim
    i = env.smoke_intermediate_dim
    out_d = env.smoke_output_dim
    smoke_rank = _effective_rank(env, d, i)
    data = synthetic_layer(env.smoke_samples, d, out_d)
    smoke_env = LayerStudentEnv(
        dataset_path=env.dataset_path,
        hidden_size=d,
        intermediate_size=i,
        ffn_rank=smoke_rank,
        layer=env.layer,
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
        smoke_input_dim=d,
        smoke_intermediate_dim=i,
        smoke_output_dim=out_d,
        device=env.device,
        holdout_fraction=env.holdout_fraction,
        split_seed=env.split_seed,
        use_ternary=env.use_ternary,
        lr_schedule=env.lr_schedule,
        lr_min=env.lr_min,
        lr_rel_threshold=env.lr_rel_threshold,
        lr_min2=env.lr_min2,
        lr_rel_threshold2=env.lr_rel_threshold2,
        grad_clip_max_norm=env.grad_clip_max_norm,
        ste_mode=env.ste_mode,
        model_name=env.model_name,
        weight_init="lcg",
        ffn_scale=env.ffn_scale,
        ffn_scale_init=env.ffn_scale_init,
    )
    train_layer(smoke_env, data, use_holdout=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Caliber158 Torch layer FFN train (v2)")
    parser.add_argument("command", choices=["train", "smoke"])
    args = parser.parse_args(argv)
    env = load_layer_student_env()
    if args.command == "train":
        cmd_train(env)
    else:
        cmd_smoke(env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
