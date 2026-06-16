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


def _make_optimizer(model: MicroNet, env: StudentEnv) -> torch.optim.AdamW:
    params = [
        {
            "params": [
                model.gate,
                model.up,
                model.head,
                model.alpha,
            ]
            + ([model.gate2, model.up2] if env.arch == "v1" else []),
            "weight_decay": env.weight_decay,
        }
    ]
    return torch.optim.AdamW(
        params,
        lr=env.learning_rate,
        betas=(env.beta1, env.beta2),
        eps=env.eps,
    )


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
        env.hidden_dim,
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
    model: MicroNet,
    holdout: HoldoutSplit,
    device: torch.device,
) -> None:
    if holdout.holdout.n_samples == 0:
        print(f"epoch {epoch} train_mse={train_mse}")
        return
    hx = torch.from_numpy(holdout.holdout.x).to(device)
    hy = torch.from_numpy(holdout.holdout.y).to(device)
    holdout_mse = model.eval_mse(hx, hy)
    rel = relative_mse(holdout_mse, holdout.var_y_holdout)
    print(
        f"epoch {epoch} train_mse={train_mse} "
        f"holdout_mse={holdout_mse} rel_holdout={rel}"
    )


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

    model = MicroNet(
        data.input_dim,
        env.hidden_dim,
        arch=env.arch,
        use_ternary=env.use_ternary,
        ternary_threshold=env.ternary_threshold,
        init_scale=env.init_scale,
    ).to(device)
    optimizer = _make_optimizer(model, env)

    _log_startup(env, model, train_data, holdout_meta)

    train_x = torch.from_numpy(train_data.x).to(device)
    train_y = torch.from_numpy(train_data.y).to(device)
    n = train_data.n_samples

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
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
            start += env.batch_size

        if epoch % env.log_every == 0 or epoch == env.epochs - 1:
            avg = total_loss / batches if batches else 0.0
            _log_epoch(epoch, avg, model, holdout_meta, device)

    if holdout_meta.holdout.n_samples > 0:
        hx = torch.from_numpy(holdout_meta.holdout.x).to(device)
        hy = torch.from_numpy(holdout_meta.holdout.y).to(device)
        final_holdout = model.eval_mse(hx, hy)
        rel = relative_mse(final_holdout, holdout_meta.var_y_holdout)
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
