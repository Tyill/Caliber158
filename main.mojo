"""Caliber158 CLI: train one ternary micro-chain or print project info."""

from std.sys import argv

from src.chain.test_batch_grad import (
    run_batch_grad_regression_test,
    run_batch_grad_v1_regression_test,
    run_gpu_backward_regression_test,
    run_gpu_backward_v1_regression_test,
)
from src.chain import BatchMicroNet, TrainConfig, init_random_weights, train_chain
from src.chain.arch import arch_label
from src.chain.buffer import ChainData
from src.chain.holdout import no_holdout, split_holdout
from src.chain.dataset import ChainDataset
from src.chain.env import TrainEnv


def print_info(env: TrainEnv) -> None:
    print("Caliber158 — ternary micro-network distillation for Qwen chains")
    print("  model     :", env.model_name)
    print("  hidden    :", env.hidden_size, "(teacher input dim)")
    print("  hidden_dim:", env.hidden_dim, "(student width, CALIBER158_HIDDEN_DIM)")
    print("  dataset   :", env.dataset_path)
    print("  device    :", env.device.label())
    print("  backend   :", env.train_backend)
    print("  quantize  :", "ternary" if env.use_ternary else "fp32 (CALIBER158_QUANTIZE=0)")
    print("  arch      :", arch_label(env.arch))
    print()
    print("Config: copy .env.example → .env, or export CALIBER158_* vars.")
    print("Usage:")
    print("  pixi run mojo main.mojo info")
    print("  pixi run mojo main.mojo train [dataset_path] [hidden_dim]")
    print("  pixi run mojo main.mojo smoke [hidden_dim]")


def make_train_config(env: TrainEnv, hidden_dim: Int, epochs: Int, batch_size: Int) -> TrainConfig:
    return TrainConfig(
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=env.learning_rate,
        weight_decay=env.weight_decay,
        beta1=env.beta1,
        beta2=env.beta2,
        eps=env.eps,
        log_every=env.log_every,
        device=env.device,
    )


def run_train(dataset_path: String, hidden_dim: Int, env: TrainEnv) raises -> None:
    print("loading dataset:", dataset_path)
    var dataset = ChainDataset.load(dataset_path)
    var data = ChainData.from_dataset(dataset)

    if dataset.input_dim != env.hidden_size:
        print(
            "note: dataset input_dim=",
            dataset.input_dim,
            " env CALIBER158_HIDDEN_SIZE=",
            env.hidden_size,
        )

    var model = BatchMicroNet(
        dataset.input_dim, hidden_dim, use_ternary=env.use_ternary, arch=env.arch
    )
    init_random_weights(model, env.init_scale)

    var split = split_holdout(data, env.holdout_fraction, env.split_seed)

    train_chain(
        model,
        split.train,
        split.holdout,
        make_train_config(env, hidden_dim, env.epochs, env.batch_size),
    )
    print("done")


def main() raises:
    var env = TrainEnv.load()
    var args = argv()

    if len(args) < 2:
        print_info(env)
        return

    var command = args[1]
    if command == "info":
        print_info(env)
        return

    if command == "train":
        var dataset_path = env.dataset_path
        var hidden_dim = env.hidden_dim

        if len(args) >= 3:
            dataset_path = args[2]
        if len(args) >= 4:
            hidden_dim = Int(args[3])

        try:
            run_train(dataset_path, hidden_dim, env)
        except e:
            print("error:", e)
        return

    if command == "test-grad":
        if env.arch.is_v1():
            run_batch_grad_v1_regression_test()
        else:
            run_batch_grad_regression_test()
        print("test-grad: ok")
        return

    if command == "test-grad-gpu":
        if env.arch.is_v1():
            run_gpu_backward_v1_regression_test()
        else:
            run_gpu_backward_regression_test()
        print("test-grad-gpu: ok")
        return

    if command == "smoke":
        var hidden_dim = env.hidden_dim
        if len(args) >= 3:
            hidden_dim = Int(args[2])
        var dataset = ChainDataset.synthetic(env.smoke_samples, env.hidden_size)
        var data = ChainData.from_dataset(dataset)
        var model = BatchMicroNet(
            env.hidden_size, hidden_dim, use_ternary=env.use_ternary, arch=env.arch
        )
        init_random_weights(model, env.init_scale)
        train_chain(
            model,
            data,
            no_holdout(data),
            make_train_config(env, hidden_dim, env.smoke_epochs, env.smoke_batch_size),
        )
        return

    print("unknown command:", command)
    print_info(env)
