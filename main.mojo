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
from src.chain.grads import ModelGrads
from src.chain.holdout import no_holdout, split_holdout
from src.chain.dataset import ChainDataset
from src.chain.env import TrainEnv
from src.chain.rng import lcg_next, unit_float


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


def run_parity_export(env: TrainEnv) raises -> None:
    """JSON lines for Torch parity tests: holdout indices + one-batch CPU loss."""
    var n = 4096
    var fraction = Float32(0.1)
    var seed = UInt64(42)

    var indices = List[Int](capacity=n)
    for i in range(n):
        indices.append(i)
    var rng = seed
    for i in range(n - 1, 0, -1):
        rng = lcg_next(rng)
        var j = Int(unit_float(rng) * Float32(i + 1))
        var tmp = indices[i]
        indices[i] = indices[j]
        indices[j] = tmp

    var holdout_count = Int(Float32(n) * fraction)
    if holdout_count < 1:
        holdout_count = 1
    if holdout_count >= n:
        holdout_count = n - 1
    var train_count = n - holdout_count

    print('{"kind":"holdout","n":', n, ',"holdout_count":', holdout_count, ',"train_count":', train_count, ',"holdout_indices":[', sep="")
    for i in range(holdout_count):
        if i > 0:
            print(",", end="")
        print(indices[i], end="")
    print("]}")

    var input_dim = 32
    var hidden_dim = 8
    var batch_size = 16
    var dataset = ChainDataset.synthetic(batch_size, input_dim)
    var data = ChainData.from_dataset(dataset)
    var model = BatchMicroNet(input_dim, hidden_dim, use_ternary=True)
    init_random_weights(model, env.init_scale)
    var grads = ModelGrads.zeros_for_model(input_dim, hidden_dim, model.arch)
    var loss = model.train_step_cpu(data, 0, batch_size, grads)
    print('{"kind":"batch_loss","input_dim":', input_dim, ',"hidden_dim":', hidden_dim, ',"batch_size":', batch_size, ',"loss":', loss, '}', sep="")


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

    if command == "parity-export":
        run_parity_export(env)
        return

    print("unknown command:", command)
    print_info(env)
