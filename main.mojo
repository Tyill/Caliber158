"""Caliber158 CLI: train one ternary micro-chain or print project info."""

from std.sys import argv

from src.chain import BatchMicroNet, TrainConfig, init_random_weights, train_chain
from src.chain.buffer import ChainData
from src.chain.dataset import ChainDataset
from src.chain.env import TrainEnv
from src.chain.test_batch_grad import run_batch_grad_regression_test


def print_info(env: TrainEnv) -> None:
    print("Caliber158 — ternary micro-network distillation for Qwen chains")
    print("  model     :", env.model_name)
    print("  hidden    :", env.hidden_size, "(teacher input dim)")
    print("  hidden_dim:", env.hidden_dim, "(student width, CALIBER158_HIDDEN_DIM)")
    print("  dataset   :", env.dataset_path)
    print("  device    :", env.device.label())
    print("  backend   :", env.train_backend)
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

    var model = BatchMicroNet(dataset.input_dim, hidden_dim)
    init_random_weights(model, env.init_scale)

    train_chain(model, data, make_train_config(env, hidden_dim, env.epochs, env.batch_size))
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
        run_batch_grad_regression_test()
        print("test-grad: ok")
        return

    if command == "smoke":
        var hidden_dim = env.hidden_dim
        if len(args) >= 3:
            hidden_dim = Int(args[2])
        var dataset = ChainDataset.synthetic(env.smoke_samples, env.hidden_size)
        var data = ChainData.from_dataset(dataset)
        var model = BatchMicroNet(env.hidden_size, hidden_dim)
        init_random_weights(model, env.init_scale)
        train_chain(
            model,
            data,
            make_train_config(env, hidden_dim, env.smoke_epochs, env.smoke_batch_size),
        )
        return

    print("unknown command:", command)
    print_info(env)
