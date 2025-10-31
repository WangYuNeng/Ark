import os
import random

os.environ["EQX_ON_ERROR"] = "nan"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=multi_output_fusion"

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from dataloader import get_dataloader
from jaxtyping import Array, PyTree
from model import OBCStateFunc, OscillatorReservoir, ReservoirWithLinear
from parser import args
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

jax.config.update("jax_compilation_cache_dir", "jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

jax.config.update("jax_enable_x64", True)

SEED = args.seed
np.random.seed(SEED)
random.seed(SEED + 1)
torch.manual_seed(SEED + 2)

INPUT_TYPE = args.input_type
INPUT_KERNEL_SIZE = args.input_kernel_size
GRID_KERNEL_SIZE = args.grid_kernel_size

N_READOUT = args.n_readout
T_END = args.t_end
SAVEAT = [T_END * (i + 1) / N_READOUT for i in range(N_READOUT)]

N_EPOCHS = args.n_epochs
BATCH_SIZE = args.batch_size
LR = args.lr
VALIDATION_SPLIT = args.validation_split
TESTING = args.testing
IMG_DOWNSAMPLE = args.image_downsample
EARLY_STOPPING = args.early_stopping
TEST_ONLY = args.test_only

DATASET = args.dataset
train_loader, val_loader = get_dataloader(
    dataset=DATASET,
    batch_size=BATCH_SIZE,
    shuffle=True,
    train=True,
    validation_split=VALIDATION_SPLIT,
)
test_loader, _ = get_dataloader(
    dataset=DATASET,
    batch_size=BATCH_SIZE,
    shuffle=False,
    train=False,
    validation_split=0,
)
WANDB = args.wandb
RUN_NAME = args.run_name
# get the image size
IMG_SIZE = next(iter(train_loader))[0].shape[1]
N_LABEL = 10  # 10 classes for MNIST and FashionMNIST
SAVE_PATH = args.save_path
LOAD_PATH = args.load_path

if WANDB:
    wandb_run = wandb.init(
        config=vars(args),
        tags=[args.tag] if args.tag else None,
        name=RUN_NAME if RUN_NAME else None,
    )


def loss_w_acc(
    model: ReservoirWithLinear,
    img: Array,
    label: Array,
    t_end: float,
) -> Array:
    initial_state = jnp.zeros((model.reservoir.n_row, model.reservoir.n_col)).flatten()
    # Flatten the image
    img = img.reshape(img.shape[0], -1)
    pred_label = jax.vmap(model, in_axes=(None, 0, None))(initial_state, img, t_end)
    loss = cross_entropy(pred_label, label)
    acc = jnp.mean(jnp.argmax(pred_label, axis=1) == label)
    return loss, acc


def cross_entropy(y_pred: Array, y_true: Array) -> Array:
    """Cross entropy loss for classification.

    Args:
        y_pred: Predicted labels in shape (BATCH_SIZE, N_LABEL).
        y_true: True labels in shape (BATCH_SIZE)."""
    y_true = jax.nn.one_hot(y_true, N_LABEL)
    return -jnp.mean(y_true * jax.nn.log_softmax(y_pred.squeeze()))


def train(
    model: ReservoirWithLinear,
    reservoir_t_end: float,
    optimizer: optax.GradientTransformation,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader = None,
):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: ReservoirWithLinear,
        opt_state: PyTree,
        img: Array,
        label: Array,
        t_end: float,
    ):
        (loss_val, acc), grads = eqx.filter_value_and_grad(loss_w_acc, has_aux=True)(
            model, img, label, t_end
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return (
            model,
            opt_state,
            loss_val,
            acc,
        )

    @eqx.filter_jit
    def val_step(
        model: ReservoirWithLinear,
        img: Array,
        label: Array,
        t_end: float,
    ):
        return loss_w_acc(model, img, label, t_end)

    print(
        "Step\tTrain loss\tTrain accuracy\tValidation loss\tValidation accuracy\tTest accuracy"
    )
    best_val_acc = 0
    test_acc_at_best_val = 0
    no_improvement = 0

    if TEST_ONLY:
        test_accs = []
        for i, (img, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
            img, label = img.numpy(), label.numpy()
            _, test_acc = val_step(model, img, label, reservoir_t_end)
            test_accs.append(test_acc)
        print(f"Test accuracy: {np.mean(test_accs)}")
        if WANDB:
            wandb.log({"test_acc": np.mean(test_accs)})
        return
    for step in range(N_EPOCHS):
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        for i, (img, label) in tqdm(
            enumerate(train_loader), total=len(train_loader) - 1
        ):
            if i == len(train_loader) - 1:
                # The last batch has a different shape. Drop to avoid recompilation
                break
            img, label = img.numpy(), label.numpy()
            model, opt_state, train_loss, train_acc = make_step(
                model, opt_state, img, label, reservoir_t_end
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
        for i, (img, label) in enumerate(val_loader):
            if i == len(val_loader) - 1:
                break
            img, label = img.numpy(), label.numpy()
            val_loss, val_acc = val_step(model, img, label, reservoir_t_end)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        if test_loader:
            test_accs = []
            for img, label in test_loader:
                if i == len(test_loader) - 1:
                    break
                img, label = img.numpy(), label.numpy()
                _, test_acc = val_step(model, img, label, reservoir_t_end)
                test_accs.append(test_acc)
            test_acc = np.mean(test_accs)
        else:
            test_acc = ["N/A"]
        train_loss, train_acc = np.mean(train_losses), np.mean(train_accs)
        val_loss, val_acc = np.mean(val_losses), np.mean(val_accs)
        print(
            f"{step}\t{train_loss:.6f}\t{train_acc:.6f}\t{val_loss:.6f}\t{val_acc:.6f}\t{test_acc}"
        )
        if WANDB:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                }
            )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc_at_best_val = test_acc
            no_improvement = 0

            if SAVE_PATH:
                eqx.tree_serialise_leaves(SAVE_PATH, model)
        else:
            no_improvement += 1
            if no_improvement == EARLY_STOPPING:
                break
    if WANDB:
        wandb.log(
            {"best_val_acc": best_val_acc, "test_acc_at_best_val": test_acc_at_best_val}
        )
    return


if __name__ == "__main__":
    n_row = IMG_SIZE
    n_col = IMG_SIZE

    # Randomly initialize the coupling matrices
    grid_coupling = jnp.array(np.random.rand(n_row, n_col, n_row, n_col) - 0.5)
    input_coupling = jnp.array(np.random.rand(n_row, n_col, n_row, n_col) - 0.5)
    init_locking = jnp.array(np.random.rand(n_row, n_col) - 0.5)

    obc_func = OBCStateFunc(
        grid_coupling=grid_coupling,
        input_coupling=input_coupling,
        init_locking=init_locking,
        input_kernel_size=INPUT_KERNEL_SIZE,
        grid_kernel_size=GRID_KERNEL_SIZE,
    )
    reservoir = OscillatorReservoir(
        ode_fn=obc_func,
        solver=diffrax.Tsit5(),
        save_at=SAVEAT,
    )
    model = ReservoirWithLinear(
        reservoir=reservoir,
        output_dim=N_LABEL,
        downsample=IMG_DOWNSAMPLE,
    )

    if LOAD_PATH:
        model = eqx.tree_deserialise_leaves(LOAD_PATH, model)

    train(
        model=model,
        optimizer=optax.adam(LR),
        reservoir_t_end=T_END,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader if TESTING else None,
    )
