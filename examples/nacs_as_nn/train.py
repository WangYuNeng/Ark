import random

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
import wandb
from classifier_dataloader import get_dataloader
from classifier_parser import args
from jaxtyping import Array, PyTree
from model import NACSysClassifier, NACSysGrid
from torch.utils.data import DataLoader
from tqdm import tqdm

from ark.optimization.base_module import TimeInfo

jax.config.update("jax_enable_x64", True)

SEED = args.seed
np.random.seed(SEED)
random.seed(SEED + 1)
torch.manual_seed(SEED + 2)

SYS_NAME = args.sys_name
MISMATCH_RSTD = args.mismatch_rstd
INPUT_TYPE = args.input_type
NEIGHBOR_DIST = args.neighbor_dist
TRAINABLE_INIT = args.trainable_init
OUTPUT_QUANTIZATION_BITS = args.output_quantization_bits

READOUT_TIME = args.readout_time
DT0 = args.dt0

N_EPOCHS = args.n_epochs
BATCH_SIZE = args.batch_size
LR = args.lr
VALIDATION_SPLIT = args.validation_split
TESTING = args.testing
HIDDEN_SIZE = args.hidden_size
IMG_DOWNSAMPLE = args.image_downsample
BATCH_NORM = args.batch_norm
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

if SAVE_PATH and BATCH_NORM or LOAD_PATH and BATCH_NORM:
    raise NotImplementedError(
        "Batch normalization is not supported with saving/loading model."
    )

time_info = TimeInfo(
    t0=0,
    t1=READOUT_TIME,
    dt0=DT0,
    saveat=[READOUT_TIME],
)

if WANDB:
    wandb_run = wandb.init(
        config=vars(args),
        tags=[args.tag] if args.tag else None,
        name=RUN_NAME if RUN_NAME else None,
    )


def loss(
    model: NACSysClassifier,
    state: eqx.nn.State,
    img: Array,
    label: Array,
    mismatch_seeds: Array,
) -> Array:
    pred_label, state = jax.vmap(
        model, axis_name="batch", in_axes=(0, None, None, 0), out_axes=(0, None)
    )(img, state, time_info, mismatch_seeds)
    return cross_entropy(pred_label, label), state


def accuracy(
    model: NACSysClassifier,
    state: eqx.nn.State,
    img: Array,
    label: Array,
    mismatch_seeds: Array,
) -> Array:
    pred_label, _ = jax.vmap(
        model, axis_name="batch", in_axes=(0, None, None, 0), out_axes=(0, None)
    )(img, state, time_info, mismatch_seeds)
    return jnp.mean(jnp.argmax(pred_label, axis=1) == label)


def cross_entropy(y_pred: Array, y_true: Array) -> Array:
    """Cross entropy loss for classification.

    Args:
        y_pred: Predicted labels in shape (BATCH_SIZE, N_LABEL).
        y_true: True labels in shape (BATCH_SIZE)."""
    y_true = jax.nn.one_hot(y_true, N_LABEL)
    return -jnp.mean(y_true * jax.nn.log_softmax(y_pred.squeeze()))


def train(
    model: NACSysClassifier,
    state: eqx.nn.State,
    optimizer: optax.GradientTransformation,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader = None,
):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def make_step(
        model: NACSysClassifier,
        state: eqx.nn.State,
        opt_state: PyTree,
        img: Array,
        label: Array,
        mismatch_seeds: Array,
    ):
        (loss_val, state), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            model, state, img, label, mismatch_seeds
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return (
            model,
            state,
            opt_state,
            loss_val,
            accuracy(model, state, img, label, mismatch_seeds),
        )

    @eqx.filter_jit
    def val_step(
        model: NACSysClassifier,
        state: eqx.nn.State,
        img: Array,
        label: Array,
        mismatch_seeds: Array,
    ):
        return loss(model, state, img, label, mismatch_seeds)[0], accuracy(
            model, state, img, label, mismatch_seeds
        )

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
            mismatch_seeds = np.random.randint(0, 2**32, size=(img.shape[0],))
            _, test_acc = val_step(model, state, img, label, mismatch_seeds)
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
            mismatch_seeds = np.random.randint(0, 2**32, size=(img.shape[0],))
            model, state, opt_state, train_loss, train_acc = make_step(
                model, state, opt_state, img, label, mismatch_seeds
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
        for i, (img, label) in enumerate(val_loader):
            if i == len(val_loader) - 1:
                break
            img, label = img.numpy(), label.numpy()
            mismatch_seeds = np.random.randint(0, 2**32, size=(img.shape[0],))
            val_loss, val_acc = val_step(model, state, img, label, mismatch_seeds)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        if test_loader:
            test_accs = []
            for img, label in test_loader:
                if i == len(test_loader) - 1:
                    break
                img, label = img.numpy(), label.numpy()
                mismatch_seeds = np.random.randint(0, 2**32, size=(img.shape[0],))
                _, test_acc = val_step(model, state, img, label, mismatch_seeds)
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
    if SYS_NAME == "None":
        nacs_sys = None
    else:
        nacs_sys = NACSysGrid(
            sys_name=SYS_NAME,
            mismatch_rstd=MISMATCH_RSTD,
            n_rows=IMG_SIZE,
            n_cols=IMG_SIZE,
            neighbor_dist=NEIGHBOR_DIST,
            input_type=INPUT_TYPE,
            trainable_initialization=TRAINABLE_INIT,
        )
    classifer, state = eqx.nn.make_with_state(NACSysClassifier)(
        n_classes=N_LABEL,
        img_size=IMG_SIZE,
        nacs_sys=nacs_sys,
        hidden_size=HIDDEN_SIZE,
        img_downsample=IMG_DOWNSAMPLE,
        use_batch_norm=BATCH_NORM,
        key=jax.random.PRNGKey(SEED),
        adc_quantization_bits=OUTPUT_QUANTIZATION_BITS,
    )

    if LOAD_PATH:
        classifer = eqx.tree_deserialise_leaves(LOAD_PATH, classifer)

    # FIXME: somehow make_with_state produce states in float32, causing
    # incompatibility issue later. For now, manually convert the state to float64
    if BATCH_NORM:
        state._state[1] = (
            state._state[1][0].astype("float64"),
            state._state[1][1].astype("float64"),
        )

    train(
        model=classifer,
        state=state,
        optimizer=optax.adam(LR),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader if TESTING else None,
    )
