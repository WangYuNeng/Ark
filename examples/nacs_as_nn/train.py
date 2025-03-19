import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from classifier_dataloader import get_dataloader
from classifier_parser import args
from jaxtyping import Array, PyTree
from model import NACSysClassifier, NACSysGrid
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from ark.optimization.base_module import TimeInfo

jax.config.update("jax_enable_x64", True)

SEED = args.seed
np.random.seed(SEED)

SYS_NAME = args.sys_name
INPUT_TYPE = args.input_type
NEIGHBOR_DIST = args.neighbor_dist
TRAINABLE_INIT = args.trainable_init

READOUT_TIME = args.readout_time
N_TIME_POINTS = args.n_time_points

N_EPOCHS = args.n_epochs
BATCH_SIZE = args.batch_size
LR = args.lr
VALIDATION_SPLIT = args.validation_split
TESTING = args.testing

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

# get the image size
IMG_SIZE = next(iter(train_loader))[0].shape[1]
N_LABEL = 10  # 10 classes for MNIST and FashionMNIST

time_points = np.linspace(0, READOUT_TIME, N_TIME_POINTS, endpoint=True)
time_info = TimeInfo(
    t0=0,
    t1=READOUT_TIME,
    dt0=READOUT_TIME / N_TIME_POINTS,
    saveat=[READOUT_TIME],
)

if WANDB:
    wandb_run = wandb.init(
        config=vars(args),
    )


def loss(model: NACSysClassifier, img: Array, label: Array) -> Array:
    pred_label = jax.vmap(model, in_axes=(0, None))(img, time_info)
    return cross_entropy(pred_label, label)


def accuracy(model: NACSysClassifier, img: Array, label: Array) -> Array:
    pred_label = jax.vmap(model, axis_name="batch", in_axes=(0, None))(img, time_info)
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
    optimizer: optax.GradientTransformation,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader = None,
):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def make_step(model: NACSysClassifier, opt_state: PyTree, img: Array, label: Array):
        loss_val, grads = eqx.filter_value_and_grad(loss)(model, img, label)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val, accuracy(model, img, label)

    @eqx.filter_jit
    def val_step(model: NACSysClassifier, img: Array, label: Array):
        return loss(model, img, label), accuracy(model, img, label)

    print(
        "Step\tTrain loss\tTrain accuracy\tValidation loss\tValidation accuracy\tTest accuracy"
    )
    best_val_acc = 0
    best_weights = model.weight()
    for step in range(N_EPOCHS):
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        for img, label in tqdm(train_loader):
            img, label = img.numpy(), label.numpy()
            model, opt_state, train_loss, train_acc = make_step(
                model, opt_state, img, label
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
        for img, label in val_loader:
            img, label = img.numpy(), label.numpy()
            val_loss, val_acc = val_step(model, img, label)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        if test_loader:
            test_accs = []
            for img, label in test_loader:
                img, label = img.numpy(), label.numpy()
                test_acc = accuracy(model, img, label)
                test_accs.append(test_acc)
            test_accs = np.mean(test_accs)
        else:
            test_accs = ["N/A"]
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
            best_weights = model.weight()
    return best_weights


if __name__ == "__main__":
    nacs_sys = NACSysGrid(
        sys_name=SYS_NAME,
        n_rows=IMG_SIZE,
        n_cols=IMG_SIZE,
        neighbor_dist=NEIGHBOR_DIST,
        input_type=INPUT_TYPE,
        trainable_initialization=TRAINABLE_INIT,
    )
    classifer = NACSysClassifier(
        n_classes=N_LABEL,
        img_size=IMG_SIZE,
        nacs_sys=nacs_sys,
    )

    train(
        model=classifer,
        optimizer=optax.adam(LR),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader if TESTING else None,
    )
