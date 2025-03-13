import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from classifier_dataloader import get_dataloader
from classifier_parser import args
from diffrax import Tsit5
from jaxtyping import Array, PyTree
from spec import build_lorenz96_sys, lorenz96_spec
from torch.utils.data import DataLoader
from tqdm import tqdm

from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.specification.trainable import TrainableMgr

jax.config.update("jax_enable_x64", True)

SEED = args.seed
np.random.seed(SEED)

N_STATE_VAR = args.n_state_var
FORCING = args.forcing

READOUT_TIME = args.readout_time
N_TIME_POINTS = args.n_time_points
VECTORIZE = args.vectorize

N_EPOCHS = args.n_epochs
BATCH_SIZE = args.batch_size
LR = args.lr
VALIDATION_SPLIT = args.validation_split
TESTING = args.testing

BATCH_NORM = args.batch_norm
NO_LORENZ = args.no_lorenz

DATASET = args.dataset
train_loader, val_loader = get_dataloader(
    dataset=DATASET,
    batch_size=BATCH_SIZE,
    shuffle=True,
    train=True,
    validation_split=VALIDATION_SPLIT,
)

# get the image size
IMG_SIZE = next(iter(train_loader))[0].shape[1]
N_LABEL = 10  # 10 classes for MNIST and FashionMNIST

trainable_mgr = TrainableMgr()
time_points = np.linspace(0, READOUT_TIME, N_TIME_POINTS, endpoint=True)
time_info = TimeInfo(
    t0=0,
    t1=READOUT_TIME,
    dt0=READOUT_TIME / N_TIME_POINTS,
    saveat=[READOUT_TIME],
)


class Classifier(eqx.Module):

    batch_norm: eqx.nn.BatchNorm
    lorenz96_sys: BaseAnalogCkt
    w_in: Array
    w_out: Array

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        n_state_var: int,
        forcing: float,
        lorenz_trainble_mgr: TrainableMgr,
        solver: diffrax.AbstractSolver,
        use_batch_norm: bool = False,
    ):
        lorenz_sys_cdg, state_vars = build_lorenz96_sys(
            n_state_var=n_state_var, init_F=forcing, trainable_mgr=lorenz_trainble_mgr
        )
        lorenz_sys_cls = OptCompiler().compile(
            prog_name="lorenz_96",
            cdg=lorenz_sys_cdg,
            cdg_spec=lorenz96_spec,
            trainable_mgr=trainable_mgr,
            readout_nodes=state_vars,
            vectorize=VECTORIZE,
            normalize_weight=False,
        )

        self.lorenz96_sys = lorenz_sys_cls(
            init_trainable=trainable_mgr.get_initial_vals(),
            is_stochastic=False,
            solver=solver,
        )
        self.w_in = jnp.array(np.random.randn(n_state_var, img_size))
        self.w_out = jnp.array(np.random.randn(n_classes, n_state_var))

        if use_batch_norm:
            self.batch_norm = eqx.nn.BatchNorm(
                input_size=n_state_var, axis_name="batch"
            )
        else:
            self.batch_norm = None

    def __call__(self, img: Array, time_info: TimeInfo, norm_state) -> Array:
        initial_state = jnp.matmul(self.w_in, img)
        if self.batch_norm is not None:
            initial_state, norm_state = self.batch_norm(initial_state, norm_state)
        trace = self.lorenz96_sys(
            time_info=time_info,
            initial_state=initial_state,
            switch=[],  # No switch
            args_seed=0,  # No random mismatch
            noise_seed=0,  # No random noise
        ).T
        return jnp.matmul(self.w_out, trace), norm_state


def loss(model: Classifier, state, img: Array, label: Array) -> Array:
    pred_label, state = jax.vmap(
        model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
    )(img, time_info, state)
    return cross_entropy(pred_label, label), state


def accuracy(model: Classifier, state, img: Array, label: Array) -> Array:
    pred_label, _ = jax.vmap(
        model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
    )(img, time_info, state)
    return jnp.mean(jnp.argmax(pred_label, axis=1) == label)


def cross_entropy(y_pred: Array, y_true: Array) -> Array:
    """Cross entropy loss for classification.

    Args:
        y_pred: Predicted labels in shape (BATCH_SIZE, N_LABEL).
        y_true: True labels in shape (BATCH_SIZE)."""
    y_true = jax.nn.one_hot(y_true, N_LABEL)
    return -jnp.mean(y_true * jax.nn.log_softmax(y_pred.squeeze()))


def train(
    model: Classifier,
    state,
    optimizer: optax.GradientTransformation,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def make_step(
        model: Classifier, state, opt_state: PyTree, img: Array, label: Array
    ):
        (loss_val, state), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            model, state, img, label
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, state, opt_state, loss_val, accuracy(model, state, img, label)

    @eqx.filter_jit
    def val_step(model: Classifier, state, img: Array, label: Array):
        return loss(model, state, img, label)[0], accuracy(model, state, img, label)

    print("Step\tTrain loss\tTrain accuracy\tValidation loss\tValidation accuracy")
    best_val_acc = 0
    best_weights = eqx.filter(model, eqx.is_array)
    for step in range(N_EPOCHS):
        for img, label in tqdm(train_loader):
            img, label = img.numpy(), label.numpy()
            model, state, opt_state, train_loss, train_acc = make_step(
                model, state, opt_state, img, label
            )
        for img, label in val_loader:
            img, label = img.numpy(), label.numpy()
            val_loss, val_acc = val_step(model, state, img, label)
        print(
            f"{step}\t{train_loss:.6f}\t{train_acc:.6f}\t{val_loss:.6f}\t{val_acc:.6f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = eqx.filter(model, eqx.is_array)
    return best_weights


if __name__ == "__main__":

    classifer, state = eqx.nn.make_with_state(Classifier)(
        img_size=IMG_SIZE,
        n_classes=N_LABEL,
        n_state_var=N_STATE_VAR,
        forcing=FORCING,
        lorenz_trainble_mgr=trainable_mgr,
        solver=Tsit5(),
        use_batch_norm=BATCH_NORM,
    )

    train(
        model=classifer,
        state=state,
        optimizer=optax.adam(LR),
        train_loader=train_loader,
        val_loader=val_loader,
    )
