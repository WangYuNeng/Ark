import argparse
from functools import partial
from types import FunctionType

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from differentiable_sspuf import SwitchableStarPUF
from jax import config
from jaxtyping import Array, Float, PyTree

config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=5e-2)
parser.add_argument("--n_branch", type=int, default=10)
parser.add_argument("--line_len", type=int, default=4)
parser.add_argument("--n_order", type=int, default=40)
parser.add_argument("--readout_time", type=float, default=10e-9)
parser.add_argument("--rand_init", action="store_true")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--print_every", type=int, default=1)
parser.add_argument("--logistic_k", type=float, default=40)
parser.add_argument("--wandb", action="store_true")
args = parser.parse_args()

if args.wandb:
    wandb_run = wandb.init(
        config=vars(args),
    )


def plot_single_star_rsp(model, switch, mismatch, time_points):
    """Sanity check: Plot the transient response of a single star

    Should behave like transmission line.
    """
    trace = [
        model._calc_one_star_rsp(switch[0], mismatch[0][0], t) for t in time_points
    ]
    trace = np.array(trace).squeeze()
    plt.title(f"switch={switch[0]}")
    plt.plot(time_points, trace)
    plt.show()
    plt.close()


def shifted_sign(x):
    """Shifted sign function that maps x to {0, 1} for ideal PUF ADC.

    If use this for optimization, the gradient can't pass through.
    All parameters are hardly updated.
    """
    return jnp.where(x > 0, 1, 0)


def logistic(x, k=10.0):
    """Smooth approximation of the ideal ADC with the logistic function."""
    return 1.0 / (1.0 + jnp.exp(-x * k))


def diff(model, switch, mismatch, t):
    return jnp.mean(jnp.abs(jax.vmap(model)(switch, mismatch, t)))


def i2o_score(model, switch, mismatch, t, quantize_fn):
    analog_out = jax.vmap(model)(switch, mismatch, t)
    digital_out: jax.Array = quantize_fn(analog_out).flatten()

    abs_diff = jnp.abs(digital_out[:-1] - digital_out[1:])
    return jnp.abs(jnp.mean(abs_diff) - 0.5)


def random_chls_and_mismatch(batch_size, n_branch, lds_a_shape, readout_time):
    while True:
        switches = np.random.randint(0, 2, size=(batch_size, n_branch))
        mismatch = np.random.normal(
            size=(batch_size, 2, *lds_a_shape), loc=1.0, scale=0.1
        )
        t = readout_time * np.ones(shape=(batch_size,))
        yield switches, mismatch, t


def bf_chls(batch_size, n_branch, lds_a_shape, readout_time):
    while True:
        switches = [np.random.randint(0, 2, size=(n_branch))]
        flipped_branch = np.random.randint(0, n_branch, size=(batch_size - 1))
        for i, pos in enumerate(flipped_branch):
            switches.append(switches[i].copy())
            switches[-1][pos] ^= 1
        switches = np.array(switches)
        mismatch = np.random.normal(size=(2, *lds_a_shape), loc=1.0, scale=0.1)
        # Use the same mismatch across all batch
        mismatch = np.repeat(mismatch[np.newaxis, ...], batch_size, axis=0)

        # Sanity check: use different mismatch the i2o score should be 0
        # mismatch = np.random.normal(
        #     size=(batch_size, 2, *lds_a_shape), loc=1.0, scale=0.1
        # )
        t = readout_time * np.ones(shape=(batch_size,))
        yield switches, mismatch, t


def train(
    model: SwitchableStarPUF,
    loss: FunctionType,
    val_loss: FunctionType,  # Due to approximation, use another loss function for validation
    dataloader: FunctionType,
    optim: optax.GradientTransformation,
    batch_size: int,
    steps: int,
    print_every: int,
):
    """Toy example that minimizes the difference between two stars.
    We know that a simple solution is to make every weight 0.
    """

    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    n_branch, lds_a_shape = model.n_branch, model.lds_a_shape
    loader = dataloader(batch_size, n_branch, lds_a_shape)

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: SwitchableStarPUF,
        opt_state: PyTree,
        switch: Array,
        mismatch: Array,
        t: Float,
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, switch, mismatch, t)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    @eqx.filter_jit
    def validate(model: SwitchableStarPUF, switch: Array, mismatch: Array, t: Float):
        loss_value, _ = eqx.filter_value_and_grad(val_loss)(model, switch, mismatch, t)
        return loss_value

    prev_gmc, prev_gml = model.gm_c, model.gm_l
    for step, (switches, mismatch, t) in zip(range(steps), loader):
        model, opt_state, train_loss = make_step(
            model, opt_state, switches, mismatch, t
        )
        if (step % print_every) == 0 or (step == steps - 1):
            val_loss = validate(model, switches, mismatch, t)
            if args.wandb:
                wandb_run.log(
                    {
                        "train_loss": train_loss.item(),
                        "validation_loss": val_loss.item(),
                    }
                )
            print(
                f"{step=}, train_loss={train_loss.item()}, validation_loss={val_loss.item()}"
            )
            print(f"gm_c:\n{model.gm_c}")
            print(f"gm_l:\n{model.gm_l}")
            print(f"gmc_diff:\n{model.gm_c - prev_gmc}")
            print(f"gml_diff:\n{model.gm_l - prev_gml}")
            prev_gmc, prev_gml = model.gm_c, model.gm_l
    return model


LEARNING_RATE = args.learning_rate
N_BRANCH = args.n_branch
LINE_LEN = args.line_len
N_ORDER = args.n_order
READOUT_TIME = args.readout_time
RAND_INIT = args.rand_init
BATCH_SIZE = args.batch_size
STEPS = args.steps
PRINT_EVERY = args.print_every
LOGISTIC_K = args.logistic_k

optim = optax.adamw(LEARNING_RATE, weight_decay=0)
rand_loader = partial(random_chls_and_mismatch, readout_time=READOUT_TIME)
bf_loader = partial(bf_chls, readout_time=READOUT_TIME)

# # Sanity check, single star response is correct
# model = SwitchableStarPUF(n_branch=N_BRANCH, line_len=LINE_LEN, n_order=N_ORDER)
# for switches, mismatch, _ in rand_loader(BATCH_SIZE, N_BRANCH, model.lds_a_shape):
#     plot_single_star_rsp(model, switches, mismatch, np.linspace(2e-9, 20e-9, 100))
#     if input() != "c":
#         exit()

# # Minimize the difference between two stars: easy case
# print(model.gm_c, model.gm_l)
# model = train(
#     model=model,
#     loss=diff,
#     dataloader=rand_loader,
#     optim=optim,
#     batch_size=BATCH_SIZE,
#     steps=STEPS,
#     print_every=PRINT_EVERY,
# )


# # Minimize the difference between two stars: arger case
# model = SwitchableStarPUF(n_branch=N_BRANCH, line_len=LINE_LEN, n_order=N_ORDER)
# print(model.gm_c, model.gm_l)
# model = train(
#     model=model,
#     loss=diff,
#     dataloader=rand_loader,
#     optim=optim,
#     batch_size=BATCH_SIZE,
#     steps=STEPS,
#     print_every=PRINT_EVERY,
# )


# Bit-flipping test
model = SwitchableStarPUF(
    n_branch=N_BRANCH, line_len=LINE_LEN, n_order=N_ORDER, random=RAND_INIT
)
i2o_sigmoid = partial(i2o_score, quantize_fn=partial(logistic, k=LOGISTIC_K))
i2o_ideal = partial(i2o_score, quantize_fn=shifted_sign)

# # Sanity Check
# n_branch, lds_a_shape = model.n_branch, model.lds_a_shape
# for i, (switches, mismatch, t) in zip(
#     range(10), bf_chls(BATCH_SIZE, n_branch, lds_a_shape, READOUT_TIME)
# ):
#     ideal_loss = i2o_ideal(model, switches, mismatch, t)
#     approx_loss = i2o_sigmoid(model, switches, mismatch, t)
#     print(ideal_loss, approx_loss)

print(model.gm_c, model.gm_l)
model = train(
    model=model,
    loss=i2o_sigmoid,
    val_loss=i2o_ideal,
    dataloader=bf_loader,
    optim=optim,
    batch_size=BATCH_SIZE,
    steps=STEPS,
    print_every=PRINT_EVERY,
)
