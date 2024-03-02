import argparse
import time
from functools import partial
from types import FunctionType
from typing import Generator

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from differentiable_sspuf import SSPUF_ODE, SSPUF_MatrixExp, SwitchableStarPUF
from jax import config
from jaxtyping import Array, PyTree

config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


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


def step(x):
    """step function that maps x to {0, 1} for ideal ADC
    for arbitrary quantization value.

    If use this for optimization, the gradient can't pass through.
    All parameters are hardly updated.
    """
    return jnp.where(x > 0, 1, 0)


def multi_step(x, sep_val: list, k=10.0):
    """step function that maps x to {0, 1} for ideal multi-bit PUF ADC."""
    rtv = jnp.zeros_like(x)
    for i, val in enumerate(sep_val):
        rtv += jnp.where(x > val, 1, 0) * (1 - 2 * (i % 2))
    return rtv


def logistic(x, k=10.0):
    """Smooth approximation of the ideal ADC with the logistic function."""
    return 0.5 * (jnp.tanh(x * k / 2) + 1)


def multi_logistic(x, sep_val: list, k=10.0):
    """Smooth approximation of the ideal ADC with the logistic function
    for arbitrary quantization value.

    Args:
        x (jax.Array): input
        sep_val (list): list of separation values
        k (float, optional): steepness of the logistic function. Defaults to 10.0.

    Returns:
        jax.Array: quantized output
    """
    rtv = jnp.zeros_like(x)
    for i, val in enumerate(sep_val):
        sign = 1 - 2 * (i % 2)
        rtv += sign * logistic(x - val, k)
    return rtv


def diff(model, switch, mismatch, t):
    return jnp.mean(jnp.abs(jax.vmap(model)(switch, mismatch, t)))


def bf_loss(model, switch, mismatch, t, quantize_fn):
    """Calculating Bit-flipping Test Loss.

    It is a weaker version of I2O score, but easier to sample and compute.
    (Low bf loss is necessary but insufficient for high I2O loss)

    Analog_out: raw star difference
    digital_out: quantized star difference

    E.g., [1,0,0,1,0] -> abs_diff =[1,0,1,1] (with ideal quantization)
                      -> bf_score = 0.25
    """
    analog_out = jax.vmap(model)(switch, mismatch, t)
    digital_out: jax.Array = quantize_fn(analog_out).flatten()

    abs_diff = jnp.abs(digital_out[:-1] - digital_out[1:])
    return jnp.abs(jnp.mean(abs_diff) - 0.5)


def i2o_loss(
    model: SwitchableStarPUF,
    switch: Array,
    mismatch: Array,
    t: float,
    quantize_fn,
):
    """Calculating the I2O score.

    Analog_out: raw star difference
    digital_out: quantized star difference

    Args:
        model (SwitchableStarPUF): PUF model
        switches (Array): batch of switches in the shape of
            (inst_per_batch, chl_per_bit, 1 + n_bit, n_bit)
        mismatch (Array): batch of mismatch samples in the shape of
            (inst_per_batch, 2, *lds_a_shape)
        t (float): scalar readout time same as the readout_time
        quantize_fn : ADC quantization function
    """
    inst_per_batch, chl_per_bit, _, n_bit = switch.shape
    analog_out = jax.vmap(model, in_axes=(0, 0, None))(
        switch.reshape(-1, n_bit),
        mismatch.repeat(chl_per_bit * (1 + n_bit), axis=0).reshape(
            -1, 2, model.mismatch_len
        ),
        t,
    )
    digital_out: jax.Array = quantize_fn(analog_out).reshape(
        inst_per_batch, chl_per_bit, 1 + n_bit
    )

    dists = jnp.abs(
        jnp.array(
            [
                [
                    jnp.mean(jnp.abs(digital_out[i, :, 0] - digital_out[i, :, j])) - 0.5
                    for j in range(1, n_bit + 1)
                ]
                for i in range(inst_per_batch)
            ]
        )
    )

    return jnp.mean(dists)


def random_chls_and_mismatch(batch_size, n_branch, lds_a_shape, readout_time):
    while True:
        switches = np.random.randint(0, 2, size=(batch_size, n_branch))
        mismatch = np.random.normal(
            size=(batch_size, 2, *lds_a_shape), loc=1.0, scale=0.1
        )
        t = readout_time * np.ones(shape=(batch_size,))
        yield switches, mismatch, t


def bf_chls(
    batch_size: int,
    inst_per_batch: int,
    n_bit: int,
    mismatch_len: int,
    readout_time: float,
) -> Generator[jax.Array, jax.Array, jax.typing.DTypeLike]:
    """Bit-flipping test training data generator.

    Args:
        batch_size (int): size of the batch
        inst_per_batch (int): # of instance (mismatch)to sample per batch
        (batch_size // inst_per_batch should be an integer >> inst_per_batch)
        n_bit (int): # of bits (branches) in the star PUF
        mismatch (int): length of mismatch sampling for one star
        readout_time (float): readout time of the PUF

    Yields:
        switches (jax.Array): batch of switches
        mismatch (jax.Array): batch of mismatch samples
        t (jax.typing.DTypeLike): batch of readout time
    """
    assert batch_size % inst_per_batch == 0
    while True:
        switches = [np.random.randint(0, 2, size=(n_bit))]
        flipped_branch = np.random.randint(0, n_bit, size=(batch_size - 1))
        for i, pos in enumerate(flipped_branch):
            switches.append(switches[i].copy())
            switches[-1][pos] ^= 1
        switches = np.array(switches)

        # Generate inst_per_batch mismatch samples
        mismatches = np.random.normal(
            size=(inst_per_batch, 2, mismatch_len), loc=1.0, scale=0.1
        )
        # Repeat each mismatch samples batch_size // inst_per_batch times
        # and stack them together
        mismatch = np.repeat(mismatches, int(batch_size // inst_per_batch), axis=0)

        # Sanity check: use different mismatch the i2o score should be 0
        # mismatch = np.random.normal(
        #     size=(batch_size, 2, *lds_a_shape), loc=1.0, scale=0.1
        # )
        t = readout_time
        yield jnp.array(switches), jnp.array(mismatch), t


def I2O_chls(
    inst_per_batch: int,
    chl_per_bit: int,
    n_bit: int,
    mismatch_len: int,
    readout_time: float,
) -> Generator[jax.Array, jax.Array, jax.typing.DTypeLike]:
    """I2O score training data generator.

    Args:
        chl_per_bit (int): size of the batch
        inst_per_batch (int): # of instance (mismatch)to sample per batch
        (batch_size // inst_per_batch * n_branch should be an integer)
        n_bit (int): # of bits (branches) in the star PUF
        mismatch_len (int): length of the mismatch vector
        readout_time (float): readout time of the PUF

    Yields:
        switches (jax.Array): batch of switches in the shape of
            (inst_per_batch, chl_per_bit, 1 + n_bit, n_bit)
        mismatch (jax.Array): batch of mismatch samples in the shape of
            (inst_per_batch, 2, mismatch_len)
        t (jax.typing.DTypeLike): scalar readout time same as the readout_time
    """
    while True:
        switches = []
        for _ in range(inst_per_batch):
            switch_in_inst = []
            for _ in range(chl_per_bit):
                base_chl = np.random.randint(0, 2, size=(n_bit))
                flipped_chls = [base_chl]
                for i in range(n_bit):
                    flipped_chl = base_chl.copy()
                    flipped_chl[i] ^= 1
                    flipped_chls.append(flipped_chl)
                switch_in_inst.append(flipped_chls)
            switches.append(switch_in_inst)

        switches = np.array(switches)
        mismatch = np.random.normal(
            size=(inst_per_batch, 2, mismatch_len), loc=1.0, scale=0.1
        )
        t = readout_time
        yield jnp.array(switches), jnp.array(mismatch), t


def print_model_params(model):
    print(f"gm_c:\n{model.gm_c}")
    print(f"gm_l:\n{model.gm_l}")
    print(f"c_val:\n{model.c_val}")
    print(f"g_val:\n{model.g_val}")
    print(f"l_val:\n{model.l_val}")
    print(f"r_val:\n{model.r_val}")


def train(
    model: SwitchableStarPUF,
    loss: FunctionType,
    precise_loss: FunctionType,  # Due to approximation, use another loss function for validation
    dataloader: Generator[jax.Array, jax.Array, jax.typing.DTypeLike],
    optim: optax.GradientTransformation,
    steps: int,
    checkpoint: str,
    print_every: int,
):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: SwitchableStarPUF,
        opt_state: PyTree,
        switch: jax.Array,
        mismatch: jax.Array,
        t: jax.typing.DTypeLike,
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, switch, mismatch, t)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    @eqx.filter_jit
    def validate(
        model: SwitchableStarPUF,
        switch: jax.Array,
        mismatch: jax.Array,
        t: jax.typing.DTypeLike,
    ):
        loss_value, _ = eqx.filter_value_and_grad(precise_loss)(
            model, switch, mismatch, t
        )
        return loss_value

    best_loss_precise = 0.5  # Upper bound of the i2o and bit-flipping test loss
    for step, (switches, mismatch, t) in zip(range(steps), dataloader):
        if (step % print_every) == 0 or (step == steps - 1):
            train_loss_precise = validate(model, switches, mismatch, t)
        model, opt_state, train_loss = make_step(
            model, opt_state, switches, mismatch, t
        )
        if (step % print_every) == 0 or (step == steps - 1):
            if args.wandb:
                wandb_run.log(
                    {
                        "train_loss": train_loss.item(),
                        "loss_precise": train_loss_precise.item(),
                    }
                )
            print(
                f"{step=}, train_loss={train_loss.item()}, loss_precise={train_loss_precise.item()}"
            )
            print_model_params(model)

        if train_loss_precise < best_loss_precise:
            best_loss_precise = train_loss_precise
            eqx.tree_serialise_leaves(checkpoint, model)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, choices=["MatrixExp", "ODE"]
    )
    parser.add_argument("--loss", type=str, default="i2o", choices=["bf", "i2o"])
    parser.add_argument("--learning_rate", type=float, default=5e-2)
    parser.add_argument("--n_branch", type=int, default=10)
    parser.add_argument("--line_len", type=int, default=4)
    parser.add_argument(
        "--lossiness", type=str, default=None, choices=["None", "terminal", "all"]
    )
    parser.add_argument("--n_order", type=int, default=40)
    parser.add_argument("--n_time_points", type=int, default=100)
    parser.add_argument("--readout_time", type=float, default=10e-9)
    parser.add_argument(
        "--rand_init", type=str, default=None, choices=["None", "uniform", "normal"]
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chl_per_bit", type=int, default=64)
    parser.add_argument("--inst_per_batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--quantize_sep_val", nargs="+", type=float)
    parser.add_argument("--logistic_k", type=float, default=40)
    parser.add_argument("--seed", type=int, default=428)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    CHECKPOINT = f"checkpoint/{time.strftime('%Y%m%d-%H%M%S')}_{time.time():.0f}.eqx"

    train_config = vars(args)
    train_config["checkpoint"] = CHECKPOINT

    MODEL = args.model
    LOSS = args.loss
    LEARNING_RATE = args.learning_rate
    N_BRANCH = args.n_branch
    LINE_LEN = args.line_len
    LOSSINESS = args.lossiness
    N_ORDER = args.n_order
    N_TIME_POINTS = args.n_time_points
    READOUT_TIME = args.readout_time
    RAND_INIT = args.rand_init
    BATCH_SIZE = args.batch_size
    CHL_PER_BIT = args.chl_per_bit
    INST_PER_BATCH = args.inst_per_batch
    QUANTIZE_SEP_VAL = args.quantize_sep_val
    print("Quantize sep val:", QUANTIZE_SEP_VAL)
    STEPS = args.steps
    SEED = args.seed
    PRINT_EVERY = args.print_every
    LOGISTIC_K = args.logistic_k

    print("Config:", train_config)
    np.random.seed(SEED)

    optim = optax.adam(LEARNING_RATE)

    if MODEL == "MatrixExp":
        model = SSPUF_MatrixExp(
            n_branch=N_BRANCH,
            line_len=LINE_LEN,
            n_order=N_ORDER,
            random=RAND_INIT,
            lossiness=LOSSINESS,
        )
    elif MODEL == "ODE":
        model = SSPUF_ODE(
            n_branch=N_BRANCH,
            line_len=LINE_LEN,
            n_time_point=N_TIME_POINTS,
            random=RAND_INIT,
            lossiness=LOSSINESS,
        )
    else:
        raise ValueError("Unknown model")

    if LOSS == "bf":
        raise NotImplementedError("Bit-flipping test loss is no longer supported!")
        loader = bf_chls(
            BATCH_SIZE, INST_PER_BATCH, N_BRANCH, model.mismatch_len, READOUT_TIME
        )
        train_loss = partial(
            bf_loss,
            quantize_fn=partial(multi_logistic, sep_val=QUANTIZE_SEP_VAL, k=LOGISTIC_K),
        )
        train_loss_precise = partial(
            bf_loss, quantize_fn=partial(multi_step, sep_val=QUANTIZE_SEP_VAL)
        )
    elif LOSS == "i2o":
        loader = I2O_chls(
            INST_PER_BATCH, CHL_PER_BIT, N_BRANCH, model.mismatch_len, READOUT_TIME
        )
        train_loss = partial(
            i2o_loss,
            quantize_fn=partial(multi_logistic, sep_val=QUANTIZE_SEP_VAL, k=LOGISTIC_K),
        )
        train_loss_precise = partial(
            i2o_loss, quantize_fn=partial(multi_step, sep_val=QUANTIZE_SEP_VAL)
        )

    print_model_params(model)

    if args.wandb:
        if args.tag:
            wandb_run = wandb.init(
                config=train_config,
                tags=[args.tag],
            )
        else:
            wandb_run = wandb.init(
                config=train_config,
            )

    model = train(
        model=model,
        loss=train_loss,
        precise_loss=train_loss_precise,
        dataloader=loader,
        optim=optim,
        steps=STEPS,
        checkpoint=CHECKPOINT,
        print_every=PRINT_EVERY,
    )
