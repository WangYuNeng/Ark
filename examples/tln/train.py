import argparse
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
from differentiable_sspuf import SSPUF_MatrixExp, SwitchableStarPUF
from jax import config
from jaxtyping import Array, Float, PyTree

config.update("jax_debug_nans", True)


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
    """step function that maps x to {0, 1} for ideal PUF ADC.

    If use this for optimization, the gradient can't pass through.
    All parameters are hardly updated.
    """
    return jnp.where(x > 0, 1, 0)


def logistic(x, k=10.0):
    """Smooth approximation of the ideal ADC with the logistic function."""
    return 0.5 * (jnp.tanh(x * k / 2) + 1)


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
            -1, 2, *model.lds_a_shape
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
    lds_a_shape: tuple[int, int],
    readout_time: float,
) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Bit-flipping test training data generator.

    Args:
        batch_size (int): size of the batch
        inst_per_batch (int): # of instance (mismatch)to sample per batch
        (batch_size // inst_per_batch should be an integer >> inst_per_batch)
        n_bit (int): # of bits (branches) in the star PUF
        lds_a_shape (tuple[int, int]): shape of the PUF matrix
        readout_time (float): readout time of the PUF

    Yields:
        switches (np.ndarray): batch of switches
        mismatch (np.ndarray): batch of mismatch samples
        t (np.ndarray): batch of readout time
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
            size=(inst_per_batch, 2, *lds_a_shape), loc=1.0, scale=0.1
        )
        # Repeat each mismatch samples batch_size // inst_per_batch times
        # and stack them together
        mismatch = np.repeat(mismatches, int(batch_size // inst_per_batch), axis=0)

        # Sanity check: use different mismatch the i2o score should be 0
        # mismatch = np.random.normal(
        #     size=(batch_size, 2, *lds_a_shape), loc=1.0, scale=0.1
        # )
        t = readout_time * np.ones(shape=(batch_size,))
        yield switches, mismatch, t


def I2O_chls(
    inst_per_batch: int,
    chl_per_bit: int,
    n_bit: int,
    lds_a_shape: tuple[int, int],
    readout_time: float,
) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """I2O score training data generator.

    Args:
        chl_per_bit (int): size of the batch
        inst_per_batch (int): # of instance (mismatch)to sample per batch
        (batch_size // inst_per_batch * n_branch should be an integer)
        n_bit (int): # of bits (branches) in the star PUF
        lds_a_shape (tuple[int, int]): shape of the PUF matrix
        readout_time (float): readout time of the PUF

    Yields:
        switches (np.ndarray): batch of switches in the shape of
            (inst_per_batch, chl_per_bit, 1 + n_bit, n_bit)
        mismatch (np.ndarray): batch of mismatch samples in the shape of
            (inst_per_batch, 2, *lds_a_shape)
        t (float): scalar readout time same as the readout_time
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
            size=(inst_per_batch, 2, *lds_a_shape), loc=1.0, scale=0.1
        )
        t = readout_time
        yield switches, mismatch, t


def train(
    model: SwitchableStarPUF,
    loss: FunctionType,
    precise_loss: FunctionType,  # Due to approximation, use another loss function for validation
    dataloader,  # Python generator
    optim: optax.GradientTransformation,
    steps: int,
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
        loss_value, _ = eqx.filter_value_and_grad(precise_loss)(
            model, switch, mismatch, t
        )
        return loss_value

    prev_gmc, prev_gml = model.gm_c, model.gm_l
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
            print(f"gm_c:\n{model.gm_c}")
            print(f"gm_l:\n{model.gm_l}")
            print(f"gmc_diff:\n{model.gm_c - prev_gmc}")
            print(f"gml_diff:\n{model.gm_l - prev_gml}")
            print(f"c_val:\n{model.c_val}")
            print(f"l_val:\n{model.l_val}")
            prev_gmc, prev_gml = model.gm_c, model.gm_l
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, choices=["MatrixExp", "ODE"]
    )
    parser.add_argument("--loss", type=str, default="bf", choices=["bf", "i2o"])
    parser.add_argument("--learning_rate", type=float, default=5e-2)
    parser.add_argument("--n_branch", type=int, default=10)
    parser.add_argument("--line_len", type=int, default=4)
    parser.add_argument("--n_order", type=int, default=40)
    parser.add_argument("--readout_time", type=float, default=10e-9)
    parser.add_argument("--rand_init", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chl_per_bit", type=int, default=64)
    parser.add_argument("--inst_per_batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--logistic_k", type=float, default=40)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    if args.wandb:
        if args.tag:
            wandb_run = wandb.init(
                config=vars(args),
                tags=[args.tag],
            )
        else:
            wandb_run = wandb.init(
                config=vars(args),
            )

    MODEL = args.model
    LOSS = args.loss
    LEARNING_RATE = args.learning_rate
    N_BRANCH = args.n_branch
    LINE_LEN = args.line_len
    N_ORDER = args.n_order
    READOUT_TIME = args.readout_time
    RAND_INIT = args.rand_init
    BATCH_SIZE = args.batch_size
    CHL_PER_BIT = args.chl_per_bit
    INST_PER_BATCH = args.inst_per_batch
    STEPS = args.steps
    PRINT_EVERY = args.print_every
    LOGISTIC_K = args.logistic_k

    print("Config:", vars(args))

    optim = optax.adam(LEARNING_RATE)

    # # Sanity check, single star response is correct
    # rand_loader = partial(random_chls_and_mismatch, readout_time=READOUT_TIME)
    # model = SwitchableStarPUF(n_branch=N_BRANCH, line_len=LINE_LEN, n_order=N_ORDER)
    # for switches, mismatch, _ in rand_loader(BATCH_SIZE, N_BRANCH, model.lds_a_shape):
    #     plot_single_star_rsp(model, switches, mismatch, np.linspace(2e-9, 20e-9, 100))
    #     if input() != "c":
    #         exit()

    if MODEL == "MatrixExp":
        model = SSPUF_MatrixExp(
            n_branch=N_BRANCH,
            line_len=LINE_LEN,
            n_order=N_ORDER,
            random=RAND_INIT,
        )
        if LOSS == "bf":
            loader = bf_chls(
                BATCH_SIZE, INST_PER_BATCH, N_BRANCH, model.lds_a_shape, READOUT_TIME
            )
            train_loss = partial(bf_loss, quantize_fn=partial(logistic, k=LOGISTIC_K))
            train_loss_precise = partial(bf_loss, quantize_fn=step)
        elif LOSS == "i2o":
            loader = I2O_chls(
                INST_PER_BATCH, CHL_PER_BIT, N_BRANCH, model.lds_a_shape, READOUT_TIME
            )
            train_loss = partial(i2o_loss, quantize_fn=partial(logistic, k=LOGISTIC_K))
            train_loss_precise = partial(i2o_loss, quantize_fn=step)

    elif MODEL == "ODE":
        pass
    else:
        raise ValueError("Unknown model")

    # # Sanity Check
    # n_branch, lds_a_shape = model.n_branch, model.lds_a_shape
    # for i, (switches, mismatch, t) in zip(
    #     range(10), bf_chls(BATCH_SIZE, n_branch, lds_a_shape, READOUT_TIME)
    # ):
    #     ideal_loss = i2o_ideal(model, switches, mismatch, t)
    #     approx_loss = i2o_sigmoid(model, switches, mismatch, t)
    #     print(ideal_loss, approx_loss)

    print(model.gm_c, model.gm_l)
    print(model.c_val, model.l_val)
    model = train(
        model=model,
        loss=train_loss,
        precise_loss=train_loss_precise,
        dataloader=loader,
        optim=optim,
        steps=STEPS,
        print_every=PRINT_EVERY,
    )
