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
from diffrax import Heun
from jaxtyping import Array, PyTree
from puf import PUFParams, create_switchable_star_cdg
from spec import IdealE, InpI, MmE, MmI, MmV, lc_range, mm_tln_spec, pulse, w_range

from ark.cdg.cdg import CDG
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.specification.trainable import Trainable, TrainableMgr


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


def i2o_loss(
    model: BaseAnalogCkt,
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
    analog_out = jax.vmap(model, in_axes=(0, 0))(
        switch.reshape(-1, n_bit),
        mismatch.repeat(chl_per_bit * (1 + n_bit), axis=0).reshape(
            -1, 2, model.mismatch_len
        ),
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


def I2O_chls(
    inst_per_batch: int,
    chl_per_bit: int,
    n_bit: int,
) -> Generator[jax.Array, jax.Array, jax.typing.DTypeLike]:
    """I2O score training data generator.

    Args:
        chl_per_bit (int): size of the batch
        inst_per_batch (int): # of instance (mismatch)to sample per batch
        (batch_size // inst_per_batch * n_branch should be an integer)
        n_bit (int): # of bits (branches) in the star PUF

    Yields:
        switches (jax.Array): batch of switches in the shape of
            (inst_per_batch, chl_per_bit, 1 + n_bit, n_bit)
        mismatch_seed (jax.Array): batch of mismatch seed in the shape of
            (inst_per_batch)
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
        mismatch = np.random.randint(0, 2**32 - 1, size=inst_per_batch)
        yield jnp.array(switches), jnp.array(mismatch)


def train(
    model: BaseAnalogCkt,
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
        model: BaseAnalogCkt,
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
        model: BaseAnalogCkt,
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

        if train_loss_precise < best_loss_precise:
            best_loss_precise = train_loss_precise
            eqx.tree_serialise_leaves(checkpoint, model)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-2)
    parser.add_argument("--n_branch", type=int, default=10)
    parser.add_argument("--line_len", type=int, default=4)
    parser.add_argument("--n_order", type=int, default=40)
    parser.add_argument("--n_time_points", type=int, default=100)
    parser.add_argument("--readout_time", type=float, default=10e-9)
    parser.add_argument("--rand_init", action="store_true")
    parser.add_argument("--normalize_weight", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chl_per_bit", type=int, default=64)
    parser.add_argument("--inst_per_batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--logistic_k", type=float, default=40)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()
    np.random.seed(args.seed)

    CHECKPOINT = f"checkpoint/{time.strftime('%Y%m%d-%H%M%S')}_{time.time():.0f}.eqx"

    train_config = vars(args)
    train_config["checkpoint"] = CHECKPOINT

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

    LEARNING_RATE = args.learning_rate
    N_BRANCH = args.n_branch
    LINE_LEN = args.line_len
    N_ORDER = args.n_order
    N_TIME_POINTS = args.n_time_points
    READOUT_TIME = args.readout_time
    RAND_INIT = args.rand_init
    NORMALIZE_WEIGHT = args.normalize_weight
    BATCH_SIZE = args.batch_size
    CHL_PER_BIT = args.chl_per_bit
    INST_PER_BATCH = args.inst_per_batch
    STEPS = args.steps
    PRINT_EVERY = args.print_every
    LOGISTIC_K = args.logistic_k

    print("Config:", train_config)

    optim = optax.adam(LEARNING_RATE)

    puf_params: PUFParams
    puf_cdg, middle_caps, switch_pairs, branch_pairs, puf_params = (
        create_switchable_star_cdg(
            n_bits=N_BRANCH,
            line_len=LINE_LEN,
            v_nt=MmV,
            i_nt=MmI,
            et=MmE,
            self_et=IdealE,
            inp_nt=InpI,
        )
    )

    if RAND_INIT:
        raise NotImplementedError
    else:

        def normalize(val, low, high):
            # Normalize val to [-1, 1]
            return 2 * (val - low) / (high - low) - 1

        lc_val = normalize(1e-9, lc_range[0], lc_range[1]) if NORMALIZE_WEIGHT else 1e-9
        gm_val = normalize(1, w_range[0], w_range[1]) if NORMALIZE_WEIGHT else 1
        puf_params.middle_cap.init_val = 1e-9
        for cap, ind in zip(puf_params.branch_caps, puf_params.branch_inds):
            cap.init_val = lc_val
            ind.init_val = lc_val
        for gm0, gm1 in zip(puf_params.branch_gms[0], puf_params.branch_gms[1]):
            gm0.init_val = gm_val
            gm1.init_val = gm_val

    mgr = puf_params.mgr
    puf_ckt_class = OptCompiler().compile(
        prog_name="puf",
        cdg=puf_cdg,
        cdg_spec=mm_tln_spec,
        trainable_mgr=mgr,
        readout_nodes=[middle_caps[0], middle_caps[1]],
        normalize_weight=NORMALIZE_WEIGHT,
        aggregate_args_lines=True,
    )

    loader = I2O_chls(INST_PER_BATCH, CHL_PER_BIT, N_BRANCH)
    train_loss = partial(i2o_loss, quantize_fn=partial(logistic, k=LOGISTIC_K))
    train_loss_precise = partial(i2o_loss, quantize_fn=step)

    trainable_init = (mgr.get_initial_vals("analog"), mgr.get_initial_vals("digital"))
    print(f"Trainable init: {trainable_init}")

    model: BaseAnalogCkt = puf_ckt_class(
        init_trainable=trainable_init, is_stochastic=False, solver=Heun()
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
