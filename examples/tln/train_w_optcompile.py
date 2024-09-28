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
from spec import IdealE, InpI, MmE, MmI, MmV, lc_range, mm_tln_spec, w_range

from ark.cdg.cdg import CDG, CDGEdge
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=5e-2)
parser.add_argument("--n_branch", type=int, default=10)
parser.add_argument("--line_len", type=int, default=4)
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
parser.add_argument(
    "--save_weight", type=str, default=None, help="Path to save weights"
)
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

time_info = TimeInfo(
    t0=0.0, t1=READOUT_TIME, dt0=READOUT_TIME / N_TIME_POINTS, saveat=[READOUT_TIME]
)


def plot_single_star_rsp(model, init_vals, switch, mismatch):
    """Sanity check: Plot the transient response of a single star

    Should behave like transmission line.
    """
    time_points = np.arange(0, READOUT_TIME, READOUT_TIME / N_TIME_POINTS)
    readout_trace_time_info = TimeInfo(
        t0=0.0,
        t1=READOUT_TIME,
        dt0=READOUT_TIME / N_TIME_POINTS,
        saveat=time_points,
    )

    switch_val = switch[0][0][0]
    trace = model(readout_trace_time_info, init_vals, switch_val, mismatch[0], 0)
    trace = np.array(trace).squeeze()
    plt.title(f"switch={switch_val}")
    plt.plot(time_points, trace[:, 0])
    plt.plot(time_points, trace[:, 1])
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
    init_vals: Array,
    switch: Array,
    mismatch: Array,
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
    inst_per_batch, chl_per_bit, _, double_n_bit = switch.shape
    n_bit = double_n_bit // 2
    chl_per_inst = chl_per_bit * (1 + n_bit)
    batch_size = inst_per_batch * chl_per_inst

    # The forward pass of the model
    # First two dimensions (time_info and initial states) are identical
    # across the whole batch.
    # Switch is uniquely chose for each instance.
    # Mismatch seed is shared per instance and require to be broadcasted.
    # No transient noise in the simulation so use an arbitrary noise seed.
    analog_out = jax.vmap(model, in_axes=(None, None, 0, 0, None))(
        time_info,
        init_vals,
        switch.reshape(batch_size, -1),
        mismatch.repeat(chl_per_inst, axis=0),
        0,
    ).reshape(-1, 2)
    # Take the difference between the two stars
    out_diff = analog_out[:, 0] - analog_out[:, 1]
    digital_out: Array = quantize_fn(out_diff).reshape(-1, 2)
    # Reshape to calculate the I2O score
    digital_out = digital_out.reshape(inst_per_batch, chl_per_bit, n_bit + 1)

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
    puf_cdg: CDG,
    switch_pairs: tuple[list[CDGEdge], list[CDGEdge]],
    puf_ckt_class: BaseAnalogCkt,
) -> Generator[Array, Array, jax.typing.DTypeLike]:
    """I2O score training data generator.

    Args:
        chl_per_bit (int): size of the batch
        inst_per_batch (int): # of instance (mismatch)to sample per batch
        (batch_size // inst_per_batch * n_branch should be an integer)
        n_bit (int): # of bits (branches) in the star PUF
        puf_cdg (CDG): the CDG of the PUF
        switch_pairs (tuple[list[CDGEdge], list[CDGEdge]]): the CDG edges controlling
        the switches
        puf_ckt_class (BaseAnalogCkt): the PUF circuit class

    Yields:
        initial_states (Array): initial states of the PUF state variables which is
        a all-zero vector
        switches (Array): batch of switches in the shape of
            (inst_per_batch, chl_per_bit, 1 + n_bit, 2 * n_bit)
            The switches maps to edges in the cdg and because the puf has
            two nominally identical branches, the switches are repeated twice
        mismatch_seed (Array): batch of mismatch seed in the shape of
            (inst_per_batch)
    """

    def switch_vals_to_model_switch_input(switch_vals):
        # Set the switch values to the correct cdg edges in the puf_cdg and map
        # to the model input format
        for val, switch0, switch1 in zip(switch_vals, *switch_pairs):
            switch0.switch_val = val
            switch1.switch_val = val
        return puf_ckt_class.cdg_to_switch_array(puf_cdg)

    # Set all the nodes to have initial state 0
    for node in puf_cdg.stateful_nodes:
        node.set_init_val(val=0, n=0)
    init_state_arr = puf_ckt_class.cdg_to_initial_states(puf_cdg)

    while True:
        switches = []
        for _ in range(inst_per_batch):
            switch_in_inst = []
            for _ in range(chl_per_bit):
                base_chl = np.random.randint(0, 2, size=(n_bit))
                flipped_chls = [switch_vals_to_model_switch_input(base_chl)]
                for i in range(n_bit):
                    flipped_chl = base_chl.copy()
                    flipped_chl[i] ^= 1
                    flipped_chls.append(switch_vals_to_model_switch_input(flipped_chl))
                switch_in_inst.append(flipped_chls)
            switches.append(switch_in_inst)

        switches = np.array(switches)
        mismatch = np.random.randint(0, 2**32 - 1, size=inst_per_batch)
        yield jnp.array(init_state_arr), jnp.array(switches), jnp.array(mismatch)


def train(
    model: BaseAnalogCkt,
    loss: FunctionType,
    precise_loss: FunctionType,  # Due to approximation, use another loss function for validation
    dataloader: Generator[Array, Array, jax.typing.DTypeLike],
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
        init_vals: Array,
        switch: Array,
        mismatch: Array,
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(
            model, init_vals, switch, mismatch
        )
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    @eqx.filter_jit
    def validate(
        model: BaseAnalogCkt,
        init_vals: Array,
        switch: Array,
        mismatch: Array,
    ):
        loss_value, _ = eqx.filter_value_and_grad(precise_loss)(
            model, init_vals, switch, mismatch
        )
        return loss_value

    best_loss_precise = 0.5  # Upper bound of the i2o and bit-flipping test loss
    best_weight = (model.a_trainable.copy(), model.d_trainable.copy())
    for step, (init_vals, switches, mismatch) in zip(range(steps), dataloader):
        if (step % print_every) == 0 or (step == steps - 1):
            train_loss_precise = validate(model, init_vals, switches, mismatch)
        model, opt_state, train_loss = make_step(
            model, opt_state, init_vals, switches, mismatch
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
            print(model.a_trainable)

        if train_loss_precise < best_loss_precise:
            best_loss_precise = train_loss_precise
            best_weight = model.weights()
            eqx.tree_serialise_leaves(checkpoint, model)
    return best_loss_precise, best_weight


if __name__ == "__main__":

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

        lc_val = (
            normalize(1e-9, lc_range.min, lc_range.max) if NORMALIZE_WEIGHT else 1e-9
        )
        gm_val = normalize(1, w_range.min, w_range.max) if NORMALIZE_WEIGHT else 1
        puf_params.middle_cap.init_val = (
            normalize(1e-9, lc_range.min, lc_range.max) if NORMALIZE_WEIGHT else 1e-9
        )
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

    loader = I2O_chls(
        INST_PER_BATCH, CHL_PER_BIT, N_BRANCH, puf_cdg, switch_pairs, puf_ckt_class
    )
    train_loss = partial(i2o_loss, quantize_fn=partial(logistic, k=LOGISTIC_K))
    train_loss_precise = partial(i2o_loss, quantize_fn=step)

    trainable_init = (mgr.get_initial_vals("analog"), mgr.get_initial_vals("digital"))
    print(f"Trainable init: {trainable_init}")

    model: BaseAnalogCkt = puf_ckt_class(
        init_trainable=trainable_init,
        is_stochastic=False,
        solver=Heun(),
        # init_trainable=trainable_init,
        # is_stochastic=False,
        # solver=Tsit5(),
    )

    # for init_vals, switches, mismatch in loader:
    #     plot_single_star_rsp(
    #         model,
    #         init_vals,
    #         switches,
    #         mismatch,
    #     )
    best_loss, best_weight = train(
        model=model,
        loss=train_loss,
        precise_loss=train_loss_precise,
        dataloader=loader,
        optim=optim,
        steps=STEPS,
        checkpoint=CHECKPOINT,
        print_every=PRINT_EVERY,
    )

    if args.save_weight:
        jnp.savez(args.save_weight, analog=best_weight[0], digital=best_weight[1])
