# Toy example to demonstrate how gradient descent
# is done in Ark for optimize scmm's parameters
# Objective: fitting a predefined weighted sum
# E.g., y = [0.25, 0.5, 0.25] * x
#
# Can play with the C_RATIO parameter (e.g., 1) to see
# that the model can learn the weights considering the
# incomplete charge transfer, better than just using
# the digital weight value (just a toy example though,
# the discrete bits will add restrictions).
#
# TODO: How to perform differentiable quantization
# or constrain the bits?


from functools import partial
from types import FunctionType
from typing import Generator

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import config
from jaxtyping import PyTree
from spec import ds_scmm_spec

from ark.ark import Ark
from ark.cdg.cdg import CDG

config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

LEARNING_RATE = 0.1

FILTER = np.array([1, 3, 1])
FILTER_LEN = len(FILTER)
N_BITS = 4

C_RATIO = 10 * (2**N_BITS - 1)
C_BASE = 1e-14
INIT_C1_SCALED = 0.924458
INIT_C1 = INIT_C1_SCALED * C_BASE

# Dummy floating point value
DUMF = 0.0


def prepare_data(n_samples):
    """Generate random 'x' vectors and corresponding 'y' values.

    y = FILTER.T  x
    """
    while True:
        x_len = FILTER_LEN
        x = np.random.rand(n_samples, x_len)
        y = np.dot(x, FILTER)
        yield jnp.asarray(x), jnp.asarray(y)


def mse_loss(model, x, y):
    """Mean squared error loss function"""
    y_pred = jax.vmap(model)(x)
    return jnp.mean(jnp.square(y - y_pred))


# Redefine functions for function attributes in the spec
def arr_fn(t: int, arr: jax.Array):
    """Make a array to an function to be compatible with Ark"""
    return arr[t]


PHI1 = partial(arr_fn, arr=jnp.array([1, 0] * FILTER_LEN))
PHI2 = partial(arr_fn, arr=jnp.array([0, 1] * FILTER_LEN))


def cap_fn(
    t: int,
    cbase: jax.typing.DTypeLike,
    c0: jax.typing.DTypeLike,
    c1: jax.typing.DTypeLike,
    c2: jax.typing.DTypeLike,
    c3: jax.typing.DTypeLike,
    bits_arr: jax.Array,
):
    """Return the cap value at a given cycle"""
    bits = bits_arr[t]
    cap_val = cbase + bits.dot(jnp.array([c0, c1, c2, c3]))
    return cap_val


def dummy_fn(t):
    """Dummy function for the time being"""
    return 0.0


# Model and training setup


def int2bit(x: int, n_bits: int):
    """Convert an integer to a bit array in little-endian format"""
    return np.array([int(x) for x in np.binary_repr(x, width=n_bits)[::-1]])


class Scmm(eqx.Module):

    # Training parameters
    weight_bits: jax.Array

    # Fixed parameters
    c1_scaled: float
    c_ratio: float
    ode_fn: FunctionType
    c_base: float

    def __init__(
        self,
        weight_bits: jax.Array,
        c1_scaled: float,
        c_ratio: float,
        ode_fn: FunctionType,
        c_base: float,
    ):
        assert weight_bits.shape == (FILTER_LEN, N_BITS)
        self.weight_bits = weight_bits
        self.c1_scaled = c1_scaled
        self.c_ratio = c_ratio
        self.ode_fn = ode_fn
        self.c_base = c_base

    def __call__(self, x: jax.Array):
        bit_arr_repeat = jnp.repeat(self.weight_bits, 2, axis=0)
        x_repeated = jnp.repeat(x, 2)
        cap_fn_attr = partial(cap_fn, bits_arr=bit_arr_repeat)
        vin = partial(arr_fn, arr=x_repeated)
        c1 = self.c1_scaled * self.c_base
        c2 = c1 * self.c_ratio
        args = [
            c2,
            DUMF,
            c1 / 100,
            c1,
            c1 * 2,
            c1 * 4,
            c1 * 8,
            DUMF,
            DUMF,
            DUMF,
            DUMF,
            DUMF,
        ]
        fargs = [cap_fn_attr, vin, PHI1, PHI2]
        n_iter = 2 * FILTER_LEN
        charge_trace = jnp.zeros((n_iter + 1, 2))
        for i in range(1, n_iter + 1):
            charge_trace = charge_trace.at[i].set(
                self.ode_fn(i - 1, charge_trace[i - 1], args, fargs)
            )

        readout = charge_trace[-1, 0]
        # V = Q / C and scale the voltage to match the original range
        voltage = readout / c2 * self.c_ratio
        return -voltage


def train(
    model,
    loss: FunctionType,
    dataloader: Generator[jax.Array, jax.Array, jax.typing.DTypeLike],
    optim: optax.GradientTransformation,
    steps: int,
):

    naive_weight = jnp.array([int2bit(x, N_BITS) for x in FILTER], dtype=jnp.float64)
    naive_weight_model = Scmm(
        weight_bits=naive_weight,
        c1_scaled=INIT_C1_SCALED,
        c_ratio=C_RATIO,
        ode_fn=ode_fn,
        c_base=C_BASE,
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state: PyTree, x: jax.Array, y: jax.Array):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        loss_naive_weight = loss(naive_weight_model, x, y)
        return model, opt_state, loss_value, loss_naive_weight

    bit_2_dec = lambda bits: bits.dot(2 ** jnp.arange(N_BITS))

    for step, (x, y) in zip(range(steps), dataloader):

        model, opt_state, train_loss, loss_nw = make_step(model, opt_state, x, y)
        print(f"{step=}, loss={train_loss.item()}, loss_naive_weight={loss_nw.item()}")
        print(f"filter_weight_bit=\n{model.weight_bits}")
        print(f"filter_weight_dec=\n{bit_2_dec(model.weight_bits)}")
    return model


if __name__ == "__main__":

    # Create the cdg and compile the system
    system = Ark(cdg_spec=ds_scmm_spec)

    Inp = ds_scmm_spec.node_type("InpV")
    CapWeight = ds_scmm_spec.node_type("CapWeight")
    CapSAR = ds_scmm_spec.node_type("CapSAR")
    Sw = ds_scmm_spec.edge_type("Sw")

    scmm = CDG()
    inp = Inp(vin=arr_fn, r=DUMF)
    cweight = CapWeight(
        c=cap_fn, Vm=DUMF, cbase=DUMF, c0=DUMF, c1=DUMF, c2=DUMF, c3=DUMF
    )
    csar = CapSAR(c=DUMF)
    sw1 = Sw(ctrl=dummy_fn, Gon=DUMF, Goff=DUMF)
    sw2 = Sw(ctrl=dummy_fn, Gon=DUMF, Goff=DUMF)
    scmm.connect(sw1, inp, cweight)
    scmm.connect(sw2, cweight, csar)

    compiler = system.compiler
    ode_fn, _, _, _, _ = compiler.compile_odeterm(cdg=scmm, cdg_spec=ds_scmm_spec)
    init_weight_bits = np.random.randint(2, size=(FILTER_LEN, N_BITS))
    scmm_model = Scmm(
        weight_bits=jnp.array(init_weight_bits, dtype=jnp.float64),
        c1_scaled=INIT_C1_SCALED,
        c_ratio=C_RATIO,
        ode_fn=ode_fn,
        c_base=C_BASE,
    )

    optim = optax.adam(LEARNING_RATE)
    train(
        model=scmm_model,
        loss=mse_loss,
        dataloader=prepare_data(32),
        optim=optim,
        steps=128,
    )
