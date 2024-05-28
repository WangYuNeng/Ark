from functools import partial
from types import FunctionType
from typing import Callable, Generator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import config
from jaxtyping import PyTree
from spec import ds_scmm_spec
from train import arr_fn, cap_fn, dummy_fn, mse_loss

from ark.ark import Ark
from ark.cdg.cdg import CDG

IMAGE_W, IMAGE_H = 3, 3
TEST_CONV_FILTER = jnp.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])


config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

LEARNING_RATE = 0.1

FILTER_LEN = len(TEST_CONV_FILTER.flatten())
FILTER_W, FILTER_H = TEST_CONV_FILTER.shape
N_BITS = 4

C_RATIO = 100 * (2**N_BITS - 1)
C_BASE = 1e-14
INIT_C1_SCALED = 0.924458
INIT_C1 = INIT_C1_SCALED * C_BASE

# Dummy floating point value
DUMF = 0.0

PHI1 = partial(arr_fn, arr=jnp.array([1, 0] * FILTER_LEN))
PHI2 = partial(arr_fn, arr=jnp.array([0, 1] * FILTER_LEN))


def prepare_data(n_samples):
    """Generate random 'x' vectors and corresponding 'y' values.

    y = conv3x3(x, TEST_CONV_FILTER)
    """
    while True:
        n_conv_w, n_conv_h = IMAGE_W // FILTER_W, IMAGE_H // FILTER_H
        x = np.random.rand(n_samples, IMAGE_W, IMAGE_H)
        y = np.zeros((n_samples, n_conv_w, n_conv_h))
        for i in range(n_samples):
            for w in range(n_conv_w):
                for h in range(n_conv_h):
                    y[i, w, h] = np.sum(
                        x[
                            i,
                            FILTER_W * w : FILTER_W * (w + 1),
                            FILTER_H * h : FILTER_H * (h + 1),
                        ]
                        * TEST_CONV_FILTER
                    )
        y = y.reshape(n_samples, -1)
        yield jnp.asarray(x), jnp.asarray(y)


class Conv3x3(eqx.Module):
    """3x3 convolutional layer (w/o bias) with 3x3 stride"""

    weight: jax.Array
    # Fixed parameters
    c1_scaled: float
    c_ratio: float
    ode_fn: FunctionType
    c_base: float

    def __init__(self, weight: jax.Array, ode_fn: Callable):
        assert weight.shape == (3, 3)
        # For simplicity, use the first "bit" to represent the full value
        # and the the seconde to fourth bits to 0
        weight_flatten = weight.flatten()
        padded_weight = jnp.zeros((9, 4))
        padded_weight = padded_weight.at[:, 0].set(weight_flatten)
        self.weight = padded_weight

        self.c1_scaled = INIT_C1_SCALED
        self.c_ratio = C_RATIO
        self.ode_fn = ode_fn
        self.c_base = C_BASE

    def __call__(self, x: jax.Array):
        width, height = x.shape
        n_conv_w = width // FILTER_W
        n_conv_h = height // FILTER_H
        conv_out = jnp.zeros((n_conv_w, n_conv_h))

        bit_arr_repeat = jnp.repeat(self.weight, 2, axis=0)
        cap_fn_attr = partial(cap_fn, bits_arr=bit_arr_repeat)
        c1 = self.c1_scaled * self.c_base
        c2 = c1 * self.c_ratio
        args = [
            c2,
            DUMF,
            c1 / 100,
            c1,
            0,  # force the 2nd - 4th bits to 0
            0,  # force the 2nd - 4th bits to 0
            0,  # force the 2nd - 4th bits to 0
            DUMF,
            DUMF,
            DUMF,
            DUMF,
            DUMF,
        ]
        n_iter = 2 * FILTER_LEN

        # Can't use lax fori_loop here because of the dynamic indexing of the array
        # This is a workaround but will incur compilation overhead for large input
        # for w in range(n_conv_w):
        #     for h in range(n_conv_h):
        #         x_slice = x[
        #             FILTER_W * w : FILTER_W * (w + 1),
        #             FILTER_H * h : FILTER_H * (h + 1),
        #         ].flatten()
        #         x_slice_repeated = jnp.repeat(x_slice, 2)
        #         vin = partial(arr_fn, arr=x_slice_repeated)

        #         fargs = [cap_fn_attr, vin, PHI1, PHI2]

        #         charge_trace = jnp.zeros((n_iter + 1, 2))
        #         for i in range(1, n_iter + 1):
        #             charge_trace = charge_trace.at[i].set(
        #                 self.ode_fn(i - 1, charge_trace[i - 1], args, fargs)
        #             )

        #         readout = charge_trace[-1, 0]
        #         # V = Q / C and scale the voltage to match the original range
        #         voltage = readout / c2 * self.c_ratio
        #         conv_out = conv_out.at[w, h].set(-voltage)

        def conv_body_fn(idx: int, val: jax.Array):
            """Perform convolution with scmm"""

            w = idx // n_conv_w
            h = idx % n_conv_h
            x_slice = jax.lax.dynamic_slice(
                x, (FILTER_W * w, FILTER_H * h), (FILTER_W, FILTER_H)
            ).flatten()
            x_slice_repeated = jnp.repeat(x_slice, 2)
            vin = partial(arr_fn, arr=x_slice_repeated)

            fargs = [cap_fn_attr, vin, PHI1, PHI2]

            charge_trace = jnp.zeros((n_iter + 1, 2))
            for i in range(1, n_iter + 1):
                charge_trace = charge_trace.at[i].set(
                    self.ode_fn(i - 1, charge_trace[i - 1], args, fargs)
                )

            readout = charge_trace[-1, 0]
            # V = Q / C and scale the voltage to match the original range
            voltage = readout / c2 * self.c_ratio
            val = val.at[w, h].set(-voltage)
            return val

        conv_out = jax.lax.fori_loop(
            lower=0, upper=n_conv_w * n_conv_h, body_fun=conv_body_fn, init_val=conv_out
        )
        return conv_out.flatten()


def train(
    model: Conv3x3,
    loss: FunctionType,
    dataloader: Generator[jax.Array, jax.Array, jax.typing.DTypeLike],
    optim: optax.GradientTransformation,
    steps: int,
):

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state: PyTree, x: jax.Array, y: jax.Array):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for step, (x, y) in zip(range(steps), dataloader):

        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        print(f"{step=}, loss={train_loss.item()}")
        print(f"true_weight=\n{TEST_CONV_FILTER}")
        print(f"trained_weight=\n{model.weight[:, 0].reshape(FILTER_W, FILTER_H)}")
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
    init_weight = np.random.random(size=(3, 3))
    scmm_model = Conv3x3(
        weight=jnp.array(init_weight, dtype=jnp.float64),
        ode_fn=ode_fn,
    )

    optim = optax.adam(LEARNING_RATE)
    train(
        model=scmm_model,
        loss=mse_loss,
        dataloader=prepare_data(32),
        optim=optim,
        steps=32,
    )
