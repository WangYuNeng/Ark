""" Test what is the root cause of long compilation time -- 6x10 OBC with all-to-all connections example

Takeaway:
    It is the ode_fn that takes a lot of time to compile. The make_args function does not affect the compilation time.
"""

import time
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from diffrax import Heun
from jaxtyping import PyTree
from ode_fn import ode_fn

from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from make_args import make_args

# ode function input dimension
N_STATE_VAR = 60
N_ARGS = 1950
N_FARGS = 120

# function called in ode_fn
FN_ARGS = [lambda x, y: y * jnp.sin(x) for _ in range(N_FARGS)]

# analog trainable dimension
N_A_TRAINABLE = 1771

time_info = TimeInfo(
    t0=0,
    t1=1,
    dt0=1 / 50,
    saveat=[1],
)


init_trainable = jnp.zeros(N_A_TRAINABLE)


# Test compilation time with forward pass only
def test_forward_pass(test_obj):
    cur_time = time.time()
    eqx.filter_jit(test_obj)
    initial_states = jnp.zeros(N_STATE_VAR)
    test_obj(time_info, initial_states, [], 0, 0)
    print("Forward pass compilation + execution time: ", time.time() - cur_time)


# Test compilation time with forward pass and backward pass
def test_backward_pass(test_obj):
    optim = optax.adam(1e-3)

    def loss_fn(model, data):
        # just an example loss function
        y_raw = jax.vmap(model, in_axes=(None, 0, None, None, None))(
            time_info, data, [], 0, 0
        )
        return jnp.sum(y_raw)

    @eqx.filter_jit
    def make_step(
        model: BaseAnalogCkt,
        opt_state: PyTree,
        loss_fn: Callable,
        data: jax.Array,
    ):
        train_loss, grads = eqx.filter_value_and_grad(loss_fn)(model, data)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, train_loss

    BATCH_SIZE = 128
    opt_state = optim.init(eqx.filter(test_obj, eqx.is_array))
    data = jnp.zeros((BATCH_SIZE, N_STATE_VAR))

    # This will take forever
    cur_time = time.time()
    test_ckt, opt_state, loss = make_step(test_obj, opt_state, loss_fn, data)
    print("Backward pass compilation + execution time: ", time.time() - cur_time)


print("\nOriginal Ark-compiled code")

BaseAnalogCkt.ode_fn = lambda self, t, y, args: ode_fn(t, y, args, FN_ARGS)
BaseAnalogCkt.make_args = make_args
BaseAnalogCkt.readout = lambda self, y: y
test_ckt = BaseAnalogCkt(init_trainable, is_stochastic=False, solver=Heun())
test_forward_pass(test_obj=test_ckt)
test_backward_pass(test_obj=test_ckt)


print("\nWith dummy ode_fn")
BaseAnalogCkt.ode_fn = lambda self, t, y, args: y
init_trainable = jnp.zeros(N_A_TRAINABLE)
test_ckt = BaseAnalogCkt(init_trainable, is_stochastic=False, solver=Heun())
test_forward_pass(test_obj=test_ckt)
test_backward_pass(test_obj=test_ckt)


print("\nWith dummy make_args")
BaseAnalogCkt.ode_fn = lambda self, t, y, args: ode_fn(t, y, args, FN_ARGS)


def dummy_make_args(self, switch, args_seed, gumbel_temp, hard_gumbel):
    args = jnp.zeros(N_ARGS)
    args = args.at[:N_A_TRAINABLE].set(self.a_trainable)
    args = args.at[N_A_TRAINABLE:].set(self.a_trainable[: N_ARGS - N_A_TRAINABLE])
    return args


BaseAnalogCkt.make_args = dummy_make_args
test_forward_pass(test_obj=test_ckt)
test_backward_pass(test_obj=test_ckt)
