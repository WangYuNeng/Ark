import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Tsit5

# jax.config.update("jax_enable_x64", True)
T = 1 / 795.8e6


def f(x, s: float):
    return s * 795.8e6 * jnp.sin(2 * jnp.pi * x)


class TestModel(eqx.Module):

    trainable: jax.Array

    # Future todo: Make time info and initial state trainable if needed

    def __init__(
        self,
        init_trainable: jax.Array,
    ) -> None:
        self.trainable = init_trainable

    def __call__(
        self,
        initial_state: jax.Array,
    ):
        args = self.trainable

        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.ode_fn),
            solver=Tsit5(),
            t0=0,
            t1=T * 10,
            dt0=T / 10,
            y0=initial_state,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0, T * 10, 100)),
            args=args,
        )

        return solution.ys

    def ode_fn(self, time, __variables, args):
        (
            s0,
            _,
            s2,
            _,
            k0,
            k1,
        ) = args
        x0, x1 = __variables
        n_x0 = -k0 * f(x0, s0)
        n_x1 = -k1 * f(x1, s2)
        return jnp.array([n_x0, n_x1])


np.random.seed(2)
init_trainable = jnp.array(np.random.normal(size=6))

model = TestModel(init_trainable=init_trainable)
xs = jnp.array([[0, 1], [1, 0]], dtype=jnp.float32)
output_loop = jnp.array([model(x) for x in xs])

output_vmap = jax.vmap(model, in_axes=(0))(xs)

for out_vmap, out_loop in zip(output_vmap, output_loop):
    print(f"Vmap: {out_vmap[-5:]}")
    print(f"Loop: {out_loop[-5:]}")
assert jnp.allclose(output_vmap, output_loop)  # Failed
