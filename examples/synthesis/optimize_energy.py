from itertools import product
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, DTypeLike
from obc_syn import random_simulation, synthesize_general, validate_synthesis

jax.config.update("jax_enable_x64", True)


class Energy(eqx.Module):
    J_raw: Array
    Kc: DTypeLike
    Kl: DTypeLike
    n_harmonic: float = 2.0

    def __init__(self, J, Kc, Kl, n_harmonic=2):
        self.J_raw = J
        self.Kc = Kc
        self.Kl = Kl
        self.n_harmonic = float(n_harmonic)

    def __call__(self, phases):
        aux_phase = jnp.pi
        p = jnp.concatenate([phases, jnp.array([aux_phase])])
        cos_diff = jnp.cos(p[:, None] - p[None, :])
        coupling_energy = 1 * jnp.sum(self.J * cos_diff)
        locking_energy = 1 * jnp.sum(jnp.cos(2 * p))
        return coupling_energy + locking_energy

    @property
    def J(self) -> Array:
        j = 0.5 * (self.J_raw + self.J_raw.T)
        j_normalized = j / jnp.mean(jnp.abs(j))
        j_zero_diag = j_normalized - jnp.diag(jnp.diag(j_normalized))
        return j_zero_diag


def loss_diff(model: Energy, p_pos: Array, p_neg: Array) -> DTypeLike:
    e_pos = jax.vmap(model)(p_pos)
    e_neg = jax.vmap(model)(p_neg)
    return jnp.mean(e_pos) - jnp.mean(e_neg)


@eqx.filter_jit
def step(
    model: Energy,
    opt_state: optax.OptState,
    p_pos: Array,
    p_neg: Array,
    optimizer: optax.GradientTransformation,
):
    loss_value, grads = jax.value_and_grad(loss_diff)(model, p_pos, p_neg)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


def gen_data(
    batch_size: int, fn: Callable, n_io_var: int, n_aux_osc: int, n_harmonic: int
):
    n_total = n_io_var + n_aux_osc
    p_pos_center, p_neg_center = [], []
    phase_sep = 2 * jnp.pi / n_harmonic
    for vars in product(range(n_harmonic), repeat=n_total):
        if fn(*vars[:n_io_var]):
            p_pos_center.append([v * phase_sep for v in vars])
        else:
            p_neg_center.append([v * phase_sep for v in vars])

    def sample_and_perturb(centers: list, n: int):
        # Sample length-n arrays from centers with uniform perturbation
        # within +/- phase_sep/2
        sample_idxs = np.random.choice(len(centers), size=n)
        phases = jnp.array([centers[i] for i in sample_idxs])
        perturb = np.random.uniform(
            low=-phase_sep / 2 * 0.1, high=phase_sep / 2 * 0.1, size=phases.shape
        )
        return phases + perturb

    while True:
        p_pos = sample_and_perturb(p_pos_center, batch_size)
        p_neg = sample_and_perturb(p_neg_center, batch_size)

        yield p_pos, p_neg


if __name__ == "__main__":

    optimizer = optax.adam(learning_rate=0.1)
    n_epochs = 100
    batch_size = 1024
    gate_func = lambda x, y, z: x | y == z
    n_io_var = 3
    n_aux_osc = 0
    Kc = 1.0
    Kl = 1.0
    n_harmonic = 2
    # np.random.seed(428)

    j_mat = synthesize_general(
        gate_func,
        n_io_var=n_io_var,
        n_aux_osc=n_aux_osc,
        symmetric=True,
        constrain_coupling=0,
        constrain_energy_threshold=False,
    )
    j_val = jnp.array(j_mat, dtype=jnp.float64)
    # j_val = jnp.array(np.random.uniform(-1, 1, size=j_val.shape))  # Random init

    dataloader = gen_data(
        batch_size=batch_size,
        fn=gate_func,
        n_io_var=n_io_var,
        n_aux_osc=n_aux_osc,
        n_harmonic=n_harmonic,
    )
    energy_model = Energy(J=j_val, Kc=Kc, Kl=Kl, n_harmonic=n_harmonic)
    opt_state = optimizer.init(energy_model)
    print(energy_model.J)
    best_loss = float("inf")
    best_j = None
    for epoch in range(n_epochs):
        p_pos, p_neg = next(dataloader)
        energy_model, opt_state, loss_value = step(
            energy_model, opt_state, p_pos, p_neg, optimizer
        )
        print(f"Epoch {epoch}: Loss = {loss_value}")
        if loss_value < best_loss:
            best_loss = loss_value
            best_j = energy_model.J.copy()

    print("Optimized J matrix:")
    print(best_loss, best_j)
    print(energy_model.Kc, energy_model.Kl, energy_model.n_harmonic)

    validate_synthesis(
        J_mat=best_j,
        logic_fn=gate_func,
        n_io_var=n_io_var,
        n_aux_osc=n_aux_osc,
        Kc=energy_model.Kc,
        Kl=energy_model.Kl,
        anneal=True,
        plot=True,
        t_span=(0, 10),
    )
