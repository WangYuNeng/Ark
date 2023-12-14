"""
Model the SSPUF and Bit-flipping test as a differentiable program.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config
from tqdm import tqdm

config.update("jax_debug_nans", True)


class SwitchableStar(eqx.Module):
    """SwitchableStar topology of transmission lines.

    - n_branch: # of branches to switch
    - line_len: # of LC section in each branch
    - n_order: # of order to approximate the matrix exponential
    - gm_c: Gm of integrators associated with capacitors
    - gm_l: Gm of integrators associated with inductors
    - lc_val: The value of the LC components
    - lds_a: The A matrix in the state-space representation of the
    SwithcableStar.
    - t: The time point of the output
    """

    n_branch: int
    line_len: int
    n_order: int
    gm_c: jax.Array
    gm_l: jax.Array
    lc_val: int = 1e-9
    pulse_amplitude: float = 1.0
    pulse_t1: float = 0.5e-9
    pulse_t2: float = 1.5e-9
    pulse_t3: float = 2.0e-9
    lds_a: jax.Array
    t: jax.typing.DTypeLike

    def __init__(
        self, n_branch: int, line_len: int, n_order: int = 40, random: bool = False
    ) -> None:
        """Initialize the SwitchableStar topology.

        Args:
            n_branch (int): # of branches to switch.
            line_len (int): # of LC section in each branch.
            n_order (int, optional): # of order to approximate the matrix exponential.
            Defaults to 40.
            random (bool, optional): Randomize the initial state.
            Defaults to False, initializing with gm_c=gm_l=1.0, t=20e-9.
        """
        self.n_branch, self.line_len = n_branch, line_len
        self.n_order = n_order
        self.gm_c = np.ones(shape=(n_branch, line_len, 2))
        self.gm_l = np.ones(shape=(n_branch, line_len, 2))

        if random:
            raise NotImplementedError

        else:
            # Construct the gm part of the A matrix
            sub_mat_len = 2 * line_len

            def ith_diag_sub_mat(i: int):
                a_i = np.zeros(shape=(sub_mat_len, sub_mat_len))
                for j in range(sub_mat_len - 1):
                    if j % 2 == 0:
                        c_idx = j // 2
                        a_i[j, j + 1] = -self.gm_c[i][c_idx][1]
                        a_i[j + 1, j] = self.gm_c[i][c_idx][0]
                    else:
                        l_idx = j // 2 + 1
                        a_i[j, j + 1] = -self.gm_l[i][l_idx][1]
                        a_i[j + 1, j] = self.gm_l[i][l_idx][0]
                return a_i

            ai_row_mats = [ith_diag_sub_mat(i) for i in range(n_branch)]
            for i, diag_mat in enumerate(ai_row_mats):
                left_zeros = np.zeros(shape=(sub_mat_len, sub_mat_len * i))
                right_zeros = np.zeros(
                    shape=(sub_mat_len, sub_mat_len * (n_branch - i - 1))
                )
                ai_row_mats[i] = np.hstack([left_zeros, diag_mat, right_zeros])

            mid_to_branch_row = np.zeros(shape=(1, sub_mat_len * n_branch))
            mid_to_branch_col = np.zeros(shape=(sub_mat_len * n_branch, 1))
            for i in range(n_branch):
                mid_to_branch_row[0, i * sub_mat_len] = -self.gm_l[i][0][1]
                mid_to_branch_col[i * sub_mat_len, 0] = self.gm_c[i][0][0]

            A_1_to_n = np.vstack([mid_to_branch_row] + ai_row_mats)

            A_mat = np.hstack(
                [np.vstack([np.zeros((1, 1)), mid_to_branch_col]), A_1_to_n]
            )

            # Scale by LC values
            LC_mat = np.diag([1 / self.lc_val] * A_mat.shape[0])

            self.lds_a = jnp.array(np.matmul(LC_mat, A_mat))
            self.t = 20e-9

    def __call__(
        self, switch: jax.Array, mismatch: jax.Array, t: jax.typing.DTypeLike
    ) -> jax.typing.DTypeLike:
        """Compute the

        Args:
            switch (jax.Array): An array of {0,1} value denoting whether the branches
            are ON or OFF.
            mismatch (jax.Array): Array of random value to model the mismatch.

        Returns:
            jax.typing.DTypeLike: The output of the SwitchableStar at time t.
        """
        a_sw_mm = jnp.multiply(self.lds_a, mismatch)
        for i, sw_val in enumerate(switch):
            a_sw_mm = a_sw_mm.at[0, 1 + i * self.line_len * 2].multiply(sw_val)

        b_mat = jnp.zeros(shape=(self.lds_a.shape[0], 1))
        b_mat = b_mat.at[0, 0].set(1 / self.lc_val)
        c_mat = jnp.zeros(shape=(1, self.lds_a.shape[0]))
        c_mat = c_mat.at[0, 0].set(1)

        amp, t1, t2, t3 = (
            self.pulse_amplitude,
            self.pulse_t1,
            self.pulse_t2,
            self.pulse_t3,
        )
        expm_t = jax.scipy.linalg.expm(a_sw_mm * t)
        expm_t1_integrals = self._calc_expm_t_integrals(-a_sw_mm, t1)
        expm_t2_integrals = self._calc_expm_t_integrals(-a_sw_mm, t2)
        expm_t3_integrals = self._calc_expm_t_integrals(-a_sw_mm, t3)

        input_integral = (
            1 / t1 * jnp.matmul(expm_t1_integrals[1], b_mat)
        )  # rise integral
        input_integral += jnp.matmul(
            (expm_t2_integrals[0] - expm_t1_integrals[0]), b_mat
        )  # hold integral
        input_integral += (
            1
            / (t2 - t3)
            * jnp.matmul(
                (
                    expm_t3_integrals[1]
                    - expm_t2_integrals[1]
                    - t3 * (expm_t3_integrals[0] - expm_t2_integrals[0])
                ),
                b_mat,
            )
        )  # fall integral
        sol = amp * jnp.matmul(jnp.matmul(c_mat, expm_t), input_integral)
        return sol

    def _calc_expm_t_integrals(
        self, mat: jax.Array, tau: jax.typing.DTypeLike
    ) -> jax.Array:
        """Calculate integrals of exp(mat * t) * dt and exp(mat * t) * t * dt

        Args:
            mat (jax.Array): The matrix to exponentiate.
            tau (jax.typing.DTypeLike): Time value.

        Returns:
            jax.Array: (2, mat.shape[0], mat.shape[1]) that stores the integrals
            of exp(mat * t) at the 0 indice and exp(mat * t) * t  at the 1 indice.
        """

        at_i_over_factorial_i = (
            jnp.eye(mat.shape[0]) * tau
        )  # (A^i * t^(i+1) / factorial(i+1)
        integral_exp_mat = at_i_over_factorial_i  # integral(e^at)
        integral_exp_mat_t = tau * at_i_over_factorial_i / 2  # integral(e^at) * t
        for i in range(1, self.n_order + 1):
            at_i_over_factorial_i = (
                jnp.matmul(mat, at_i_over_factorial_i) * tau / (i + 1)
            )
            integral_exp_mat += at_i_over_factorial_i
            integral_exp_mat_t += at_i_over_factorial_i * tau * (i + 1) / (i + 2)
        return jnp.stack([integral_exp_mat, integral_exp_mat_t])


test = SwitchableStar(n_branch=1, line_len=1)
print(test.lds_a)
switches = jnp.array([1])
mismatch = jnp.ones_like(test.lds_a)
test(switch=switches, mismatch=mismatch, t=5e-9)

mismatch = jnp.array(np.random.normal(size=test.lds_a.shape, loc=1, scale=0.1))

traj = []
times = np.linspace(2e-9, 10e-9, 100)
for t in tqdm(times):
    traj.append(test(switch=switches, mismatch=mismatch, t=t)[0, 0])

eqx.filter_jit(test)
for t in tqdm(times):
    traj.append(test(switch=switches, mismatch=mismatch, t=t)[0, 0])

plt.plot(times, traj[: len(times)])
plt.plot(times, traj[len(times) :])
plt.show()
