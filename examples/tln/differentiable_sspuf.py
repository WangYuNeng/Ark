"""
Model the SSPUF and Bit-flipping test as a differentiable program.
"""


import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_debug_nans", True)


class SwitchableStarPUF(eqx.Module):
    """SwitchableStar PUF in transmission lines.

    - n_branch: # of branches to switch
    - line_len: # of LC section in each branch
    - n_order: # of order to approximate the matrix exponential
    - gm_c: Trainable Gm of integrators associated with capacitors
    - gm_l: Trainable Gm of integrators associated with inductors
    - c_val: Trainable weights for C components
    (c_val[-1] is the middle capacitor)
    - l_val: Trainable weights for L components
    - lc_val_base: The base value of the LC components
    (l, c = lc_val_base * lc_val)
    - lds_a: The A matrix in the state-space representation of the
    SwithcableStar.
    - t: The time point of the output
    """

    n_branch: int
    line_len: int
    n_order: int
    lds_a_shape: tuple[int, int]
    gm_c: jax.Array
    gm_l: jax.Array
    c_val: jax.Array
    l_val: jax.Array
    lc_val_base: int = 1e-9
    pulse_amplitude: float = 1.0
    pulse_t1: float = 0.5e-9
    pulse_t2: float = 1.5e-9
    pulse_t3: float = 2.0e-9

    def __init__(
        self,
        n_branch: int,
        line_len: int,
        n_order: int = 40,
        random: bool = False,
        init_vals: dict = None,
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
        self.lds_a_shape = (1 + n_branch * line_len * 2, 1 + n_branch * line_len * 2)
        self.n_order = n_order

        if init_vals:
            self.gm_c = init_vals["gm_c"]
            self.gm_l = init_vals["gm_l"]
            self.c_val = init_vals["c_val"]
            self.l_val = init_vals["l_val"]
        elif random:
            self.gm_c = np.random.uniform(
                low=0.5,
                high=1.5,
                size=(line_len, 2),
            )
            self.gm_l = np.random.uniform(
                low=0.5,
                high=1.5,
                size=(line_len, 2),
            )
            self.c_val = np.random.uniform(low=0.5, high=1.5, size=(line_len + 1))
            self.l_val = np.random.uniform(low=0.5, high=1.5, size=(line_len))

        else:
            self.gm_c = np.ones(shape=(line_len, 2))
            self.gm_l = np.ones(shape=(line_len, 2))
            self.c_val = np.ones(shape=(line_len + 1))
            self.l_val = np.ones(shape=(line_len))

    def __call__(self, switch: jax.Array, mismatch: jax.Array, t: jax.typing.DTypeLike):
        """Compute the analog PUF response at time t.

        The response is the difference between two stars under two mismatch samples.

        Args:
            switch (jax.Array): An array of {0,1} value denoting whether the branches
            are ON or OFF.
            mismatch (jax.Array): Two arrays of random value to model the mismatch.

        Returns:
            jax.typing.DTypeLike: The output of the SwitchableStar at time t.
        """
        rsp0 = self._calc_one_star_rsp(switch, mismatch[0], t)
        rsp1 = self._calc_one_star_rsp(switch, mismatch[1], t)
        return rsp0 - rsp1

    def _calc_one_star_rsp(
        self, switch: jax.Array, mismatch: jax.Array, t: jax.typing.DTypeLike
    ) -> jax.typing.DTypeLike:
        """Compute the transient response of a SwitchableStar at time t.

        Args:
            switch (jax.Array): An array of {0,1} value denoting whether the branches
            are ON or OFF.
            mismatch (jax.Array): Array of random value to model the mismatch.

        Returns:
            jax.typing.DTypeLike: The output of the SwitchableStar at time t.
        """
        lds_a = self._calc_lds_a_matrix()
        a_sw_mm = jnp.multiply(lds_a, mismatch)
        for i, sw_val in enumerate(switch):
            a_sw_mm = a_sw_mm.at[0, 1 + i * self.line_len * 2].multiply(sw_val)

        b_mat = jnp.zeros(shape=(lds_a.shape[0], 1))
        b_mat = b_mat.at[0, 0].set(1 / self.lc_val_base * self.c_val[-1])
        c_mat = jnp.zeros(shape=(1, lds_a.shape[0]))
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

    def _calc_lds_a_matrix(self) -> jax.Array:
        """Calculate the A matrix of the SwitchableStar in state-space repr."""
        sub_mat_len = 2 * self.line_len
        n_branch = self.n_branch

        def ith_diag_sub_mat():
            a_i = jnp.zeros(shape=(sub_mat_len, sub_mat_len))
            for j in range(sub_mat_len - 1):
                if j % 2 == 0:
                    c_idx = j // 2
                    a_i = a_i.at[j, j + 1].set(-self.gm_c[c_idx, 1] / self.l_val[c_idx])
                    a_i = a_i.at[j + 1, j].set(self.gm_c[c_idx, 0] / self.c_val[c_idx])
                else:
                    l_idx = j // 2 + 1
                    a_i = a_i.at[j, j + 1].set(
                        -self.gm_l[l_idx, 1] / self.c_val[l_idx - 1]
                    )
                    a_i = a_i.at[j + 1, j].set(self.gm_l[l_idx, 0] / self.l_val[l_idx])
            return a_i

        ai_row_mats = [ith_diag_sub_mat() for _ in range(n_branch)]
        for i, diag_mat in enumerate(ai_row_mats):
            left_zeros = jnp.zeros(shape=(sub_mat_len, sub_mat_len * i))
            right_zeros = jnp.zeros(
                shape=(sub_mat_len, sub_mat_len * (n_branch - i - 1))
            )
            ai_row_mats[i] = jnp.hstack([left_zeros, diag_mat, right_zeros])

        mid_to_branch_row = jnp.zeros(shape=(1, sub_mat_len * n_branch))
        mid_to_branch_col = jnp.zeros(shape=(sub_mat_len * n_branch, 1))
        for i in range(n_branch):
            mid_to_branch_row = mid_to_branch_row.at[0, i * sub_mat_len].set(
                -self.gm_l[0, 1] / self.c_val[-1]
            )
            mid_to_branch_col = mid_to_branch_col.at[i * sub_mat_len, 0].set(
                self.gm_l[0, 0] / self.l_val[0]
            )

        A_1_to_n = jnp.vstack([mid_to_branch_row] + ai_row_mats)

        A_mat = jnp.hstack(
            [jnp.vstack([jnp.zeros((1, 1)), mid_to_branch_col]), A_1_to_n]
        )

        # Scale by LC values
        LC_mat = jnp.eye(A_mat.shape[0]) / self.lc_val_base

        lds_a = jnp.matmul(LC_mat, A_mat)
        return lds_a

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
