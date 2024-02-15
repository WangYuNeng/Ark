"""
Model the SSPUF and Bit-flipping test as a differentiable program.
"""

from functools import partial
from typing import Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import config
from puf import create_switchable_star_cdg
from spec import mm_tln_spec

from ark.cdg.cdg import CDG
from ark.compiler import ArkCompiler

config.update("jax_debug_nans", True)


class SwitchableStarPUF(eqx.Module):
    """SwitchableStar PUF in transmission lines.

    - n_branch: # of branches to switch
    - line_len: # of LC section in each branch
    - gm_c: Trainable Gm of integrators associated with capacitors
    - gm_l: Trainable Gm of integrators associated with inductors
    - c_val: Trainable weights for C components
    (c_val[-1] is the middle capacitor)
    - l_val: Trainable weights for L components
    - lc_val_base: The base value of the LC components
    (l, c = lc_val_base * lc_val)
    - t: The time point of the output
    - mismatch_len: The length required for the mismatch vector
    """

    n_branch: int
    line_len: int
    gm_c: jax.Array
    gm_l: jax.Array
    c_val: jax.Array
    l_val: jax.Array
    lc_val_base: int = 1e-9
    pulse_amplitude: float = 1.0
    pulse_t1: float = 0.5e-9
    pulse_t2: float = 1.5e-9
    pulse_t3: float = 2.0e-9
    mismatch_len: int
    param_split_idx: list[int]

    def __init__(
        self,
        n_branch: int,
        line_len: int,
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

        self.mismatch_len = (
            len(self.gm_c) * 2
            + len(self.gm_l) * 2
            + len(self.c_val[:-1])
            + len(self.l_val)
        ) * n_branch + 1

        param_len = np.array(
            [
                len(self.gm_c) * 2,
                len(self.gm_l) * 2,
                len(self.c_val) - 1,
                len(self.l_val),
            ]
            * self.n_branch
        )
        self.param_split_idx = np.cumsum(param_len).tolist()

    def __call__(self, switch: jax.Array, mismatch: jax.Array, t: jax.typing.DTypeLike):
        """Compute the analog PUF response at time t.

        The response is the difference between two stars under two mismatch samples.

        Args:
            switch (jax.Array): An array of {0,1} value denoting whether the branches
            are ON or OFF.
            mismatch (jax.Array): Two arrays of random value to model the mismatch.
            t (jax.typing.DTypeLike): The time point of the output.

        Returns:
            jax.typing.DTypeLike: The output of the SwitchableStar at time t.
        """
        raise NotImplementedError

    def _apply_mismatch_single_star(self, mismatch: jax.Array):
        """Apply the mismatch to the components of a single star and return
        the mismatched parameters.

        Args:
            mismatch (jax.Array): The mismatch values.
            The length of the mismatch should be self.mismatch_len.

        Returns:
            tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.typing.DTypeLike]:
            The mismatched components for each branch and the middle capacitor.
            gmc_mm and gml_mm have the shape of (n_branch, line_len, 2).
            c_mm and l_mm have the shape of (n_branch, line_len).
        """
        mismatch_split = jnp.split(mismatch, self.param_split_idx)
        gmc_mm, gml_mm, c_mm, l_mm = [], [], [], []
        for i in range(self.n_branch):
            gmc_mm.append(mismatch_split[i * 4].reshape(self.gm_c.shape) * self.gm_c)
            gml_mm.append(
                mismatch_split[i * 4 + 1].reshape(self.gm_l.shape) * self.gm_l
            )
            c_mm.append(mismatch_split[i * 4 + 2] * self.c_val[:-1])
            l_mm.append(mismatch_split[i * 4 + 3] * self.l_val)

        middle_c_mm = mismatch_split[-1][0] * self.c_val[-1]
        return (
            jnp.array(gmc_mm),
            jnp.array(gml_mm),
            jnp.array(c_mm),
            jnp.array(l_mm),
            middle_c_mm,
        )


class SSPUF_MatrixExp(SwitchableStarPUF):
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

    n_order: int
    lds_a_shape: tuple[int, int]

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
        super().__init__(n_branch, line_len, random, init_vals)
        self.lds_a_shape = (1 + n_branch * line_len * 2, 1 + n_branch * line_len * 2)
        self.n_order = n_order

    def __call__(self, switch: jax.Array, mismatch: jax.Array, t: jax.typing.DTypeLike):
        """Compute the analog PUF response at time t.

        The response is the difference between two stars under two mismatch samples.

        Args:
            switch (jax.Array): An array of {0,1} value denoting whether the branches
            are ON or OFF.
            mismatch (jax.Array): Two arrays of random value to model the mismatch.
            t (jax.typing.DTypeLike): The time point of the output.

        Returns:
            jax.typing.DTypeLike: The output of the SwitchableStar at time t.
        """
        rsp = []
        for mm_arr in mismatch:
            gmc_mm, gml_mm, c_mm, l_mm, middle_c_mm = self._apply_mismatch_single_star(
                mm_arr
            )
            rsp.append(
                self._calc_one_star_rsp(
                    switch, gmc_mm, gml_mm, c_mm, l_mm, middle_c_mm, t
                )
            )
        return rsp[0] - rsp[1]

    def _calc_one_star_rsp(
        self,
        switch: jax.Array,
        gmc_mm: jax.Array,
        gml_mm: jax.Array,
        c_mm: jax.Array,
        l_mm: jax.Array,
        middle_c_mm: jax.typing.DTypeLike,
        t: jax.typing.DTypeLike,
    ) -> jax.typing.DTypeLike:
        """Compute the transient response of a SwitchableStar at time t.

        Args:
            switch (jax.Array): An array of {0,1} value denoting whether the branches
            are ON or OFF.
            mismatch (jax.Array): Array of random value to model the mismatch.

        Returns:
            jax.typing.DTypeLike: The output of the SwitchableStar at time t.
        """
        lds_a = self._calc_lds_a_matrix(gmc_mm, gml_mm, c_mm, l_mm)
        a_sw_mm = lds_a
        # a_sw_mm = jnp.multiply(lds_a, mismatch)
        for i, sw_val in enumerate(switch):
            a_sw_mm = a_sw_mm.at[0, 1 + i * self.line_len * 2].multiply(sw_val)

        b_mat = jnp.zeros(shape=(lds_a.shape[0], 1))
        b_mat = b_mat.at[0, 0].set(1 / self.lc_val_base / middle_c_mm)
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

    def _calc_lds_a_matrix(
        self,
        gmc_mm: jax.Array,
        gml_mm: jax.Array,
        c_mm: jax.Array,
        l_mm: jax.Array,
    ) -> jax.Array:
        """Calculate the A matrix of the SwitchableStar in state-space repr."""
        sub_mat_len = 2 * self.line_len
        n_branch = self.n_branch

        def ith_diag_sub_mat(i: jax.typing.DTypeLike):
            a_i = jnp.zeros(shape=(sub_mat_len, sub_mat_len))
            for j in range(sub_mat_len - 1):
                if j % 2 == 0:
                    c_idx = j // 2
                    a_i = a_i.at[j, j + 1].set(-gmc_mm[i, c_idx, 1] / l_mm[i, c_idx])
                    a_i = a_i.at[j + 1, j].set(gmc_mm[i, c_idx, 0] / c_mm[i, c_idx])
                else:
                    l_idx = j // 2 + 1
                    a_i = a_i.at[j, j + 1].set(
                        -gml_mm[i, l_idx, 1] / c_mm[i, l_idx - 1]
                    )
                    a_i = a_i.at[j + 1, j].set(gml_mm[i, l_idx, 0] / l_mm[i, l_idx])
            return a_i

        ai_row_mats = [ith_diag_sub_mat(i) for i in range(n_branch)]
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
                -gml_mm[i, 0, 1] / c_mm[i, -1]
            )
            mid_to_branch_col = mid_to_branch_col.at[i * sub_mat_len, 0].set(
                gml_mm[i, 0, 0] / l_mm[i, 0]
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


compiler = ArkCompiler()
InpI = mm_tln_spec.node_type("InpI")
MmV = mm_tln_spec.node_type("MmV")
MmI = mm_tln_spec.node_type("MmI")
MmE = mm_tln_spec.edge_type("MmE")
IdealE = mm_tln_spec.edge_type("IdealE")


def pulse_jax(
    t, amplitude=1, delay=0, rise_time=5e-9, fall_time=5e-9, pulse_width=10e-9
):
    """Trapezoidal pulse function that is compatible with JAX"""
    t_offset = t - delay
    return jnp.where(
        t_offset < rise_time,
        amplitude * t_offset / rise_time,
        jnp.where(
            t_offset < pulse_width + rise_time,
            amplitude,
            jnp.where(
                t_offset < pulse_width + rise_time + fall_time,
                amplitude * (1 - (t_offset - pulse_width - rise_time) / fall_time),
                0,
            ),
        ),
    )


class SSPUF_ODE(SwitchableStarPUF):
    """SwitchableStar PUF in transmission lines implemented with ODEs.

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

    - ode_fn: The ODE form of the SwitchableStarPUF
    - middle_c_mapping: The mapping of the middle capacitors in the CDG (2)
    - switch_args_mapping: The mapping of the switch arguments in the CDG (2, n_branch)
    - c_args_mapping: The mapping of the C components in the CDG (line_len, 2, n_branch)
    - l_args_mapping: The mapping of the L components in the CDG (line_len, 2, n_branch)
    - gmc_args_mapping: The mapping of the Gm of C components in the CDG
    (line_len, 2, 2, n_branch)
    - gml_args_mapping: The mapping of the Gm of L components in the CDG
    (line_len, 2, 2, n_branch)
    - n_state: The # of state variables in the ODE
    - n_num_attrs: The # of numerical attributes in the CDG
    - read_out_idx: The index of the middle capacitors in the ODE state variables
    """

    ode_fn: Callable
    middle_c_mapping: np.ndarray
    switch_args_mapping: np.ndarray
    c_args_mapping: np.ndarray
    l_args_mapping: np.ndarray
    gmc_args_mapping: np.ndarray
    gml_args_mapping: np.ndarray
    n_state: int
    n_num_attrs: int
    read_out_idx: tuple[int, int]

    def __init__(
        self,
        n_branch: int,
        line_len: int,
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
        super().__init__(n_branch, line_len, random, init_vals)
        self._create_and_compile_cdg()

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

        args = jnp.zeros(self.n_num_attrs)

        for star_idx in range(2):
            gmc_mm, gml_mm, c_mm, l_mm, middle_c_mm = self._apply_mismatch_single_star(
                mismatch[star_idx]
            )
            # Set the middle capacitance
            args = args.at[self.middle_c_mapping[star_idx]].set(
                self.lc_val_base * middle_c_mm
            )
            args = args.at[self.switch_args_mapping[star_idx, :]].set(switch)
            args = args.at[self.c_args_mapping[star_idx, :, :]].set(
                self.lc_val_base * c_mm
            )
            args = args.at[self.l_args_mapping[star_idx, :, :]].set(
                self.lc_val_base * l_mm
            )
            args = args.at[self.gmc_args_mapping[star_idx, :, :, :]].set(gmc_mm)
            args = args.at[self.gml_args_mapping[star_idx, :, :, :]].set(gml_mm)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ode_fn),
            diffrax.Tsit5(),
            t0=0,
            t1=t,
            dt0=t,
            y0=jnp.zeros(self.n_state),
            saveat=diffrax.SaveAt(ts=[t]),
            args=args,
        )
        sol = solution.ys.squeeze()
        return sol[self.read_out_idx[0]] - sol[self.read_out_idx[1]]

    def _create_and_compile_cdg(self) -> CDG:
        puf, middle_caps, switch_pairs, branch_pairs = create_switchable_star_cdg(
            n_bits=self.n_branch,
            line_len=self.line_len,
            v_nt=MmV,
            i_nt=MmI,
            et=MmE,
            self_et=IdealE,
            inp_nt=InpI,
        )

        (
            ode_fn,
            node_mapping,
            switch_mapping,
            num_attr_mapping,
            fn_attr_mapping,
        ) = compiler.compile_odeterm(cdg=puf, cdg_spec=mm_tln_spec)

        pulse_input = partial(
            pulse_jax,
            amplitude=1,
            delay=0,
            rise_time=self.pulse_t1,
            fall_time=self.pulse_t3 - self.pulse_t2,
            pulse_width=self.pulse_t2 - self.pulse_t1,
        )
        self.ode_fn = partial(ode_fn, fargs=[pulse_input, pulse_input])

        # Create the mapping for the arguments
        # l,c nominal params (line_len) maps to each branch on 2 stars
        # gm nominal params on forward and backward direction (line_len, 2)
        # maps to each branch on 2 stars
        c_args_mapping = np.zeros((2, self.n_branch, self.line_len), dtype=int)
        l_args_mapping = np.zeros((2, self.n_branch, self.line_len), dtype=int)
        gmc_args_mapping = np.zeros((2, self.n_branch, self.line_len, 2), dtype=int)
        gml_args_mapping = np.zeros((2, self.n_branch, self.line_len, 2), dtype=int)

        self.middle_c_mapping = np.array(
            [num_attr_mapping[f"{cap.name}"]["c"] for cap in middle_caps]
        )
        self.switch_args_mapping = np.array(
            [[switch_mapping[sw.name] for sw in switches] for switches in switch_pairs]
        )
        gml_args_mapping[:, :, 0, :] = jnp.array(
            [
                [
                    [
                        num_attr_mapping[f"{sw.name}"]["wt"],
                        num_attr_mapping[f"{sw.name}"]["ws"],
                    ]
                    for sw in switches
                ]
                for switches in switch_pairs
            ]
        )

        for star_idx, star_branches in enumerate(branch_pairs):
            for branch_idx, branch_item in enumerate(star_branches):
                _, vnodes, inodes, edges = branch_item
                c_args_mapping[:star_idx, branch_idx, :] = np.array(
                    [num_attr_mapping[f"{cap.name}"]["c"] for cap in vnodes]
                )
                l_args_mapping[star_idx, branch_idx, :] = np.array(
                    [num_attr_mapping[f"{ind.name}"]["l"] for ind in inodes]
                )
                gmc_args_mapping[star_idx, branch_idx, :, :] = np.array(
                    [
                        [
                            num_attr_mapping[f"{edge.name}"]["wt"],
                            num_attr_mapping[f"{edge.name}"]["ws"],
                        ]
                        for edge in edges[::2]
                    ]
                )
                gml_args_mapping[star_idx, branch_idx, 1:, :] = np.array(
                    [
                        [
                            num_attr_mapping[f"{edge.name}"]["wt"],
                            num_attr_mapping[f"{edge.name}"]["ws"],
                        ]
                        for edge in edges[1::2]
                    ]
                )

        self.c_args_mapping = c_args_mapping
        self.l_args_mapping = l_args_mapping
        self.gmc_args_mapping = gmc_args_mapping
        self.gml_args_mapping = gml_args_mapping
        self.n_state = len(node_mapping)
        self.n_num_attrs = sum([len(vals) for vals in num_attr_mapping.values()]) + len(
            switch_mapping
        )
        self.read_out_idx = [node_mapping[f"{cap.name}"] for cap in middle_caps]

        return


if __name__ == "__main__":
    n_branch = 4
    line_len = 2
    sspuf_neural_ode = SSPUF_ODE(n_branch, line_len)
