import pickle

import jax
import jax.numpy as jnp
import numpy as np
from obc_syn import random_simulation, synthesize_general

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    gates = {
        "2-OR": [lambda x, y, z: (x | y) == z, 3, 0],
        "2-AND": [lambda x, y, z: (x & y) == z, 3, 0],
        "2-XOR": [lambda x, y, z: (x ^ y) == z, 3, 1],
        "3-OR": [lambda x, y, z, a: (x | y | z) == a, 4, 1],
        "1-ADDER": [lambda a0, b0, s0, c: ((a0 ^ b0) == s0 and (a0 & b0) == c), 4, 1],
    }

    Kl_values = np.linspace(0, 2, 5, endpoint=True)
    Kc = 1
    Kt = 0.01
    Kt_ratio = 100
    t_end_values = [5, 10, 20]
    anneal = True

    for gate_name, [gate_func, n_io_var, n_aux_osc] in gates.items():
        model, j_mat = synthesize_general(
            gate_func,
            n_io_var=n_io_var,
            n_aux_osc=n_aux_osc,
            symmetric=True,
            constrain_coupling=0,
            constrain_energy_threshold=False,
        )
        j_val = np.array(
            [
                [model[j_mat[i, j]].as_long() for i in range(len(j_mat))]
                for j in range(len(j_mat))
            ]
        )
        for Kl in Kl_values:
            for t_end in t_end_values:
                np.random.seed(428)
                n_sim = 2 ** (n_io_var + n_aux_osc + 8)
                traces, ts = random_simulation(
                    J=j_val,
                    Kc=Kc,
                    Kl=Kl,
                    Kt=Kt,
                    Kt_ratio=Kt_ratio,
                    t_end=t_end,
                    n_sim=n_sim,
                    anneal=anneal,
                )
                print(gate_name, Kc, Kl, t_end, traces.shape)
                file_name = (
                    f"sweep_results/noisy/{gate_name}_Kc{Kc}_Kl{Kl}_t_end{t_end}.npz"
                )
                with open(file_name, "wb") as f:
                    jnp.savez(
                        f,
                        traces=traces,
                        J=j_val,
                        Kc=Kc,
                        Kl=Kl,
                        Kt=Kt,
                        ts=ts,
                        Kt_ratio=Kt_ratio,
                    )
