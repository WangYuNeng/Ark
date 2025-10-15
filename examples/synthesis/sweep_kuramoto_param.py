import pickle

import numpy as np
from obc_syn import synthesize_general, validate_synthesis

if __name__ == "__main__":
    gates = {
        "2-OR": [lambda x, y, z: (x | y) == z, 3, 0],
        "2-AND": [lambda x, y, z: (x & y) == z, 3, 0],
        "2-XOR": [lambda x, y, z: (x ^ y) == z, 3, 1],
        "3-OR": [lambda x, y, z, a: (x | y | z) == a, 4, 1],
        "1-ADDER": [lambda a0, b0, s0, c: ((a0 ^ b0) == s0 and (a0 & b0) == c), 4, 1],
    }

    Kl_values = np.linspace(0, 6, endpoint=True, num=13)
    t_span = (0, 6)
    Kc_values = np.linspace(0.5, 4, endpoint=True, num=8)

    for gate_name, [gate_func, n_io_var, n_aux_osc] in gates.items():
        for exclude_ref_energy in [False, True]:
            model, j_mat = synthesize_general(
                gate_func,
                n_io_var=n_io_var,
                n_aux_osc=n_aux_osc,
                symmetric=True,
                constrain_coupling=0,
                constrain_energy_threshold=False,
                exclude_ref_energy=exclude_ref_energy,
            )
            j_val = np.array(
                [
                    [model[j_mat[i, j]].as_long() for i in range(len(j_mat))]
                    for j in range(len(j_mat))
                ]
            )
            for Kl in Kl_values:
                for Kc in Kc_values:
                    for anneal in [False, True]:
                        np.random.seed(428)
                        hist, (w_ref_energy_data, wo_ref_energy_data) = (
                            validate_synthesis(
                                model,
                                j_mat,
                                gate_func,
                                n_io_var=n_io_var,
                                n_aux_osc=n_aux_osc,
                                Kc=Kc,
                                Kl=Kl,
                                plot=False,
                                anneal=anneal,
                                t_span=t_span,
                            )
                        )

                        file_name = f"sweep_results/{gate_name}_ref{int(exclude_ref_energy)}_Kc{Kc}_Kl{Kl}_anneal{int(anneal)}.pkl"
                        with open(file_name, "wb") as f:
                            pickle.dump(
                                {
                                    "hist": hist,
                                    "w_ref_energy_data": w_ref_energy_data,
                                    "wo_ref_energy_data": wo_ref_energy_data,
                                    "j_val": j_val,
                                },
                                f,
                            )
