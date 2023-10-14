import functools as ft
import pickle

import matplotlib.pyplot as plt
import numpy as np
from puf import SwitchableStarPUF, single_bit_flip_test, single_bit_flipped_neighbors
from spec import mm_tln_spec

from ark.optimization.sa import SimulatedAnnealing


def evaluate_puf_single_bit_flip(
    params: list[dict | list[dict]],
    puf: SwitchableStarPUF,
    center_chls_size: int,
    n_inst: int,
    plot: bool = False,
):
    n_bits = puf.n_chl_bits
    cost = 0
    puf.set_circuit_param(*params)
    puf.sample_instances(n_inst=n_inst)
    for inst_id in range(n_inst):
        center_chls = np.random.choice(
            2**n_bits, size=center_chls_size, replace=False
        ).tolist()
        neighbors = [single_bit_flipped_neighbors(chl, n_bits) for chl in center_chls]
        evaluate_chls = list(set(center_chls + sum(neighbors, [])))
        rsps_enumerate = [
            puf.evaluate_instance(inst_id=0, challenge=i) for i in evaluate_chls
        ]
        crps = {
            chl: np.array(time_series_out)
            for chl, time_series_out in zip(evaluate_chls, rsps_enumerate)
        }
        flipped_prob = single_bit_flip_test(
            n_chl_bit=n_bits,
            crps=crps,
            center_chls=center_chls,
        )

        cost += np.sum(np.abs(flipped_prob - 0.5))

        if plot:
            for bit_pos, prob in enumerate(flipped_prob):
                plt.plot(time_points, prob, label=f"Bit {bit_pos}")
            plt.show()
    cost /= n_inst
    return cost


def perturb_attr(params: list[dict | list[dict]], puf: SwitchableStarPUF):
    (
        middle_cap_param,
        middle_edge_param,
        branch_v_param,
        branch_i_param,
        branch_e_param,
    ) = params
    # Uniform randomly choose one parameter to perturb
    param: dict = np.random.choice(
        [
            middle_cap_param,
            middle_edge_param,
            *branch_v_param,
            *branch_i_param,
            *branch_e_param,
        ]
    )
    for key in param.keys():
        if key == "c" or key == "l":
            param[key] = np.random.uniform(0.1e-9, 10e-9)
        elif key == "r" or key == "g":
            param[key] = np.random.uniform(0.0, 2.0)
        elif key == "ws" or key == "wt":
            param[key] = np.random.uniform(0.5, 2.0)
    puf.set_circuit_param(
        middle_cap_param,
        middle_edge_param,
        branch_v_param,
        branch_i_param,
        branch_e_param,
    )
    return params


def save_params(params: list[dict | list[dict]], cost: float) -> None:
    pickle.dump(params, open(f"params_{cost}.pkl", "wb"))


if __name__ == "__main__":
    N_BITS, LINE_LEN, N_INST = 12, 4, 1
    CENTER_CHL_SIZE = 10
    optimizer = SimulatedAnnealing(
        temperature=10,
        frozen_temp=5,
        temp_decay=0.9,
        inner_iteraion=5,
    )
    np.random.seed(428)
    n_bits = N_BITS
    time_range = [0, 5e-8]
    time_points = np.linspace(*time_range, 1001, endpoint=True)
    ss_puf = SwitchableStarPUF(
        n_chl_bits=n_bits,
        n_rsp_bits=1,
        line_len=LINE_LEN,
        spec=mm_tln_spec,
        time_points=time_points,
    )
    vnode_param = {"c": 1e-9, "g": 0.0}
    inode_param = {"l": 1e-9, "r": 0.0}
    et_param = {"ws": 1.0, "wt": 1.0}
    middle_cap_param = vnode_param.copy()
    middle_edge_param = et_param.copy()
    branch_v_param = [vnode_param.copy() for _ in range(ss_puf.branch_n_nodes)]
    branch_i_param = [inode_param.copy() for _ in range(ss_puf.branch_n_nodes)]
    branch_e_param = [et_param.copy() for _ in range(ss_puf.branch_n_edges)]
    ss_puf.set_circuit_param(
        middle_cap_param,
        middle_edge_param,
        branch_v_param,
        branch_i_param,
        branch_e_param,
    )
    ss_puf.sample_instances(n_inst=N_INST)

    init_sol = (
        middle_cap_param,
        middle_edge_param,
        branch_v_param,
        branch_i_param,
        branch_e_param,
    )

    eval_fn = ft.partial(
        evaluate_puf_single_bit_flip,
        puf=ss_puf,
        center_chls_size=CENTER_CHL_SIZE,
        n_inst=N_INST,
    )
    neighbor_fn = ft.partial(perturb_attr, puf=ss_puf)

    final_params = optimizer.optimize(
        init_sol=init_sol,
        neighbor_func=neighbor_fn,
        cost_func=eval_fn,
        logging=True,
        use_wandb=False,
        check_point_func=save_params,
    )
