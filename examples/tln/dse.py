import functools as ft
import json
import os
import pickle
import time
from argparse import ArgumentParser
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from multiprocess import Pool
from puf import (
    PUF,
    SwitchableStarPUF,
    sample_challenges,
    single_bit_flip_test,
    single_bit_flipped_neighbors,
)
from spec import mm_tln_spec
from tqdm import tqdm

from ark.optimization.optimizer import BaseOptimizer
from ark.optimization.sa import SimulatedAnnealing


def sim_and_calc_flip_prob(
    puf: SwitchableStarPUF,
    inst_id: int,
    center_chls: list[int],
    return_crps: bool = False,
) -> list[list[float]] | dict[int, np.ndarray]:
    """Simulate a puf instance and calculate the single-bit flipping probability test restult.

    Args:
        puf (SwitchableStarPUF): The PUF description.
        inst_id (int): The instance id to simulate.
        center_chls (list[int]): challenges to be flipped in the test.
        return_crps (bool, optional): Return the challenge-response pairs instead of
        the flipping probability. Defaults to False.

    Returns:
        list[list[float]]: (n_chl_bit, n_rsp_bit) Flipping probability of each
        response bit for each challenge position.
    """
    neighbors = [
        single_bit_flipped_neighbors(chl, puf.n_chl_bits) for chl in center_chls
    ]
    evaluate_chls = list(set(center_chls + sum(neighbors, [])))
    rsps_enumerate = [
        puf.evaluate_instance(inst_id=inst_id, challenge=i) for i in evaluate_chls
    ]
    crps = {
        chl: np.array(time_series_out)
        for chl, time_series_out in zip(evaluate_chls, rsps_enumerate)
    }
    flipped_prob = single_bit_flip_test(
        n_chl_bit=puf.n_chl_bits,
        crps=crps,
        center_chls=center_chls,
    )
    if return_crps:
        return crps
    return flipped_prob


def evaluate_puf_single_bit_flip(
    params: list[dict | list[dict]],
    puf: SwitchableStarPUF,
    center_chls_size: int,
    window_size: int,
    n_inst: int,
    n_core: int = 1,
    plot: bool = False,
    return_crps: bool = False,
    tqdm_process: bool = False,
) -> float:
    """Evaluate the cost of a PUF by single-bit flipping probability test.

    Args:
        params (list[dict  |  list[dict]]): nominal parameters of the PUF.
        puf (SwitchableStarPUF): the PUF description.
        center_chls_size (int): number of challenges to be flipped in the test.
        n_inst (int): number of instances to simulate.
        n_core (int, optional): number of core to use. Defaults to 1.
        plot (bool, optional): plot the test results. Defaults to False.
        return_crps: (bool, optional): return the challenge-response pairs
        instead of the aggregated cost. Defaults to False.
        tqdm_process (bool, optional): use tqdm to monitor the simulation progress.

    Returns:
        float: cost
    """
    n_bits = puf.n_chl_bits
    puf.set_circuit_param(*params)
    puf.sample_instances(n_inst=n_inst)
    center_chlss = [
        sample_challenges(n_bits=n_bits, n_chl=center_chls_size) for _ in range(n_inst)
    ]
    if n_core == 1:
        flipped_probs = []
        if tqdm_process:
            iterable = tqdm(center_chlss)
        else:
            iterable = center_chlss
        for inst_id, center_chls in enumerate(iterable):
            flipped_probs.append(
                sim_and_calc_flip_prob(puf, inst_id, center_chls, return_crps)
            )
    else:
        with Pool(n_core) as pool:
            flipped_probs = pool.starmap(
                sim_and_calc_flip_prob,
                [
                    (puf, inst_id, center_chls, return_crps)
                    for inst_id, center_chls in enumerate(center_chlss)
                ],
            )
    if return_crps:  # the "flipped_probs" are actually raw crps
        return flipped_probs
    dist_to_ideal = np.abs(np.array(flipped_probs) - 0.5)
    cost = np.min(
        [
            np.mean(dist_to_ideal[:, :, i : i + window_size])
            for i in range(len(puf.time_points) - window_size)
        ]
    )

    # cost = np.mean(dist_to_ideal)
    if plot:
        plt.title(f"Avg Distance to 0.5 vs Time, Tot Cost={cost:.4f}")
        for bit_pos, prob in enumerate(np.mean(dist_to_ideal, axis=0)):
            plt.plot(puf.time_points, prob, label=f"Bit {bit_pos}")
    return cost


def rand_in_range(range: tuple[float, float] = None, bound: tuple[float, float] = None):
    if range is None:
        range = bound
    if range[0] < bound[0]:
        range[0] = bound[0]
    if range[1] > bound[1]:
        range[1] = bound[1]
    return np.random.uniform(*range)


def rand_lc(range: tuple[float, float] = None):
    return rand_in_range(range, (0.1e-9, 10e-9))


def rand_rg(range: tuple[float, float] = None):
    return rand_in_range(range, (0.0, 2.0))


def rand_w(range: tuple[float, float] = None):
    return rand_in_range(range, (0.5, 2.0))


def perturb_attr(
    params: list[dict | list[dict]],
    puf: SwitchableStarPUF,
    perturb_all: bool = False,
    increment: bool = False,
) -> list[dict | list[dict]]:
    """Perturb one parameter of the PUF.

    Args:
        params (list[dict  |  list[dict]]): PUF nominal parameters.
        puf (SwitchableStarPUF): the PUF description.
        perturb_all (bool, optional): perturb all parameters. Defaults to False.
        increment (bool, optional): perturb parameter by increment of
        the current value. Defaults to False.

    Returns:
        list[dict | list[dict]]: perturbed parameters.
    """
    (
        middle_cap_param,
        middle_edge_param,
        branch_v_param,
        branch_i_param,
        branch_e_param,
    ) = params
    # Uniform randomly choose one parameter to perturb
    perturbed_params = [
        middle_cap_param,
        middle_edge_param,
        *branch_v_param,
        *branch_i_param,
        *branch_e_param,
    ]
    if not perturb_all:
        param: dict = np.random.choice(perturbed_params)
        perturbed_params = [param]
    for param in perturbed_params:
        for key, val in param.items():
            perturb_range = None
            if key == "c" or key == "l":
                if increment:
                    perturb_range = [val / 2, val * 2]
                param[key] = rand_lc(range=perturb_range)

            elif key == "r" or key == "g":
                if increment:
                    perturb_range = [val - 0.2, val + 0.2]
                param[key] = rand_rg()

            elif key == "ws" or key == "wt":
                if increment:
                    perturb_range = [val / 2, val * 2]
                param[key] = rand_w()
    puf.set_circuit_param(
        middle_cap_param,
        middle_edge_param,
        branch_v_param,
        branch_i_param,
        branch_e_param,
    )
    return params


def save_params(
    params: list[dict | list[dict]],
    cost: float,
    eval_fn_more_samples: Callable,
    save_dir: str,
) -> None:
    cost_more_sampe = eval_fn_more_samples(params)
    save_prefix = os.path.join(save_dir, f"params_{cost:.3f}_{cost_more_sampe:.3f}")
    plt.savefig(f"{save_prefix}.png")
    plt.cla()
    pickle.dump(params, open(f"{save_prefix}.pkl", "wb"))


def setup_puf(config: dict) -> tuple[PUF, dict]:
    """Setup the PUF according to the config.

    Args:
        config (dict): PUF config.

    Returns:
        SwitchableStarPUF: the PUF description.
    """
    np.random.seed(config["seed"])
    n_bits = config["n_bits"]
    line_len = config["line_len"]
    n_inst = config["n_inst"]
    n_time_point = config["n_time_point"]
    rand_init = config["rand_init"]
    if "checkpoint" in config:
        checkpoint = pickle.load(open(config["checkpoint"], "rb"))
    else:
        checkpoint = None
    time_end = config["time_end"]

    time_range = [0, time_end]
    time_points = np.linspace(*time_range, n_time_point, endpoint=True)
    ss_puf = SwitchableStarPUF(
        n_chl_bits=n_bits,
        n_rsp_bits=1,  # dummy, currently not used
        line_len=line_len,
        spec=mm_tln_spec,
        time_points=time_points,
    )
    if checkpoint:
        (
            middle_cap_param,
            middle_edge_param,
            branch_v_param,
            branch_i_param,
            branch_e_param,
        ) = checkpoint
    elif rand_init:
        middle_cap_param = {
            "c": rand_lc(),
            "g": rand_rg(),
        }
        middle_edge_param = {
            "ws": rand_w(),
            "wt": rand_w(),
        }
        branch_v_param = [
            {
                "c": rand_lc(),
                "g": rand_rg(),
            }
            for _ in range(ss_puf.branch_n_nodes)
        ]
        branch_i_param = [
            {
                "l": rand_lc(),
                "r": rand_rg(),
            }
            for _ in range(ss_puf.branch_n_nodes)
        ]
        branch_e_param = [
            {
                "ws": rand_w(),
                "wt": rand_w(),
            }
            for _ in range(ss_puf.branch_n_edges)
        ]
    else:
        vnode_param = {"c": 1e-9, "g": 0.0}
        inode_param = {"l": 1e-9, "r": 0.0}
        et_param = {"ws": 1.0, "wt": 1.0}
        middle_cap_param = vnode_param.copy()
        middle_edge_param = et_param.copy()
        branch_v_param = [vnode_param.copy() for _ in range(ss_puf.branch_n_nodes)]
        branch_i_param = [inode_param.copy() for _ in range(ss_puf.branch_n_nodes)]
        branch_e_param = [et_param.copy() for _ in range(ss_puf.branch_n_edges)]

    init_sol = (
        middle_cap_param,
        middle_edge_param,
        branch_v_param,
        branch_i_param,
        branch_e_param,
    )
    ss_puf.set_circuit_param(*init_sol)
    ss_puf.sample_instances(n_inst=n_inst)
    return ss_puf, init_sol


def setup_optimization(config: dict, puf: PUF) -> BaseOptimizer:
    """Setup the optimizer according to the config.

    Args:
        config (dict): optimizer config.
        puf (PUF): the PUF description.
    Returns:
        BaseOptimizer: the optimizer.
    """
    # PUD parameters
    n_inst = config["n_inst"]
    center_chls_size = config["center_chls_size"]

    # Optimizer parameters
    temperature = config["temperature"]
    frozen_temp = config["frozen_temp"]
    temp_decay = config["temp_decay"]
    inner_iteration = config["inner_iteration"]
    checkpoint_dir = f"_run_{time.strftime('%Y%m%d-%H%M%S')}_{time.time():.0f}"
    config["checkpoint_dir"] = checkpoint_dir
    os.mkdir(checkpoint_dir)
    json.dump(config, open(f"{checkpoint_dir}/config.json", "w"), indent=4)
    print(f"Checkpoints store at {checkpoint_dir}")

    # Evaluation function parameters
    window_size = config["window_size"]
    n_core = config["n_core"]
    checkpoint_n_inst = config["checkpoint_n_inst"]
    checkpoint_center_chls_size = config["checkpoint_center_chls_size"]
    perturb_all = config["perturb_all"]
    increment = config["increment"]

    optimizer = SimulatedAnnealing(
        temperature=temperature,
        frozen_temp=frozen_temp,
        temp_decay=temp_decay,
        inner_iteraion=inner_iteration,
    )
    eval_fn = ft.partial(
        evaluate_puf_single_bit_flip,
        puf=puf,
        center_chls_size=center_chls_size,
        n_inst=n_inst,
        window_size=window_size,
        n_core=n_core,
    )
    eval_fn_more_sample = ft.partial(
        evaluate_puf_single_bit_flip,
        puf=puf,
        center_chls_size=checkpoint_center_chls_size,
        n_inst=checkpoint_n_inst,
        window_size=window_size,
        plot=True,
        n_core=n_core,
    )
    neighbor_fn = ft.partial(
        perturb_attr, puf=puf, perturb_all=perturb_all, increment=increment
    )
    check_point_fn = ft.partial(
        save_params, eval_fn_more_samples=eval_fn_more_sample, save_dir=checkpoint_dir
    )
    return optimizer, eval_fn, neighbor_fn, check_point_fn


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "-g",
        "--greedy",
        action="store_true",
        help="Use greedy mode of the simulated annealing engine, i.e., only accept a \
            solution when the cost is strictly smaller.",
    )
    parser.add_argument(
        "-w",
        "--wandb",
        action="store_true",
        help="Use weight and bias for real time monitoring.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    use_wandb = args.wandb
    greedy = args.greedy

    ss_puf, init_sol = setup_puf(config)
    optimizer, eval_fn, neighbor_fn, check_point_fn = setup_optimization(config, ss_puf)

    final_params = optimizer.optimize(
        init_sol=init_sol,
        neighbor_func=neighbor_fn,
        cost_func=eval_fn,
        logging=True,
        use_wandb=use_wandb,
        meta_data=config,
        check_point_func=check_point_fn,
        greedy=greedy,
    )
