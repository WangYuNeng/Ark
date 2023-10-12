from copy import deepcopy
from functools import partial
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sensitivity_ananlysis import create_single_line
from spec import mm_tln_spec
from tqdm import tqdm

from ark.ark import Ark
from ark.cdg.cdg import CDG, CDGElement
from ark.optimization.sa import SimulatedAnnealing


def bits2int(bits: list[bool], msb_first: bool = True) -> int:
    """Convert a base-2 representation in bit array to an integer

    Args:
        bits (list[bool]): base-2 representation, MSB first if msb_first is True
        msb_first (bool, optional): Whether the bits start with MSB. Defaults to True.

    Returns:
        int: The decimal integer value
    """
    n_bits = len(bits)
    if not msb_first:
        bit_enum = enumerate(bits[::-1])
    else:
        bit_enum = enumerate(bits)
    return sum([int(bit) * 2 ** (n_bits - i - 1) for i, bit in bit_enum])


def int2bits(val: int, n_bits: int, msb_first: bool = True) -> npt.NDArray[np.bool_]:
    """Convert an integer to base-2 representation

    Args:
        val (int): Decimal integer value
        n_bits (int): # of bits
        msb_first (bool, optional): whether the bit array starts with MSB.
        Defaults to True.

    Returns:
        npt.NDArray[np.bool_]: value in base-2 representation, MSB first if msb_first
        is True, LSB first otherwise
    """
    bits = []
    base = 2 ** (n_bits - 1)
    for i in range(n_bits):
        bits.append(int(val >= base))
        if val >= base:
            val -= base
        base //= 2
    if not msb_first:
        bits.reverse()
    return np.array(bits, dtype=np.bool_)


def single_bit_flip_test(
    n_chl_bit: int, crps: Mapping[int, npt.NDArray[np.bool_]], center_chls: list[int]
) -> list[list[float]]:
    """Perform 1-bit flipping test from Uli's paper

    Args:
        n_chl_bit (int): Number of challenge bits
        crps (Mapping[int, npt.NDArray[np.bool_]]): Mapping from challenges to
        response(s). The response can be multiple bits.
        center_chls (list[int]): Center challenge values
    """

    n_rsp_bit = len(crps[center_chls[0]])
    flipped_cnt = np.zeros(shape=(n_chl_bit, n_rsp_bit))
    for chl in center_chls:
        chl_bits = int2bits(chl, n_chl_bit)
        rsp = crps[chl]
        for i, _ in enumerate(chl_bits):
            chl_bits[i] ^= 1
            flipped_chl = bits2int(chl_bits)
            rsp_flipped = crps[flipped_chl]
            flipped_cnt[i] += rsp != rsp_flipped
            chl_bits[i] ^= 1
    flipped_prob = flipped_cnt / len(center_chls)
    return flipped_prob


def evaluate_puf(
    puf: CDG, system: Ark, time_eval: list[float], n_eval: int, plot: bool = False
):
    rsps = []
    for _ in range(n_eval):
        cdg_execution_data = puf.execution_data()
        node2trace = system.execute(
            cdg_execution_data=cdg_execution_data,
            time_eval=time_eval,
            store_inplace=False,
        )
        rsp = np.sum(node2trace["IdealV_0"][0])
        if plot:
            plt.plot(time_eval, node2trace["IdealV_0"][0])
        rsps.append(rsp)
    plt.show()
    return np.average(rsps)


def perturb_attr(puf: CDG):
    new_puf = deepcopy(puf)
    elements = new_puf.edges
    mod_ele: CDGElement = np.random.choice(elements)
    attr_name = np.random.choice(list(mod_ele.attr_def.keys()))
    new_val = 0.5 + np.random.random() * 1.5
    mod_ele.attrs[attr_name] = new_val
    return new_puf


def bit_flip(bit_vec: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Randomly flip a bit in the vector

    Args:
        bit_vec (npt.NDArray[np.bool_]): Bit vector

    Returns:
        npt.NDArray[np.bool_]: bit vector with one bit flipped
    """
    flip_pos = np.random.randint(len(bit_vec))
    vec_cpy = bit_vec.copy()
    vec_cpy[flip_pos] ^= 1
    return vec_cpy


# np.random.seed(10)
VEC_LEN, N_VEC = 1000, 100
rand_vecs = np.random.randint(2, size=(N_VEC, VEC_LEN))


def random_xor(bit_vec: npt.NDArray[np.bool_]) -> int:
    return (
        np.sum(bit_vec ^ rand_vecs)
        + np.sum(bit_vec ^ np.roll(bit_vec, 1))
        + np.sum(bit_vec ^ np.roll(bit_vec, 3))
    )


def random_xor_optimum() -> int:
    best_vec = np.zeros(VEC_LEN, dtype=np.bool_)
    for i in range(VEC_LEN):
        if np.sum(rand_vecs[:, i]) > N_VEC / 2:
            best_vec[i] = 1
    return random_xor(best_vec)


if __name__ == "__main__":
    optimizer = SimulatedAnnealing(
        temperature=100,
        frozen_temp=1,
        temp_decay=0.9,
        inner_iteraion=100,
    )
    tline, source, v_nodes, i_nodes, edges = create_single_line(4, CDG())
    system = Ark(cdg_spec=mm_tln_spec)
    system.compile(tline)
    time_range = [0, 2.5e-8]
    time_points = np.linspace(*time_range, 101, endpoint=True)
    eval_func = partial(evaluate_puf, system=system, time_eval=time_points, n_eval=2)
    init_sol = tline
    sol = optimizer.optimize(
        init_sol=init_sol,
        neighbor_func=perturb_attr,
        cost_func=evaluate_puf,
    )
    optimizer.visualize_log()

    print(evaluate_puf(tline, system, time_points, 10, plot=True))
    print(evaluate_puf(sol, system, time_points, 10, plot=True))

    best_cost, best_line = eval_func(tline), tline
    for _ in tqdm(range(100 * 21)):
        perturbed = perturb_attr(best_line)
        cost = eval_func(perturbed)
        if cost < best_cost:
            best_cost = cost
            best_line = perturbed
    print(evaluate_puf(sol, system, time_points, 10, plot=True))

    # init_sol = np.random.randint(2, size=VEC_LEN)
    # # print(init_sol)
    # print(random_xor(init_sol))
    # sol = optimizer.optimize(
    #     init_sol=init_sol,
    #     neighbor_func=bit_flip,
    #     cost_func=random_xor,
    # )
    # optimizer.visualize_log()
    # # print(sol)
    # print(random_xor(sol))
    # print(random_xor_optimum())
