import argparse
import re
from functools import partial
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np
from spec import clk, scmm_spec, sequential_array, sequential_cap

from ark.ark import Ark
from ark.cdg.cdg import CDG


def weight_to_sequential(w: list[float], clk_period: float) -> FunctionType:
    """
    Convert a list of weights to a sequential function.
    """
    return partial(sequential_array, arr=w, period=clk_period)


def plot_mc_csv(file_path: str, ax: plt.Axes):
    """
    Plot the Monte Carlo simulation results from a CSV file.

    The CSV file should have the following format:
    ```
    HEADER (Ignored)
    time, mc_run_1, time, mc_run_2, time, mc_run_3, ...
    ```
    Args:
        file_path (str): The path to the CSV file.
        ax (plt.Axes): The axis to plot on.
    """
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    time_points = data[:, 0::2].T - 30e-9
    mc_runs = data[:, 1::2].T
    for time, run in zip(time_points, mc_runs):
        ax.plot(time, run)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Monte Carlo Simulation")
    return ax, time_points, mc_runs


def align_data(
    time_points: np.ndarray, data: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    """
    Align the data to the target time points using linear interpolation.

    Args:
        time_points (np.ndarray): The time points of the data.
        data (np.ndarray): The data to align.
        target_time (np.ndarray): The target time points.
    Returns:
        np.ndarray: The aligned data.
    """
    return np.interp(target_time, time_points, data)


def parse_scs(scs_path: str) -> dict:
    """
    Parse the SCS file and return the parsed data as a dictionary.

    Args:
        scs_path (str): The path to the SCS file.
    Returns:
        dict: The parsed data.
    """
    with open(scs_path, "r") as f:
        lines = f.readlines()

    mode = "WAIT"
    data = {"in_val": [], "weight_bit_pos": [], "weight_val": []}
    for line in lines:
        if mode == "WAIT":
            if line.startswith("_vSCM_IN"):
                mode = "READ_IN"
            elif line.startswith("//digital weight"):
                mode = "READ_WEIGHT"
        elif mode == "READ_IN":
            vals = line.split()
            if len(vals) == 1:
                mode = "WAIT"
            else:
                data["in_val"].append(float(vals[2]))
        elif mode == "READ_WEIGHT":
            # Use regex to find data="..." in the line and parse the value
            data["weight_bit_pos"].append(int(line[7]))
            match = re.search(r'data="[01]+"', line)
            if match:
                data_str = match.group(0).split("=")[1][1:-1]
                data["weight_val"].append([int(bit) for bit in data_str])
            else:
                raise ValueError("Could not find weight bit value in line!")

    # Reorder the weight values based on the bit position
    data["weight_val"] = [
        val for _, val in sorted(zip(data["weight_bit_pos"], data["weight_val"]))
    ]
    return data


def weight_bit_to_val(weight_val: list[list[int]]) -> list[float]:
    """
    Convert a list of weight bit values to a list of weight values.

    The weight_val is in the shape of (n_bits, n_weights).

    Args:
        weight_val (list[list[int]]): The list of weight bit values.
    Returns:
        list[float]: The list of weight values.
    """
    weight_val_bits = np.array(weight_val, dtype=float)
    n_bits, n_weights = weight_val_bits.shape
    weight_val_arr = weight_val_bits * (2 ** np.arange(n_bits)[:, None])
    weight_val_float = weight_val_arr.sum(axis=0)
    return weight_val_bits.T, weight_val_float


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run SCMM example.")
    parser.add_argument("--vec_len", type=int, help="Length of input vector.")
    parser.add_argument("--c_ratio", type=float, help="Capacitor ratio.", required=True)
    parser.add_argument("--c1", type=float, help="Value of C1.", required=True)
    parser.add_argument("--freq", type=float, help="Value of FREQ.", required=True)
    parser.add_argument("--g_on", type=float, help="Value of G_ON.", default=1e6)
    parser.add_argument("--g_off", type=float, help="Value of G_OFF.", default=0.0)
    parser.add_argument("--r_in", type=float, help="Value of R_IN.", default=0.0)
    parser.add_argument("--v_bias", type=float, help="Value of V_BIAS.", default=0.0)
    parser.add_argument(
        "--from_scs", type=str, help="Initialize weight with scs file.", default=None
    )
    parser.add_argument(
        "--plot_csv", type=str, help="Plot mc simulation fomr csv.", default=None
    )
    parser.add_argument(
        "--plot_analytical",
        action="store_true",
        help="Plot analytical solution.",
        default=False,
    )

    args = parser.parse_args()
    C_RATIO = args.c_ratio
    C1 = args.c1
    C2 = C1 * C_RATIO
    G_ON = args.g_on
    G_OFF = args.g_off
    R_IN = args.r_in
    FREQ = args.freq
    V_BIAS = args.v_bias

    assert (
        args.vec_len or args.from_scs
    ), "Either vec_len or from_scs should be provided!"
    if args.from_scs:
        scs_data = parse_scs(args.from_scs)
        in_val = np.array(scs_data["in_val"])
        weight_bit, weight = weight_bit_to_val(scs_data["weight_val"])
        weight_bit, weight = weight_bit, weight
        weight *= C1
        assert len(weight) == len(in_val), "Input and weight length mismatch!"
        vin = weight_to_sequential(in_val, 1 / FREQ)
        c1_bit = weight_to_sequential(weight_bit, 1 / FREQ)
        c1_fn = partial(sequential_cap, bit_arr=c1_bit)
        vec_len = len(in_val)

    else:
        vec_len = args.vec_len
        in_val = np.random.randint(-4, 5, size=vec_len)
        weight = np.random.randint(1, 5, size=vec_len) * C1

        raise NotImplementedError(
            "Random input and weight for bit version not implemented!"
        )
        vin = weight_to_sequential(in_val, 1 / FREQ)
        c1 = weight_to_sequential(weight, 1 / FREQ)

    system = Ark(cdg_spec=scmm_spec)

    Inp = scmm_spec.node_type("InpV")
    CapWeight = scmm_spec.node_type("CapWeight")
    CapSAR = scmm_spec.node_type("CapSAR")
    Sw = scmm_spec.edge_type("Sw")

    TIME_RANGE = [0, vec_len / FREQ]

    phi1 = partial(clk, period=1 / FREQ, duty_cycle=0.5, offset=0)
    phi2 = partial(clk, period=1 / FREQ, duty_cycle=0.5, offset=0.5 / FREQ)

    scmm = CDG()
    inp = Inp(vin=vin, r=R_IN)
    cweight = CapWeight(
        c=c1_fn, Vm=V_BIAS, cbase=C1 / 100, c0=C1, c1=C1 * 2, c2=C1 * 4, c3=C1 * 8
    )
    csar = CapSAR(c=C2)
    sw1 = Sw(ctrl=phi1, Gon=G_ON, Goff=G_OFF)
    sw2 = Sw(ctrl=phi2, Gon=G_ON, Goff=G_OFF)
    scmm.connect(sw1, inp, cweight)
    scmm.connect(sw2, cweight, csar)

    system.compile(cdg=scmm)
    scmm.initialize_all_states(val=0.0)
    cweight.init_vals = [0]
    csar.init_vals = [V_BIAS]

    system.print_prog()

    time_points = np.linspace(*TIME_RANGE, 100)
    system.execute(
        cdg=scmm,
        time_eval=time_points,
        max_step=1 / FREQ / 100,
    )

    fig, ax = plt.subplots(nrows=5)
    ax[0].plot(time_points, [vin(t) for t in time_points], label="Input")
    ax[0].legend()
    for i, (node, phi) in enumerate(zip([cweight, csar], [phi1, phi2])):
        trace = node.get_trace(n=0)
        ax[2 * i + 1].plot(
            time_points, [phi(t) for t in time_points], label="phi %d" % (i + 1)
        )
        ax[2 * i + 1].legend()
        ax[2 * (i + 1)].plot(time_points, trace, label="Cap %d" % (i + 1))
        ax[2 * (i + 1)].legend()
    plt.tight_layout()
    plt.savefig("ark_trasient.png")
    plt.show()
    plt.close()

    if args.plot_csv:
        fig, ax = plt.subplots()
        _, mc_time_points, mc_runs = plot_mc_csv(args.plot_csv, ax)
        ax.plot(time_points, csar.get_trace(n=0), label="V_C2 (Ark, deterministic)")
        plt.legend()
        plt.savefig("ark_mc_compare.png")
        plt.show()
        plt.close()

        alignged_run = align_data(
            mc_time_points[0], np.average(mc_runs, axis=0), time_points
        )
        plt.figure()
        plt.plot(
            time_points,
            (csar.get_trace(n=0) - 0.5) / (alignged_run - 0.5),
        )
        plt.title("ark sim / spectre sim (both traces are subtracted 0.5 first)")
        plt.savefig("ark_mc_compare_ratio.png")
        plt.show()
        plt.close()

    if args.plot_analytical:
        plt.figure()
        readout_point = [i / FREQ for i in range(1, vec_len + 1)]
        in_val = (in_val - V_BIAS) * -1
        # in_val = (in_val) * -1
        # ideal_cumsum = np.cumsum(in_val * weight / C1) / C_RATIO

        k = C2 / (C2 + np.abs(weight))
        mu = weight / C2
        a_arr = [
            [mu[j] * np.prod(k[j : i + 1]) for j in range(i + 1)] for i in range(len(k))
        ]
        scmm_analytic = np.array(
            [np.sum(a_arr[i] * in_val[: i + 1]) for i in range(len(a_arr))]
        )
        # scmm_analytic += V_BIAS * np.array([np.prod(k[: i + 1]) for i in range(len(k))])

        scmm_cumsum = csar.get_trace(n=0) - V_BIAS
        # plt.plot(readout_point, ideal_cumsum, label="Ideal MM", marker="^")
        plt.plot(readout_point, scmm_analytic, label="SCMM Analytical", marker="o")
        plt.plot(time_points, scmm_cumsum, label="V_C2")
        plt.legend()
        plt.savefig("ark_analytical_compare.png")
        plt.show()
