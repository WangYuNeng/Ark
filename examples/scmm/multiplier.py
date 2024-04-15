import argparse
import re
import time
from functools import partial
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np
from spec import clk, ds_scmm_spec, scmm_spec, sequential_array, sequential_cap

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


def sim(system: Ark, scmm: CDG, time_points: np.ndarray):
    system.compile(cdg=scmm)
    system.print_prog()
    system.execute(
        cdg=scmm,
        time_eval=time_points,
        max_step=1 / FREQ / 100,
    )


def sim_ds(system: Ark, scmm: CDG, time_points: np.ndarray, args, fargs):
    compiler = system.compiler
    ode_fn, _, _, _, _ = compiler.compile_odeterm(cdg=scmm, cdg_spec=ds_scmm_spec)
    compiler.print_odeterm()
    trace = [[C2 * V_BIAS, 0]]
    for t in time_points:
        trace.append(ode_fn(t, trace[-1], args, fargs))

    trace = np.array(trace)
    return trace


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
        "--sim_discrete",
        action="store_true",
        help="Simulate with the discrete state space model too.",
    )
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

    N_BITS = 4

    args = parser.parse_args()
    C_RATIO = args.c_ratio * (2**N_BITS - 1)
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

    system = Ark(cdg_spec=ds_scmm_spec)

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

    scmm.initialize_all_states(val=0.0)
    cweight.init_vals = [0]
    csar.init_vals = [V_BIAS]
    time_points = np.linspace(*TIME_RANGE, 1000)

    if args.sim_discrete:

        ds_system = Ark(cdg_spec=ds_scmm_spec)
        # CapSAR_0_c, CapWeight_0_Vm, CapWeight_0_cbase, CapWeight_0_c0, CapWeight_0_c1, CapWeight_0_c2, CapWeight_0_c3, InpV_0_r, Sw_0_Gon, Sw_0_Goff, Sw_1_Gon, Sw_1_Goff = args
        # CapWeight_0_c, InpV_0_vin, Sw_0_ctrl, Sw_1_ctrl = fargs
        ds_args = [
            C2,
            V_BIAS,
            C1 / 100,
            C1,
            C1 * 2,
            C1 * 4,
            C1 * 8,
            R_IN,
            G_ON,
            G_OFF,
            G_ON,
            G_OFF,
        ]
        ds_fargs = [c1_fn, vin, phi1, phi2]
        readout_point = [i / FREQ / 2 - 0.1 / FREQ for i in range(1, 2 * vec_len + 1)]

        t = time.time()
        trace = sim_ds(ds_system, scmm, readout_point, ds_args, ds_fargs)
        print("Discrete simulation time: ", time.time() - t)
        plt.plot(readout_point, trace[:-1, 0] / C2, label="V_C2 (Ark, discrete)")

    system = Ark(cdg_spec=scmm_spec)
    t = time.time()
    sim(system, scmm, time_points)
    print("Continuous simulation time: ", time.time() - t)
    plt.plot(time_points, csar.get_trace(n=0), label="V_C2 (Ark, continuous)")
    plt.legend()

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

        readout_point = [i / FREQ for i in range(1, vec_len + 1)]
        alignged_spectre_sim = align_data(
            mc_time_points[0], np.average(mc_runs, axis=0), readout_point
        )
        aligned_ark_sim = align_data(time_points, csar.get_trace(n=0), readout_point)
        fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(15, 15))

        sim_diff = alignged_spectre_sim - aligned_ark_sim
        ax[0, 0].plot(readout_point, sim_diff, marker="o")
        ax[0, 0].set_xlabel("Time (s)")
        ax[0, 0].set_ylabel("Voltage (V)")
        ax[0, 0].set_title("Voltage difference. (spectre - ark)")

        ax[0, 1].plot(readout_point, sim_diff / aligned_ark_sim * 100, marker="o")
        ax[0, 1].set_xlabel("Time (s)")
        ax[0, 1].set_ylabel("Precentage (%)")
        ax[0, 1].set_title("Percentage difference. (spectre - ark) / ark")

        # Hide ax[0, 2].
        ax[0, 2].axis("off")

        ax[1, 0].scatter(in_val, sim_diff)
        ax[1, 0].axhline(0, color="black", lw=1)
        ax[1, 0].axvline(0, color="black", lw=1)
        ax[1, 0].axvline(V_BIAS, color="red", lw=1, linestyle="--", label="V_BIAS")
        ax[1, 0].set_xlabel("Volt (V)")
        ax[1, 0].set_ylabel("Volt (V)")
        ax[1, 0].set_title("Volt Diff vs Input Volt")
        ax[1, 0].legend()

        ax[1, 1].scatter(weight, sim_diff)
        ax[1, 1].axhline(0, color="black", lw=1)
        ax[1, 1].axvline(0, color="black", lw=1)
        ax[1, 1].set_xlabel("Weight (F)")
        ax[1, 1].set_ylabel("Volt (V))")
        ax[1, 1].set_title("Volt Diff vs Weight")

        ax[1, 2].scatter(aligned_ark_sim, sim_diff)
        ax[1, 2].axvline(V_BIAS, color="red", lw=1, linestyle="--", label="V_BIAS")
        ax[1, 2].set_xlabel("Volt (V)")
        ax[1, 2].set_ylabel("Volt(V)")
        ax[1, 2].set_title("Volt Diff vs Ark Simulation")
        ax[1, 2].legend()

        per_cycle_diff = sim_diff
        per_cycle_diff[1:] -= per_cycle_diff[:-1]
        ax[2, 0].scatter(in_val, per_cycle_diff)
        ax[2, 0].axhline(0, color="black", lw=1)
        ax[2, 0].axvline(0, color="black", lw=1)
        ax[2, 0].axvline(V_BIAS, color="red", lw=1, linestyle="--", label="V_BIAS")
        ax[2, 0].set_xlabel("Volt (V)")
        ax[2, 0].set_ylabel("Volt (V)")
        ax[2, 0].set_title("In-Cycle Diff vs Input")
        ax[2, 0].legend()

        ax[2, 1].scatter(weight, per_cycle_diff)
        ax[2, 1].axhline(0, color="black", lw=1)
        ax[2, 1].axvline(0, color="black", lw=1)
        ax[2, 1].set_xlabel("Weight (F)")
        ax[2, 1].set_ylabel("Volt (V))")
        ax[2, 1].set_title("In-Cycle Diff vs Weight")

        ax[2, 2].scatter(aligned_ark_sim, per_cycle_diff)
        ax[2, 2].axvline(V_BIAS, color="red", lw=1, linestyle="--", label="V_BIAS")
        ax[2, 2].set_xlabel("Volt (V)")
        ax[2, 2].set_ylabel("Volt(V)")
        ax[2, 2].set_title("In-Cycle Diff vs Ark Simulation")
        ax[2, 2].legend()

        plt.tight_layout()
        plt.savefig("ark_mc_compare_diff.png", dpi=300)
        plt.show()
        plt.close()

    if args.plot_analytical:
        plt.figure()
        readout_point = [i / FREQ for i in range(1, vec_len + 1)]
        in_val = (in_val - V_BIAS) * -1
        ideal_cumsum = np.cumsum(in_val * weight / C1) / C_RATIO

        k = C2 / (C2 + np.abs(weight))
        mu = weight / C2
        a_arr = [
            [mu[j] * np.prod(k[j : i + 1]) for j in range(i + 1)] for i in range(len(k))
        ]
        scmm_analytic = np.array(
            [np.sum(a_arr[i] * in_val[: i + 1]) for i in range(len(a_arr))]
        )

        scmm_cumsum = csar.get_trace(n=0) - V_BIAS
        plt.plot(readout_point, ideal_cumsum, label="Ideal MM", marker="^")
        plt.plot(readout_point, scmm_analytic, label="SCMM Analytical", marker="o")
        plt.plot(time_points, scmm_cumsum, label="V_C2")
        plt.legend()
        plt.savefig("ark_analytical_compare.png")
        plt.show()
