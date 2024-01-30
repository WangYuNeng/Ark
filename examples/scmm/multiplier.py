import sys
from functools import partial
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np
from spec import clk, scmm_spec, sequential_array

from ark.ark import Ark
from ark.cdg.cdg import CDG


def weight_to_sequential(w: list[float], clk_period: float) -> FunctionType:
    """
    Convert a list of weights to a sequential function.
    """
    return partial(sequential_array, arr=w, period=clk_period)


if __name__ == "__main__":
    system = Ark(cdg_spec=scmm_spec)

    Inp = scmm_spec.node_type("InpV")
    CapWeight = scmm_spec.node_type("CapWeight")
    CapSAR = scmm_spec.node_type("CapSAR")
    Sw = scmm_spec.edge_type("Sw")

    # N-path filter implementation
    C_RATIO = 39
    C1 = 300e-10
    C2 = C1 * C_RATIO
    G_ON = 1e6
    G_OFF = 0.0
    R_IN = 1e-9
    FREQ = 1e9

    vec_len = int(sys.argv[1])

    in_val = np.random.randint(-4, 5, size=vec_len)
    weight = np.random.randint(1, 5, size=vec_len) * C1

    vin = weight_to_sequential(in_val, 1 / FREQ)
    c1 = weight_to_sequential(weight, 1 / FREQ)

    TIME_RANGE = [0, vec_len / FREQ]

    phi1 = partial(clk, period=1 / FREQ, duty_cycle=0.5, offset=0)
    phi2 = partial(clk, period=1 / FREQ, duty_cycle=0.5, offset=0.5 / FREQ)

    scmm = CDG()
    inp = Inp(vin=vin, r=R_IN)
    cweight = CapWeight(c=c1)
    csar = CapSAR(c=C2)
    sw1 = Sw(ctrl=phi1, Gon=G_ON, Goff=G_OFF)
    sw2 = Sw(ctrl=phi2, Gon=G_ON, Goff=G_OFF)
    scmm.connect(sw1, inp, cweight)
    scmm.connect(sw2, cweight, csar)

    system.compile(cdg=scmm)
    scmm.initialize_all_states(val=0)

    system.print_prog()

    time_points = np.linspace(*TIME_RANGE, 1000)
    system.execute(
        cdg=scmm,
        time_eval=time_points,
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
    plt.show()
    plt.clf()

    readout_point = [i / FREQ for i in range(1, vec_len + 1)]
    ideal_cumsum = np.cumsum(in_val * weight / C1) / C_RATIO

    k = C2 / (C2 + np.abs(weight))
    mu = weight / C2
    a_arr = [
        [mu[j] * np.prod(k[j : i + 1]) for j in range(i + 1)] for i in range(len(k))
    ]
    scmm_analytic = [np.sum(a_arr[i] * in_val[: i + 1]) for i in range(len(a_arr))]

    scmm_cumsum = csar.get_trace(n=0)
    plt.plot(readout_point, ideal_cumsum, label="Ideal", marker="^")
    plt.plot(readout_point, scmm_analytic, label="SCMM Analytical", marker="o")
    plt.plot(time_points, scmm_cumsum, label="V_C2")
    plt.legend()
    plt.show()
