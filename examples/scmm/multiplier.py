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

    in_val = np.array([1, 0, -1, 2, -1, -2])
    weight = np.array([2, 3, 1, 4, 2, 1]) * C1

    vin = weight_to_sequential(in_val, 1 / FREQ)
    c1 = weight_to_sequential(weight, 1 / FREQ)

    TIME_RANGE = [0, 6 / FREQ]

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

    ideal_cumsum_fn = partial(
        sequential_array, arr=np.cumsum(in_val * weight / C1), period=1 / FREQ
    )
    ideal_cumsum = [ideal_cumsum_fn(t) for t in time_points]
    scmm_cumsum = csar.get_trace(n=0) * C_RATIO
    plt.plot(time_points, ideal_cumsum, label="Ideal")
    plt.plot(time_points, scmm_cumsum, label="SCMM")
    plt.legend()
    plt.show()
