from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from spec import clk, sc_filter_spec, sinosoidal

from ark.ark import Ark
from ark.cdg.cdg import CDG

Cap = sc_filter_spec.node_type("Cap")
Res = sc_filter_spec.node_type("Res")
InpV = sc_filter_spec.node_type("InpV")
Wire = sc_filter_spec.edge_type("Wire")
SwE = sc_filter_spec.edge_type("SwE")

system = Ark(cdg_spec=sc_filter_spec)


if __name__ == "__main__":
    # N-path filter implementation
    N_PATH = 4
    CENTER_FREQ = 1e2
    PERIOD = 1 / CENTER_FREQ
    DUTY_CYCLE = 1 / N_PATH
    TIME_RANGE = [0, 50 * PERIOD]
    INP_FREQ = 1e2

    n_path_filter = CDG()
    inp_v = InpV(fn=partial(sinosoidal, freq=INP_FREQ))
    inp_res = Res(r=1.0)
    n_path_filter.connect(Wire(), inp_v, inp_res)

    # Capacitors and switches
    caps = [Cap(c=1e-2) for _ in range(N_PATH)]
    for i in range(N_PATH):
        ctrl_clk = partial(
            clk,
            offset=PERIOD / N_PATH * i,
            period=PERIOD,
            duty_cycle=DUTY_CYCLE,
        )
        n_path_filter.connect(SwE(ctrl=ctrl_clk), inp_res, caps[i])

    system.compile(cdg=n_path_filter)
    n_path_filter.initialize_all_states(val=0)
    time_points = np.linspace(*TIME_RANGE, 1000)
    system.execute(
        cdg=n_path_filter,
        time_eval=time_points,
        max_step=PERIOD / 100,
    )

    # Plot the output, can observe the voltage across the capacitors converge to
    # the value which is the average during the sampling time.
    # The actual filtered output will be choosing the corresponding capacitor voltage in its
    # duty cycle, which is shown in the last column.
    fig, ax = plt.subplots(nrows=N_PATH + 2, figsize=(4, (N_PATH + 2)))
    ax[0].plot(
        time_points, [sinosoidal(t, freq=INP_FREQ) for t in time_points], label="Input"
    )
    ax[0].legend()
    output = np.zeros(len(time_points))
    for i, node in enumerate(caps):
        trace = node.get_trace(n=0)
        ax[i + 1].plot(time_points, trace, label="Cap %d" % i)
        output += trace * np.array(
            [clk(t, PERIOD, DUTY_CYCLE, PERIOD / N_PATH * i) for t in time_points]
        )
        ax[i + 1].legend()

    ax[-1].plot(time_points, output, label="Output")
    ax[-1].legend()
    plt.tight_layout()
    plt.show()
