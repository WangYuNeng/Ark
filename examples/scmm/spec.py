"""
Example: Switch-capacitor matrix multiplier (scmm)
https://ieeexplore.ieee.org/abstract/document/7579580
"""

from functools import partial
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np

from ark.ark import Ark
from ark.cdg.cdg import CDG
from ark.reduction import SUM
from ark.specification.attribute_def import AttrDef
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.range import Range
from ark.specification.rule_keyword import DST, EDGE, SRC, TIME, VAR
from ark.specification.specification import CDGSpec

# Capacitors
Cap = NodeType(
    name="Cap",
    attrs={
        "order": 1,
        "reduction": SUM,
        "attr_def": {
            "c": AttrDef(attr_type=float, attr_range=Range(min=0)),
        },
    },
)
# Voltage input with series resistor
Inp = NodeType(
    name="InpV",
    attrs={
        "order": 0,
        "attr_def": {
            "vin": AttrDef(attr_type=FunctionType, nargs=1),
            "r": AttrDef(attr_type=float, attr_range=Range(min=0)),
        },
    },
)

# Switches
Sw = EdgeType(
    name="Sw",
    attrs={
        "attr_def": {
            "ctrl": AttrDef(attr_type=FunctionType, nargs=1),
            "Gon": AttrDef(attr_type=float, attr_range=Range(min=0)),
            "Goff": AttrDef(attr_type=float, attr_range=Range(min=0)),
        },
    },
)

CapWeight = NodeType(
    name="CapWeight",
    bases=Cap,
    attrs={
        "attr_def": {
            "c": AttrDef(attr_type=FunctionType, nargs=1),
            # Function type to support sequential scheduling of the weights
            "Vm": AttrDef(attr_type=float),  # Common mode bias voltage
            "cbase": AttrDef(
                attr_type=float, attr_range=Range(min=0)
            ),  # Base capacitance when the weight is 0
            # Model the weight caps separately for mismatch analysis, fixed to 4 bits
            # TODO: Add Ark support for list/array type
            "c0": AttrDef(attr_type=float, attr_range=Range(min=0)),  # LSB cap
            "c1": AttrDef(attr_type=float, attr_range=Range(min=0)),
            "c2": AttrDef(attr_type=float, attr_range=Range(min=0)),
            "c3": AttrDef(attr_type=float, attr_range=Range(min=0)),  # MSB cap
        },
    },
)

CapSAR = NodeType(
    name="CapSAR",
    bases=Cap,
)

# Production rules
inp_cweight_conn = ProdRule(
    Sw,
    Inp,
    CapWeight,
    DST,
    (SRC.vin(TIME) - VAR(DST) - DST.Vm)  # -SRC.vin so that the sign is correct
    / DST.c(TIME, DST.cbase, DST.c0, DST.c1, DST.c2, DST.c3)
    * (
        EDGE.ctrl(TIME) * EDGE.Gon / (EDGE.Gon * SRC.r + 1)
        + (-EDGE.ctrl(TIME) + 1) * EDGE.Goff / (EDGE.Goff * SRC.r + 1)
        # Require non-zero ON resistance for finite Gon
    ),
)
cweight_sar_conn = ProdRule(
    Sw,
    CapWeight,
    CapSAR,
    SRC,
    (SRC.Vm - VAR(SRC) - VAR(DST))
    / SRC.c(TIME, SRC.cbase, SRC.c0, SRC.c1, SRC.c2, SRC.c3)
    * (EDGE.ctrl(TIME) * EDGE.Gon + (-EDGE.ctrl(TIME) + 1) * EDGE.Goff),
)

sar_cweight_conn = ProdRule(
    Sw,
    CapWeight,
    CapSAR,
    DST,
    (SRC.Vm - VAR(SRC) - VAR(DST))
    / DST.c
    * (EDGE.ctrl(TIME) * EDGE.Gon + (-EDGE.ctrl(TIME) + 1) * EDGE.Goff),
)
prod_rules = [inp_cweight_conn, cweight_sar_conn, sar_cweight_conn]
cdg_types = [Cap, Inp, Sw, CapWeight, CapSAR]
scmm_spec = CDGSpec(
    cdg_types=cdg_types, production_rules=prod_rules, validation_rules=None
)


def clk(t, period, duty_cycle, offset):
    """Control clock function"""
    t = (t - offset) % period
    return float(t <= duty_cycle * period)


def sequential_array(t, period, arr):
    """Return the value at a given clock cycle

    if value out of range, return the last value

    Args:
        t (float): time
        period (float): period of the clock
        val_array (list): list of values at each clock cycle

    Returns:
        float: value at the t // period clock cycle
    """
    idx = int(t / period)
    if idx >= len(arr):
        # TODO: Need to come up with a different way to handle
        # out of bound query for future jax compatibility.
        return arr[-1]
    return arr[idx]


def sequential_cap(time, *cap_vals, bit_arr):
    """Return the cap value at a given clock cycle

    cap_val[t] = cap_vals[0] + [cap_vals[i] * bit_arr[t][i - 1]
      for i in range(1, len(cap_vals))]
    """
    bits = bit_arr(time)
    cap_val = cap_vals[0]
    for i, bit in enumerate(bits):
        cap_val += cap_vals[i + 1] * bit
    return cap_val


def constant(t, val):
    return val


if __name__ == "__main__":
    raise NotImplementedError(
        "input and weight for bit version not implemented here! \
            Please refer to multipler.py for the complete example."
    )
    system = Ark(cdg_spec=scmm_spec)

    # N-path filter implementation
    C_RATIO = 39
    C1 = 300e-10
    C2 = C1 * C_RATIO
    G_ON = 1e6
    G_OFF = 0.0
    R_IN = 1e-6
    FREQ = 1e9
    VDD = partial(constant, val=-1.0)

    c1_fixed = partial(constant, val=C1)

    TIME_RANGE = [0, 20 / FREQ]

    phi1 = partial(clk, period=1 / FREQ, duty_cycle=0.5, offset=0)
    phi2 = partial(clk, period=1 / FREQ, duty_cycle=0.5, offset=0.5 / FREQ)

    scmm = CDG()
    inp = Inp(vin=VDD, r=R_IN)
    cweight = CapWeight(c=c1_fixed, Vm=0.0)
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
    ax[0].plot(time_points, [VDD(t) for t in time_points], label="Input")
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
