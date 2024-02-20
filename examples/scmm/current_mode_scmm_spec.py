"""
Example: Switch-capacitor Convolution with current mode integration
https://ieeexplore.ieee.org/document/9250500
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
from ark.specification.rule_keyword import DST, EDGE, SRC, TIME
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

# Pixel input
PwmPixel = NodeType(
    name="PwmPixel",
    attrs={
        "order": 0,
        "attr_def": {
            "pwm": AttrDef(attr_type=FunctionType, nargs=2),
        },
    },
)

# Kernel
KernelEdge = EdgeType(
    name="KenerlEdge",
    attrs={
        "attr_def": {
            "weight": AttrDef(attr_type=float, attr_range=Range(min=0)),
        },
    },
)

# Production rules
pixel_kernel_cap = ProdRule(
    KernelEdge,
    PwmPixel,
    Cap,
    DST,
    SRC.pwm(TIME) * EDGE.weight / DST.c,
)
prod_rules = [pixel_kernel_cap]
cdg_types = [Cap, PwmPixel, KernelEdge]
current_scmm_spec = CDGSpec(
    cdg_types=cdg_types, production_rules=prod_rules, validation_rules=None
)


def pwm(t, val):
    """Pulse width modulation function"""
    return t <= val


if __name__ == "__main__":
    system = Ark(cdg_spec=current_scmm_spec)

    pxl0 = PwmPixel(pwm=partial(pwm, val=0.5))
    pxl1 = PwmPixel(pwm=partial(pwm, val=0.75))

    c0 = Cap(c=1.0)

    k0 = KernelEdge(weight=4.0)
    k1 = KernelEdge(weight=1.0)

    scmm = CDG()
    scmm.connect(k0, pxl0, c0)
    scmm.connect(k1, pxl1, c0)

    TIME_RANGE = [0, 1]

    system.compile(cdg=scmm)
    scmm.initialize_all_states(val=0)

    system.print_prog()

    time_points = np.linspace(*TIME_RANGE, 1000)
    system.execute(
        cdg=scmm,
        time_eval=time_points,
        max_step=1e-3,
    )

    plt.plot(time_points, c0.get_trace(n=0), label="V_C")
    plt.plot(time_points, pwm(time_points, 0.5), label="pwm0")
    plt.plot(time_points, pwm(time_points, 0.75), label="pwm1")
    plt.axvline(x=0.5, color="r", linestyle="--")
    plt.axvline(x=0.75, color="r", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.show()
