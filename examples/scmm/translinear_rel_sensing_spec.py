"""
Example: Relative image sensing through translinear circuits
https://ieeexplore.ieee.org/abstract/document/7061436
"""

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
from ark.specification.rule_keyword import DST, EDGE, SRC
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

# Pixel input (Assume static for simplicity, can extend to func type)
PixelDiode = NodeType(
    name="PixelDiode",
    attrs={
        "order": 0,
        "attr_def": {
            "i_pd": AttrDef(attr_type=float),
        },
    },
)

# Translinear circuit
TransLinearEdge = EdgeType(
    name="TransLinearEdge",
    attrs={
        "attr_def": {
            "i2vgs": AttrDef(attr_type=FunctionType, nargs=1),
            "vgs2i": AttrDef(attr_type=FunctionType, nargs=1),
        },
    },
)

V_GS_TOT = 2

# Production rules
pixel_kernel_cap = ProdRule(
    TransLinearEdge,
    PixelDiode,
    Cap,
    DST,
    EDGE.vgs2i(-EDGE.i2vgs(SRC.i_pd) + V_GS_TOT) / DST.c,
)
prod_rules = [pixel_kernel_cap]
cdg_types = [Cap, PixelDiode, TransLinearEdge]
current_scmm_spec = CDGSpec(
    cdg_types=cdg_types, production_rules=prod_rules, validation_rules=None
)

I_0 = 1.0
N_N = 1.0
U_T = 1.0
V_T0 = 0.5


def vgs2i(
    v_gs: float,
    i_0: float = I_0,
    n_n: float = N_N,
    u_t: float = U_T,
    v_t0: float = V_T0,
) -> float:
    """Mapping from vgs to ids for NMOS in subthreshold region.

    Args:
        v_gs (float): gate-source voltage
        i_0 (float): 2 * n_n * mu_n * c_ox * u_t ^ 2 * w / l_eff.
        n_n (float): nmos subthreshold slop factor.
        u_t (float): TODO: check slides, what is it?
        v_t0 (float): threshold voltage.
    Returns:
        float: i_ds, drain-source current.
    """
    return i_0 * (np.exp((v_gs - v_t0) / (u_t * n_n)))


def i2vgs(
    i_ds: float,
    i_0: float = I_0,
    n_n: float = N_N,
    u_t: float = U_T,
    v_t0: float = V_T0,
) -> float:
    """Inverse mapping of vgs to ids for NMOS in subthreshold region.

    Args:
        i_ds (float): drain-source current.
        i_0 (float): 2 * n_n * mu_n * c_ox * u_t ^ 2 * w / l_eff.
        n_n (float): nmos subthreshold slop factor.
        u_t (float): TODO: check slides, what is it?
        v_t0 (float): threshold voltage.
    Returns:
        float: gate-source voltage v_gs
    """
    return v_t0 + u_t * n_n * np.log(i_ds / i_0)


if __name__ == "__main__":
    system = Ark(cdg_spec=current_scmm_spec)

    pxl0 = PixelDiode(i_pd=1.0)
    pxl1 = PixelDiode(i_pd=2.0)

    c0 = Cap(c=1.0)
    c1 = Cap(c=1.0)

    t0 = TransLinearEdge(i2vgs=i2vgs, vgs2i=vgs2i)
    t1 = TransLinearEdge(i2vgs=i2vgs, vgs2i=vgs2i)

    scmm = CDG()
    scmm.connect(t0, pxl0, c0)
    scmm.connect(t1, pxl1, c1)

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

    plt.plot(time_points, c0.get_trace(n=0), label="V_C0")
    plt.plot(time_points, c1.get_trace(n=0), label="V_C1")
    plt.legend()
    plt.tight_layout()
    plt.show()

    V_COUNTING_THRESHOLD = 0.5
    ratios = []
    deltas = []
    ipds = np.arange(0.01, 1, 0.05)
    for i_pd in ipds:
        pxl1.set_attr_val("i_pd", i_pd)
        system.execute(
            cdg=scmm,
            time_eval=time_points,
            max_step=1e-3,
        )

        # find the first idx when c1 crosses the threshold
        idx0 = np.where(c0.get_trace(n=0) > V_COUNTING_THRESHOLD)[0][0]
        idx1 = np.where(c1.get_trace(n=0) > V_COUNTING_THRESHOLD)[0][0]
        print(idx0, idx1)
        t_delta = time_points[idx0] - time_points[idx1]
        deltas.append(t_delta)
        ratios.append(1 / i_pd)

    plt.plot(deltas, ratios)
    plt.show()
