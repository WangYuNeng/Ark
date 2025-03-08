"""
Example: TLN (Transmission Line Network) Circuit
Use LC ladders to emulate the telegrapher's equation
Provide specification for
- Ideal LC ladder
- LC mismatched ladder
- Gain mismatched ladder
"""

import os

import jax.numpy as jnp
import numpy as np

from ark.reduction import SUM
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.attribute_type import AnalogAttr, FunctionAttr
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.range import Range
from ark.specification.rule_keyword import DST, EDGE, SELF, SRC, TIME, VAR
from ark.specification.specification import CDGSpec
from ark.specification.validation_rule import ValPattern, ValRule

tln_spec = CDGSpec("tln")
mm_tln_spec = CDGSpec("mm-tln", inherit=tln_spec)


# Example input function
def pulse(t, amplitude=1, delay=0, rise_time=5e-9, fall_time=5e-9, pulse_width=10e-9):
    """Trapezoidal pulse function"""
    t_offset = t - delay
    return jnp.where(
        t_offset < rise_time,
        amplitude * t_offset / rise_time,
        jnp.where(
            t_offset < pulse_width + rise_time,
            amplitude,
            jnp.where(
                t_offset < pulse_width + rise_time + fall_time,
                amplitude * (1 - (t_offset - pulse_width - rise_time) / fall_time),
                0,
            ),
        ),
    )


# Read the V-GM_LUT
lut_file = "V-Gm_LUT.csv"
if os.path.exists(lut_file):
    v2gm_lut = np.loadtxt(lut_file, delimiter=",").T
    v2gm_lut[1] /= max(v2gm_lut[1])  # Normalize the gm values
    v2gm_lut = jnp.array(v2gm_lut)
else:
    v2gm_lut = None


# Ideal gm: not a function of input voltage, always modeled as 1 * ws (or wt)
def unity(x):
    return 1


# Lookup table for gm based on input voltage (in a integrator)
def lut_from_data(v):
    return jnp.interp(v, v2gm_lut[0], v2gm_lut[1])


lc_range, gr_range = Range(min=0.1e-9, max=10e-9), Range(min=-1, max=1)
w_range = Range(min=0.5, max=2.0)

#### Type definitions start ####
# Ideal implementation
# Parallel capacitor(c=capacitance) and resistor(g=conductance)
IdealV = NodeType(
    name="IdealV",
    attrs={
        "order": 1,
        "reduction": SUM,
        "attr_def": {
            "c": AttrDef(attr_type=AnalogAttr(lc_range)),
            "g": AttrDef(attr_type=AnalogAttr(gr_range)),
        },
    },
)
# Series inductor(l=inductance) and resistor(r=resistance)
IdealI = NodeType(
    name="IdealI",
    attrs={
        "order": 1,
        "reduction": SUM,
        "attr_def": {
            "l": AttrDef(attr_type=AnalogAttr(lc_range)),
            "r": AttrDef(attr_type=AnalogAttr(gr_range)),
        },
    },
)

IdealE = EdgeType(name="IdealE")
# Voltage source in Thevenin equivalent
InpV = NodeType(
    name="InpV",
    attrs={
        "order": 0,
        "attr_def": {
            "fn": AttrDef(attr_type=FunctionAttr(nargs=1)),
            "r": AttrDef(attr_type=AnalogAttr(gr_range)),
        },
    },
)
# Current source in Thevenin equivalent
InpI = NodeType(
    name="InpI",
    attrs={
        "order": 0,
        "attr_def": {
            "fn": AttrDef(attr_type=FunctionAttr(nargs=1)),
            "g": AttrDef(attr_type=AnalogAttr(gr_range)),
        },
    },
)
# Mismatched implementation (c, l and gm (modeled with edge weights))
MmV = NodeType(
    name="MmV",
    bases=IdealV,
    attrs={
        "attr_def": {"c": AttrDefMismatch(attr_type=AnalogAttr(lc_range), rstd=0.1)}
    },
)
MmI = NodeType(
    name="MmI",
    bases=IdealI,
    attrs={
        "attr_def": {"l": AttrDefMismatch(attr_type=AnalogAttr(lc_range), rstd=0.1)}
    },
)
MmE = EdgeType(
    name="MmE",
    bases=IdealE,
    attrs={
        "attr_def": {
            "ws": AttrDefMismatch(attr_type=AnalogAttr(w_range), rstd=0.1),
            "wt": AttrDefMismatch(attr_type=AnalogAttr(w_range), rstd=0.1),
            "gm_lut": AttrDef(
                attr_type=FunctionAttr(nargs=1)
            ),  # Lookup table for gm based on input voltage (in a integrator)
        }
    },
)
cdg_types = [IdealV, IdealI, IdealE, InpV, InpI]
hw_cdg_types = [MmV, MmI, MmE]
tln_spec.add_cdg_types(cdg_types)
mm_tln_spec.add_cdg_types(hw_cdg_types)
#### Type definitions end ####

#### Production rules start ####
_v2i = ProdRule(IdealE, IdealV, IdealI, SRC, -VAR(DST) / SRC.c)
v2_i = ProdRule(IdealE, IdealV, IdealI, DST, VAR(SRC) / DST.l)
_i2v = ProdRule(IdealE, IdealI, IdealV, SRC, -VAR(DST) / SRC.l)
i2_v = ProdRule(IdealE, IdealI, IdealV, DST, VAR(SRC) / DST.c)
vself = ProdRule(IdealE, IdealV, IdealV, SELF, -VAR(SRC) * SRC.g / SRC.c)
iself = ProdRule(IdealE, IdealI, IdealI, SELF, -VAR(SRC) * SRC.r / SRC.l)
inpv2_v = ProdRule(IdealE, InpV, IdealV, DST, (SRC.fn(TIME) - VAR(DST)) / DST.c / SRC.r)
inpv2_i = ProdRule(IdealE, InpV, IdealI, DST, (SRC.fn(TIME) - VAR(DST) * SRC.r) / DST.l)
inpi2_v = ProdRule(IdealE, InpI, IdealV, DST, (SRC.fn(TIME) - VAR(DST) * SRC.g) / DST.c)
inpi2_i = ProdRule(IdealE, InpI, IdealI, DST, (SRC.fn(TIME) - VAR(DST)) / DST.l / SRC.g)
prod_rules = [_v2i, v2_i, _i2v, i2_v, vself, iself, inpv2_v, inpv2_i, inpi2_v, inpi2_i]
# Production rules account for mismatched parameters
_v2i_mm = ProdRule(
    MmE, IdealV, IdealI, SRC, -EDGE.ws * EDGE.gm_lut(-VAR(DST)) * VAR(DST) / SRC.c
)
v2_i_mm = ProdRule(
    MmE, IdealV, IdealI, DST, EDGE.wt * EDGE.gm_lut(VAR(SRC)) * VAR(SRC) / DST.l
)
_i2v_mm = ProdRule(
    MmE, IdealI, IdealV, SRC, -EDGE.ws * EDGE.gm_lut(-VAR(DST)) * VAR(DST) / SRC.l
)
i2_v_mm = ProdRule(
    MmE, IdealI, IdealV, DST, EDGE.wt * EDGE.gm_lut(VAR(SRC)) * VAR(SRC) / DST.c
)

# If want to do noise simulation, uncomment the following lines
# integrate 4kTrgm (k=1.38e-23, T=300K, r=2/3, g=1e-6) to 100GHz to get noise amplitude
# noise amp = sqrt(4 * 1.38e-23 * 300 * 2/3 * 1e-5 * 100e9) ~= 1e-7
# noise_amp = 1e-7
# _v2i_mm = ProdRule(
#     MmE,
#     IdealV,
#     IdealI,
#     SRC,
#     -EDGE.ws * VAR(DST) / SRC.c,
#     noise_exp=EDGE.ws * noise_amp / SRC.c,
# )
# v2_i_mm = ProdRule(
#     MmE,
#     IdealV,
#     IdealI,
#     DST,
#     EDGE.wt * VAR(SRC) / DST.l,
#     noise_exp=EDGE.wt * noise_amp / DST.l,
# )
# _i2v_mm = ProdRule(
#     MmE,
#     IdealI,
#     IdealV,
#     SRC,
#     -EDGE.ws * VAR(DST) / SRC.l,
#     noise_exp=EDGE.ws * noise_amp / SRC.l,
# )
# i2_v_mm = ProdRule(
#     MmE,
#     IdealI,
#     IdealV,
#     DST,
#     EDGE.wt * VAR(SRC) / DST.c,
#     noise_exp=EDGE.wt * noise_amp / DST.c,
# )
inpv2_v_mm = ProdRule(
    MmE, InpV, IdealV, DST, EDGE.wt * (SRC.fn(TIME) - VAR(DST)) / DST.c / SRC.r
)
inpv2_i_mm = ProdRule(
    MmE, InpV, IdealI, DST, EDGE.wt * (SRC.fn(TIME) - VAR(DST) * SRC.r) / DST.l
)
inpi2_v_mm = ProdRule(
    MmE, InpI, IdealV, DST, EDGE.wt * (SRC.fn(TIME) - VAR(DST) * SRC.g) / DST.c
)
inpi2_i_mm = ProdRule(
    MmE, InpI, IdealI, DST, EDGE.wt * (SRC.fn(TIME) - VAR(DST)) / DST.l / SRC.g
)
# Input current/voltage is not fed through gm, so no lookup table is needed
hw_prod_rules = [
    _v2i_mm,
    v2_i_mm,
    _i2v_mm,
    i2_v_mm,
    inpv2_v_mm,
    inpv2_i_mm,
    inpi2_v_mm,
    inpi2_i_mm,
]
tln_spec.add_production_rules(prod_rules)
mm_tln_spec.add_production_rules(hw_prod_rules)
#### Production rules end ####

#### Validation rules start ####
v_val = ValRule(
    IdealV,
    [
        ValPattern(SRC, IdealE, IdealI, Range(min=0)),
        ValPattern(DST, IdealE, IdealI, Range(min=0)),
        ValPattern(DST, IdealE, InpV, Range(min=0)),
        ValPattern(DST, IdealE, InpI, Range(min=0)),
        ValPattern(SELF, IdealE, IdealV, Range(exact=1)),
    ],
)
i_val = ValRule(
    IdealI,
    [
        ValPattern(SRC, IdealE, IdealV, Range(min=0, max=1)),
        ValPattern(DST, IdealE, [IdealV, InpV, InpI], Range(min=0, max=1)),
        ValPattern(SELF, IdealE, IdealI, Range(exact=1)),
    ],
)
inpv_val = ValRule(
    InpV,
    [
        ValPattern(SRC, IdealE, IdealV, Range(min=0, max=1)),
        ValPattern(SRC, IdealE, IdealI, Range(min=0, max=1)),
    ],
)
inpi_val = ValRule(
    InpI,
    [
        ValPattern(SRC, IdealE, IdealV, Range(min=0, max=1)),
        ValPattern(SRC, IdealE, IdealI, Range(min=0, max=1)),
    ],
)
val_rules = [v_val, i_val, inpv_val, inpi_val]
tln_spec.add_validation_rules(val_rules)
#### Validation rules end ####

if __name__ == "__main__":
    print(v2gm_lut.shape)
    x = np.arange(-1.3, 1.3, 0.001)
    y = lut_from_data(x)
    import matplotlib.pyplot as plt

    plt.plot(x, y)
    plt.show()
