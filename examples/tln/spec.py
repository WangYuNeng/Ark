"""
Example: TLN (Transmission Line Network) Circuit
Use LC ladders to emulate the telegrapher's equation
Provide specification for
- Ideal LC ladder
- LC mismatched ladder
- Gain mismatched ladder
"""
from types import FunctionType

import numpy as np
import sympy as sp

from ark.reduction import SUM
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.production_rule import ProdRule
from ark.specification.range import Range
from ark.specification.rule_keyword import (
    SRC,
    DST,
    SELF,
    EDGE,
    VAR,
    TIME,
    sympy_function,
)
from ark.specification.specification import CDGSpec
from ark.specification.validation_rule import ValRule, ValPattern

tln_spec = CDGSpec("tln")
mm_tln_spec = CDGSpec("mm-tln", inherit=tln_spec)


# Example input function
def pulse(
    t, amplitude=1, delay=0, rise_time=5e-9, fall_time=5e-9, pulse_width=10e-9, period=1
):
    """Trapezoidal pulse function"""
    t = (t - delay) % period
    if rise_time <= t and pulse_width + rise_time >= t:
        return amplitude
    elif t < rise_time:
        return amplitude * t / rise_time
    elif pulse_width + rise_time < t and pulse_width + rise_time + fall_time >= t:
        return amplitude * (1 - (t - pulse_width - rise_time) / fall_time)
    return 0

@sympy_function
def pulse_sympy(
        t,
        amplitude=1, delay=0, rise_time=5e-9, fall_time=5e-9, pulse_width=10e-9, period=1
):
    t = (t - delay) % period
    # Use a sympy piecewise function to represent the pulse
    return sp.Piecewise(
        (0, t < rise_time),
        (amplitude * t / rise_time, t < rise_time + pulse_width),
        (amplitude * (1 - (t - pulse_width - rise_time) / fall_time), t < rise_time + pulse_width + fall_time),
        (0, True)
    )

t = sp.symbols('t')
print(pulse_sympy(t))
print(sp.lambdify(t, pulse_sympy(t), 'numpy')(np.linspace(0.0, 1.0, 1000)))


lc_range, gr_range = Range(min=0.1e-9, max=10e-9), Range(min=0)
w_range = Range(min=0.5, max=2)

#### Type definitions start ####
# Ideal implementation
# Parallel capacitor(c=capacitance) and resistor(g=conductance)
IdealV = NodeType(
    name="IdealV",
    attrs={
        "order": 1,
        "reduction": SUM,
        "attr_def": {
            "c": AttrDef(attr_type=float, attr_range=lc_range),
            "g": AttrDef(attr_type=float, attr_range=gr_range),
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
            "l": AttrDef(attr_type=float, attr_range=lc_range),
            "r": AttrDef(attr_type=float, attr_range=gr_range),
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
            "fn": AttrDef(attr_type=FunctionType, nargs=1),
            "r": AttrDef(attr_type=float, attr_range=gr_range),
        },
    },
)
# Current source in Thevenin equivalent
InpI = NodeType(
    name="InpI",
    attrs={
        "order": 0,
        "attr_def": {
            "fn": AttrDef(attr_type=FunctionType, nargs=1),
            "g": AttrDef(attr_type=float, attr_range=gr_range),
        },
    },
)
# Mismatched implementation (c, l and gm (modeled with edge weights))
MmV = NodeType(
    name="MmV",
    bases=IdealV,
    attrs={
        "attr_def": {
            "c": AttrDefMismatch(attr_type=float, attr_range=lc_range, rstd=0.1)
        }
    },
)
MmI = NodeType(
    name="MmI",
    bases=IdealI,
    attrs={
        "attr_def": {
            "l": AttrDefMismatch(attr_type=float, attr_range=lc_range, rstd=0.1)
        }
    },
)
MmE = EdgeType(
    name="MmE",
    bases=IdealE,
    attrs={
        "attr_def": {
            "ws": AttrDefMismatch(attr_type=float, attr_range=w_range, rstd=0.1),
            "wt": AttrDefMismatch(attr_type=float, attr_range=w_range, rstd=0.1),
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
_v2i_mm = ProdRule(MmE, IdealV, IdealI, SRC, -EDGE.ws * VAR(DST) / SRC.c)
v2_i_mm = ProdRule(MmE, IdealV, IdealI, DST, EDGE.wt * VAR(SRC) / DST.l)
_i2v_mm = ProdRule(MmE, IdealI, IdealV, SRC, -EDGE.ws * VAR(DST) / SRC.l)
i2_v_mm = ProdRule(MmE, IdealI, IdealV, DST, EDGE.wt * VAR(SRC) / DST.c)
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
    import ark.visualize.latex_gen as latexlib

    latexlib.language_to_latex(tln_spec)
    latexlib.language_to_latex(mm_tln_spec)
