"""
Example: N-Path Filter implemented with a switch-capacitor network
Assume ideal switches
"""
from types import FunctionType

import numpy as np

from ark.reduction import SUM
from ark.specification.attribute_def import AttrDef
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.range import Range
from ark.specification.rule_keyword import DST, EDGE, SRC, TIME, VAR
from ark.specification.specification import CDGSpec

sc_filter_spec = CDGSpec("sc-filter")


def clk(t, period, duty_cycle, offset):
    """clock function"""
    t = (t - offset) % period
    if t <= duty_cycle * period:
        return 1
    return 0


def sinosoidal(t, freq=1e2):
    return np.sin(2 * np.pi * freq * t)


#### Type definitions start ####
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
# Resistor
Res = NodeType(
    name="Res",
    attrs={
        "order": 0,
        "reduction": SUM,
        "attr_def": {
            "r": AttrDef(attr_type=float, attr_range=Range(min=0)),
        },
    },
)

# Voltage input with series resistor
InpV = NodeType(
    name="InpV",
    attrs={
        "order": 0,
        "attr_def": {"fn": AttrDef(attr_type=FunctionType, nargs=1)},
    },
)
# Edge to denote connection only
Wire = EdgeType(name="Wire")
# Switches controlled by clock
SwE = EdgeType(
    name="SwE",
    attrs={
        "attr_def": {
            "ctrl": AttrDef(attr_type=FunctionType, nargs=1),
        },
    },
)
cdg_types = [Cap, InpV, Res, SwE, Wire]
sc_filter_spec.add_cdg_types(cdg_types)
#### Type definitions end ####

#### Production rules start ####
inpv_to_r = ProdRule(Wire, InpV, Res, DST, (SRC.fn(TIME) / DST.r))
r_to_cap = ProdRule(SwE, Res, Cap, SRC, -VAR(DST) / SRC.r * EDGE.ctrl(TIME))
cap_from_r = ProdRule(SwE, Res, Cap, DST, VAR(SRC) / DST.c * EDGE.ctrl(TIME))
prod_rules = [inpv_to_r, r_to_cap, cap_from_r]
sc_filter_spec.add_production_rules(prod_rules)
#### Production rules end ####
