"""
Cellular Nonlinear Network (CNN) example.
ref: https://onlinelibrary.wiley.com/doi/abs/10.1002/cta.564
"""

import jax.numpy as jnp

from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.attribute_type import AnalogAttr, DigitalAttr, FunctionAttr
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.range import Range
from ark.specification.rule_keyword import DST, EDGE, SELF, SRC, VAR
from ark.specification.specification import CDGSpec
from ark.specification.validation_rule import ValPattern, ValRule

cnn_spec = CDGSpec()
mm_cnn_spec = CDGSpec("mm-cnn", inherit=cnn_spec)


def saturation(sig):
    """Saturate the value at 1"""
    return 0.5 * (jnp.abs(sig + 1) - jnp.abs(sig - 1))


def saturation_diffpair(sig):
    """Saturation function for diffpair implementation"""
    sat_sig = saturation(sig)
    return sat_sig / 0.707107 * jnp.sqrt(1 - jnp.square(sat_sig / 2 / 0.707107))


lc_range, gr_range = Range(min=0.1e-9, max=10e-9), Range(min=0)
param_range = (-10, 10)
positive = Range(min=0)
#### Type definitions start ####
# Cells in CNN, z is the bias
IdealV = NodeType(
    name="IdealV",
    attrs={
        "order": 1,
        "attr_def": {
            "z": AttrDef(attr_type=AnalogAttr(param_range)),
        },
    },
)
# Output function
Out = NodeType(
    name="Out",
    attrs={
        "order": 0,
        "attr_def": {
            "act": AttrDef(attr_type=FunctionAttr(nargs=1)),
        },
    },
)
# Input, should be stateless, setting to 1 just for convenience of setting its value
Inp = NodeType(name="Inp", attrs={"order": 1})
MapE = EdgeType(name="MapE")
FlowE = EdgeType(
    name="FlowE", attrs={"attr_def": {"g": AttrDef(attr_type=AnalogAttr(param_range))}}
)

# Mismatched implementation
Vm = NodeType(
    name="Vm",
    bases=IdealV,
    attrs={
        "attr_def": {
            "mm": AttrDefMismatch(attr_type=AnalogAttr((1, 1)), rstd=0.1),
        },
    },
)
fEm_1p = EdgeType(
    name="fEm_1p",
    bases=FlowE,
    attrs={
        "attr_def": {
            "g": AttrDefMismatch(attr_type=AnalogAttr(param_range), rstd=0.01),
        },
    },
)

fEm_10p = EdgeType(
    name="fEm_10p",
    bases=FlowE,
    attrs={
        "attr_def": {
            "g": AttrDefMismatch(attr_type=AnalogAttr(param_range), rstd=0.1),
        },
    },
)
quantization_levels = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
fEm_1p_quantized = EdgeType(
    name="fEm_1p_quantized",
    bases=FlowE,
    attrs={
        "attr_def": {
            "g": AttrDefMismatch(attr_type=DigitalAttr(quantization_levels), rstd=0.01),
        },
    },
)
V_qunatized = NodeType(
    name="V_quantized",
    bases=IdealV,
    attrs={
        "attr_def": {
            "z": AttrDef(attr_type=DigitalAttr(quantization_levels)),
        },
    },
)

cdg_types = [IdealV, Out, Inp, MapE, FlowE]
mm_cdg_types = [Vm, fEm_1p, fEm_10p, fEm_1p_quantized, V_qunatized]
cnn_spec.add_cdg_types(cdg_types)
mm_cnn_spec.add_cdg_types(mm_cdg_types)
#### Type definitions end ####

#### Production rules start ####
Bmat = ProdRule(FlowE, Inp, IdealV, DST, EDGE.g * VAR(SRC))
Dummy = ProdRule(FlowE, Inp, IdealV, SRC, 0)  # Dummy rule to make sure Inp is not used
ReadOut = ProdRule(MapE, IdealV, Out, DST, DST.act(VAR(SRC)))
SelfFeedback = ProdRule(MapE, IdealV, IdealV, SELF, -VAR(SRC) + SRC.z)
Amat = ProdRule(FlowE, Out, IdealV, DST, EDGE.g * VAR(SRC))
# Production rules for msimatch v
Bmat_mm = ProdRule(FlowE, Inp, Vm, DST, DST.mm * EDGE.g * VAR(SRC))
SelfFeedback_mm = ProdRule(MapE, Vm, Vm, SELF, SRC.mm * (-VAR(SRC) + SRC.z))
Amat_mm = ProdRule(FlowE, Out, Vm, DST, DST.mm * EDGE.g * VAR(SRC))
prod_rules = [Bmat, Dummy, ReadOut, SelfFeedback, Amat, Bmat_mm]
mm_prod_rules = [SelfFeedback_mm, Amat_mm]
cnn_spec.add_production_rules(prod_rules)
mm_cnn_spec.add_production_rules(mm_prod_rules)
#### Production rules end ####

#### Validation rules start ####
v_val = ValRule(
    IdealV,
    [
        ValPattern(SRC, MapE, Out, Range(exact=1)),
        ValPattern(DST, FlowE, Out, Range(min=4, max=9)),
        ValPattern(DST, FlowE, Inp, Range(min=4, max=9)),
        ValPattern(SELF, MapE, IdealV, Range(exact=1)),
    ],
)
out_val = ValRule(
    Out,
    [
        ValPattern(SRC, FlowE, IdealV, Range(min=4, max=9)),
        ValPattern(DST, MapE, IdealV, Range(exact=1)),
    ],
)
inp_val = ValRule(Inp, [ValPattern(SRC, FlowE, IdealV, Range(min=4, max=9))])
val_rules = [v_val, out_val, inp_val]
cnn_spec.add_validation_rules(val_rules)
#### Validation rules end ####
