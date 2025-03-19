from ark.specification.attribute_def import AttrDef
from ark.specification.attribute_type import AnalogAttr, FunctionAttr
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import DST, EDGE, SELF, SRC, VAR
from ark.specification.specification import CDGSpec

# Cellular nonlinear network
cnn_spec = CDGSpec("cnn")

# Oscillator-based computing
obc_spec = CDGSpec("obc")

# Nonlinear network that might exhibit chaos
# https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.61.259
# dh_i/dt = -h_i + sum_j J_ij tanh(g*h_j)
cann_spec = CDGSpec("nln")

# Input that is fixed in the dynamic
# should be stateless, setting for convenience of setting its value
Inp = NodeType(name="Inp", attrs={"order": 1})

# ------- CNN specification -------
#### Type definitions start ####
# Cells in CNN, z is the bias
Neuron = NodeType(
    name="IdealV",
    attrs={
        "order": 1,
        "attr_def": {
            "z": AttrDef(attr_type=AnalogAttr()),
            "act": AttrDef(attr_type=FunctionAttr(nargs=1)),
        },
    },
)

FlowE = EdgeType(
    name="FlowE", attrs={"attr_def": {"g": AttrDef(attr_type=AnalogAttr())}}
)

cdg_types = [Neuron, Inp, FlowE]
cnn_spec.add_cdg_types(cdg_types)
#### Type definitions end ####

#### Production rules start ####
Amat = ProdRule(FlowE, Neuron, Neuron, DST, EDGE.g * DST.act(VAR(SRC)))
Bmat = ProdRule(FlowE, Inp, Neuron, DST, EDGE.g * VAR(SRC))
SelfFeedback = ProdRule(FlowE, Neuron, Neuron, SELF, -VAR(SRC) + SRC.z)
prod_rules = [Bmat, SelfFeedback, Amat]
cnn_spec.add_production_rules(prod_rules)
#### Production rules end ####
# ------- CNN specification end -------

# ------- OBC specification -------
#### Type definitions start ####
# Phase of an oscillator:
# lock_fn: injection locking, e.g., omeag_s sin (2 phi)
# osc_fn: coupling, e.g., omega_c sin (phi_i - phi_j)
Osc = NodeType(
    name="Osc",
    attrs={
        "order": 1,
        "attr_def": {
            "lock_fn": AttrDef(attr_type=FunctionAttr(nargs=2)),
            "osc_fn": AttrDef(attr_type=FunctionAttr(nargs=2)),
            "lock_strength": AttrDef(attr_type=AnalogAttr()),
            "cpl_strength": AttrDef(attr_type=AnalogAttr()),
        },
    },
)

# k: coupling strength
Coupling = EdgeType(
    name="Coupling", attrs={"attr_def": {"k": AttrDef(attr_type=AnalogAttr())}}
)
cdg_types = [Osc, Coupling, Inp]
obc_spec.add_cdg_types(cdg_types)
#### Type definitions end ####

#### Production rules start ####
r_cp_src = ProdRule(
    Coupling,
    Osc,
    Osc,
    SRC,
    -EDGE.k * SRC.osc_fn(VAR(SRC) - VAR(DST), SRC.cpl_strength),
)
r_cp_dst = ProdRule(
    Coupling,
    Osc,
    Osc,
    DST,
    -EDGE.k * DST.osc_fn(VAR(DST) - VAR(SRC), DST.cpl_strength),
)
inp_cp_osc = ProdRule(
    Coupling,
    Inp,
    Osc,
    DST,
    -EDGE.k * DST.osc_fn(VAR(DST) - VAR(SRC), DST.cpl_strength),
)
r_lock = ProdRule(
    Coupling,
    Osc,
    Osc,
    SELF,
    -EDGE.k * SRC.lock_fn(VAR(SRC), SRC.lock_strength),
    noise_exp=1e-1,
)
production_rules = [r_cp_src, r_cp_dst, r_lock]
obc_spec.add_production_rules(production_rules)


def set_obc_noise_val(obc_spec: CDGSpec, noise_val: float):
    obc_spec.production_rules[2].noise_exp = noise_val


#### Production rules end ####
# ------- OBC specification end -------

# ------- N specification -------
#### Type definitions start ####
# Share Neuron type with CNN,
# Share Coupling type with OBC
cann_spec.add_cdg_types([Neuron, Inp, Coupling])
#### Type definitions end ####
#### Production rules start ####
n_cp_dst = ProdRule(
    Coupling,
    Neuron,
    Neuron,
    DST,
    -EDGE.k * DST.act_fn(DST.g * VAR(SRC)),
)  # The J matrix is not symmetric, the "k" value is not shared and need edges
# for source to destination and destination to source, respectively (unlike OBC)
cp_self = ProdRule(
    Coupling,
    Neuron,
    Neuron,
    SELF,
    -VAR(SRC),
)
inp_cp_neuron = ProdRule(
    Coupling,
    Inp,
    Neuron,
    DST,
    -EDGE.k * DST.act_fn(DST.g * VAR(SRC)),
)
production_rules = [n_cp_dst, cp_self, inp_cp_neuron]
cann_spec.add_production_rules(production_rules)
#### Production rules end ####
# ------- NNL specification end -------
