"""
Example: Coupled Oscillator Network
ref: https://ieeexplore.ieee.org/document/9531734

"""

from types import FunctionType
import numpy as np
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.specification import CDGSpec
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import SRC, DST, SELF, EDGE, VAR
from ark.specification.range import Range

obc_spec = CDGSpec()

FREQ = 795.8e6
T = 1 / FREQ

def calc_offset_std(rstd: float):
    """Calculated the standard deviation of the offset

    Args:
        rstd (float): relative std to the
        amplitude of the coupling function

    Returns:
        float: standard deviation of the offset
    """
    return rstd * 2 * FREQ
OFFSET_STD = calc_offset_std(0.1)

def locking_fn(x):
    """Injection locking function from [2]
    Modify the leading coefficient to 1.2 has a better outcome
    """
    return 1.2 * 795.8e6 * np.sin(2 * np.pi * x)

def coupling_fn(x):
    """Coupling function from [2]"""
    return 2 * 795.8e6 * np.sin(np.pi * x)

#### Type definitions start ####
# Phase of an oscillator:
# lock_fn: injection locking, e.g., omeag_s sin (2 phi)
# osc_fn: coupling, e.g., omega_c sin (phi_i - phi_j)
Osc = NodeType(name='Osc', order=1, attr_def=[AttrDef('lock_fn', attr_type=FunctionType,
                                                      nargs=1),
                                               AttrDef('osc_fn', attr_type=FunctionType,
                                                       nargs=1)])

# k: coupling strength
Coupling = EdgeType(name='Coupling', attr_def=[AttrDef('k', attr_type=float)])
Coupling_offset = EdgeType(name='Coupling_offset', base=Coupling,
                                  attr_def=[AttrDefMismatch('offset', attr_type=float,
                                                            std=OFFSET_STD,
                                                            attr_range=Range(exact=0))])
cdg_types = [Osc, Coupling, Coupling_offset]
obc_spec.add_cdg_types(cdg_types)
#### Type definitions end ####

#### Production rules start ####
r_cp_src = ProdRule(Coupling, Osc, Osc, SRC, - EDGE.k * SRC.osc_fn(VAR(SRC) - VAR(DST)))
r_cp_dst = ProdRule(Coupling, Osc, Osc, DST, - EDGE.k * DST.osc_fn(VAR(DST) - VAR(SRC)))
r_lock = ProdRule(Coupling, Osc, Osc, SELF, - SRC.lock_fn(VAR(SRC)))
r_cp_offset_src = ProdRule(Coupling_offset, Osc, Osc, SRC,
                           - EDGE.k * (SRC.osc_fn(VAR(SRC) - VAR(DST)) + EDGE.offset))
r_cp_offset_dst = ProdRule(Coupling_offset, Osc, Osc, DST,
                           - EDGE.k * (SRC.osc_fn(VAR(DST) - VAR(SRC)) + EDGE.offset))
production_rules = [r_cp_src, r_cp_dst, r_lock, r_cp_offset_src, r_cp_offset_dst]
obc_spec.add_production_rules(production_rules)
#### Production rules end ####

#### Validation rules start ####
#### Validation rules end ####

