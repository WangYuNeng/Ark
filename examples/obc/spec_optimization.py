import jax.numpy as jnp
from pattern_recog_parser import args
from spec import Coupling, Osc, obc_spec

from ark.specification.attribute_def import AttrDef
from ark.specification.attribute_type import AnalogAttr, DigitalAttr, FunctionAttr
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import DST, EDGE, SELF, SRC, VAR

T = 1


def locking_fn(x, lock_strength: float):
    """Injection locking function from [2]
    Modify the leading coefficient to 1.2 has a better outcome
    """
    return lock_strength * jnp.sin(2 * jnp.pi * x)


def coupling_fn(x, cpl_strength: float):
    """Coupling function from [2]"""
    return cpl_strength * jnp.sin(jnp.pi * x)


Osc_modified = NodeType(
    "Osc_modified",
    bases=Osc,
    attrs={
        "order": 1,
        "attr_def": {
            "lock_fn": AttrDef(attr_type=FunctionAttr(nargs=2)),
            "osc_fn": AttrDef(attr_type=FunctionAttr(nargs=2)),
            "lock_strength": AttrDef(attr_type=AnalogAttr((-10, 10))),
            "cpl_strength": AttrDef(attr_type=AnalogAttr((-10, 10))),
        },
    },
)

# Digital coupling with 3 bit resolution -2**(n_bit-1) to 2**(n_bit-1) -1
# Rescale to between +/- 1
if args.weight_bits is None:
    N_BITS = 3
else:
    N_BITS = args.weight_bits
N_CHOICES = 2**N_BITS


def nbits_to_val_choices(n_bits: int) -> list[float]:
    if n_bits == 1:
        return [-1, 1]
    return [(-(2 ** (n_bits - 1)) + i) * 2 / N_CHOICES for i in range(N_CHOICES)]


val_choices = nbits_to_val_choices(N_BITS)
Cpl_digital = EdgeType(
    "Cpl_digital",
    bases=Coupling,
    attrs={
        "attr_def": {
            "k": AttrDef(attr_type=DigitalAttr(val_choices=val_choices)),
        },
    },
)

modified_cp_src = ProdRule(
    Coupling,
    Osc_modified,
    Osc_modified,
    SRC,
    -EDGE.k * SRC.osc_fn(VAR(SRC) - VAR(DST), SRC.cpl_strength),
    noise_exp=1e-1,
)

modified_cp_dst = ProdRule(
    Coupling,
    Osc_modified,
    Osc_modified,
    DST,
    -EDGE.k * DST.osc_fn(VAR(DST) - VAR(SRC), DST.cpl_strength),
    noise_exp=1e-1,
)

modified_cp_self = ProdRule(
    Coupling,
    Osc_modified,
    Osc_modified,
    SELF,
    -EDGE.k * SRC.lock_fn(VAR(SRC), SRC.lock_strength),
)

obc_spec.add_production_rules([modified_cp_src, modified_cp_dst, modified_cp_self])
