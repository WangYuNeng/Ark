from diffrax.solver import Tsit5
from sympy import *

from ark.ark import ArkCompiler
from ark.cdg.cdg import CDG
from ark.optimization.opt_compiler import OptCompiler
from ark.reduction import PRODUCT
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.attribute_type import (
    AnalogAttr,
    DigitalAttr,
    FunctionAttr,
    Trainable,
)
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import DST, EDGE, SRC, VAR
from ark.specification.specification import CDGSpec

# Specification of coupled oscillator
co_spec = CDGSpec("co")

#### Type definitions start ####
# Oscillator node: The state variable models the displacement of the oscillator
# Order = 2 means the state variable controlled by its second derivative - acceleration
# The mass attribute models the mass of the oscillator
Osc = NodeType(
    name="Osc",
    attrs={
        "order": 2,
        "reduction": PRODUCT,
        "attr_def": {
            "mass": AttrDefMismatch(
                attr_type=AnalogAttr(val_range=(0.0, 10000.0)), rstd=0.1
            )
        },
    },
)

# Coupling springs
# k: coupling strength
Coupling = EdgeType(
    name="Coupling",
    attrs={
        "attr_def": {
            "k": AttrDefMismatch(
                attr_type=DigitalAttr(val_choices=[2, 4, 6, 8]), rstd=0.2
            )
        }
    },
)
test_w_fn = EdgeType(
    name="Cpl_w_fn",
    attrs={"attr_def": {"fn": AttrDef(attr_type=FunctionAttr(nargs=1))}},
)

cdg_types = [Osc, Coupling]
co_spec.add_cdg_types(cdg_types)
#### Type definitions end ####

#### Production rules start ####
# F = ma = -kx -> a = x'' = -kx/m
r_cp_src = ProdRule(
    Coupling, Osc, Osc, SRC, -EDGE.k * (VAR(SRC) / SRC.mass - VAR(DST) / SRC.mass)
)
r_cp_dst = ProdRule(
    Coupling, Osc, Osc, DST, -EDGE.k * (VAR(DST) / SRC.mass - VAR(SRC) / SRC.mass)
)
production_rules = [r_cp_src, r_cp_dst]
co_spec.add_production_rules(production_rules)

# Manipulate the dynamical system with the CDG and execute it with Ark
compiler = ArkCompiler()

# Initialize 2 couplied oscillatorsxx
graph = CDG()
node1 = Osc(mass=1000.0)
node2 = Osc(mass=2000.0)
cpl = Coupling(k=Trainable())

graph.connect(cpl, node1, node2)
# print(graph.element_to_attr_sample())

exprs = compiler.compile_sympy_diffeqs(cdg=graph, cdg_spec=co_spec)
a = pretty(exprs, use_unicode=False)
print(a)

TestClass = OptCompiler().compile("test", graph, co_spec)
test = TestClass(init_trainable=[1, 2, 10], is_stochastic=False, solver=Tsit5())
a = test([], 0, 0)
import matplotlib.pyplot as plt

plt.plot(a)
plt.show()
plt.show()
import matplotlib.pyplot as plt

plt.plot(a)
plt.show()
plt.show()
