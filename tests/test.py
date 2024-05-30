import jax.numpy as jnp
from diffrax.solver import Euler, Tsit5
from sympy import *

from ark.ark import ArkCompiler
from ark.cdg.cdg import CDG
from ark.optimization.base_module import TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.reduction import PRODUCT
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.attribute_type import AnalogAttr, FunctionAttr, Trainable
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
                attr_type=AnalogAttr(val_range=(0.0, 10.0)), rstd=0.1
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
            "k": AttrDefMismatch(attr_type=AnalogAttr(val_range=(0.0, 10.0)), rstd=0.2)
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
    Coupling,
    Osc,
    Osc,
    SRC,
    -EDGE.k * (VAR(SRC) / SRC.mass - VAR(DST) / SRC.mass),
    noise_exp=VAR(SRC) / VAR(SRC) * 0.1 / DST.mass,
)
r_cp_dst = ProdRule(
    Coupling,
    Osc,
    Osc,
    DST,
    -EDGE.k * (VAR(DST) / DST.mass - VAR(SRC) / DST.mass),
    noise_exp=VAR(SRC) / VAR(SRC) * 1 / DST.mass,
)
r_fn = ProdRule(test_w_fn, Osc, Osc, DST, VAR(DST) + EDGE.fn(VAR(SRC)))
production_rules = [r_cp_src, r_cp_dst, r_fn]
co_spec.add_production_rules(production_rules)

# Manipulate the dynamical system with the CDG and execute it with Ark
compiler = ArkCompiler()

# Initialize 2 couplied oscillatorsxx
graph = CDG()
node1 = Osc(mass=5.0)
node2 = Osc(mass=Trainable(1))


def test_fn(x):
    return x**2


fn_edge = test_w_fn(fn=test_fn)
cpl = Coupling(k=Trainable(0))

graph.connect(cpl, node1, node2)
# graph.connect(fn_edge, node1, node2)
# print(graph.element_to_attr_sample())

exprs = compiler.compile_sympy_diffeqs(cdg=graph, cdg_spec=co_spec)
a = pretty(exprs, use_unicode=False)
print(a)
exprs = compiler.compile_sympy_diffeqs(cdg=graph, cdg_spec=co_spec, noise_ode=True)
a = pretty(exprs, use_unicode=False)
print(a)

time_info = TimeInfo(t0=0, t1=1, dt0=0.01, saveat=jnp.linspace(0, 1, 100))
y0 = jnp.array([1, 0, 2, 0])

TestClass = OptCompiler().compile(
    "test", graph, co_spec, normalize_weight=True, do_clipping=False
)
test = TestClass(init_trainable=jnp.array([0, 0]), is_stochastic=False, solver=Tsit5())
import matplotlib.pyplot as plt

for i in range(10):
    a = test(
        time_info,
        y0,
        [],
        i,
        0,
    )

    plt.plot(a)
plt.show()

test = TestClass(init_trainable=jnp.array([1]), is_stochastic=True, solver=Euler())
for i in range(10):
    a = test(
        time_info,
        y0,
        [],
        0,
        i,
    )
    import matplotlib.pyplot as plt

    plt.plot(a)
plt.show()
plt.show()
plt.show()
