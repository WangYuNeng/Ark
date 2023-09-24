"""
Example: N-Path Filter implemented with a switch-capacitor network
Single input single output, assume ideal switches
"""
from types import FunctionType
import matplotlib.pyplot as plt
import numpy as np
from ark.compiler import ArkCompiler
from ark.rewrite import RewriteGen
from ark.solver import SMTSolver
from ark.validator import ArkValidator
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.range import Range
from ark.specification.specification import CDGSpec
from ark.cdg.cdg import CDG, CDGNode
from ark.specification.cdg_types import NodeType, EdgeType
from ark.specification.production_rule import ProdRule
from ark.specification.rule_keyword import SRC, DST, SELF, EDGE, VAR, TIME
from ark.specification.validation_rule import ValRule, ValPattern
from ark.reduction import SUM

# Capacitors
Cap = NodeType(
    name="Cap",
    order=1,
    reduction=SUM,
    attr_def=[AttrDef("c", attr_type=float, attr_range=Range(min=0))],
)
# Voltage input with series resistor
InpV = NodeType(
    name="InpV",
    order=0,
    attr_def=[
        AttrDef("fn", attr_type=FunctionType, nargs=1),
        AttrDef("r", attr_type=float, attr_range=Range(min=0)),
    ],
)

# Switches controlled by clock
SwE = EdgeType(
    name="SwE",
    attr_def=[
        AttrDef("ctrl", attr_type=FunctionType, nargs=1),
        AttrDef("offset", attr_type=float, attr_range=Range(min=0)),
        AttrDef("period", attr_type=float, attr_range=Range(min=0)),
        AttrDef("duty_cycle", attr_type=float, attr_range=Range(min=0, max=1)),
    ],
)


def ctrl_clk(t, period, duty_cycle, offset):
    """Control clock function"""
    t = (t - offset) % period
    if t <= duty_cycle * period:
        return 1
    return 0


# Example input voltage
def sinosoidal(t):
    # current don't support parameterized function
    # fix the frequency to 100Hz
    return np.sin(2 * np.pi * 0.72e2 * t)


# Production rules
switch_cap_conn = ProdRule(
    SwE,
    InpV,
    Cap,
    DST,
    (SRC.fn(TIME) - VAR(DST))
    / SRC.r
    / DST.c
    * EDGE.ctrl(TIME, EDGE.period, EDGE.duty_cycle, EDGE.offset),
)
prod_rules = [switch_cap_conn]
cdg_types = [Cap, InpV, SwE]
help_fn = [ctrl_clk, sinosoidal]
spec = CDGSpec(cdg_types, prod_rules, None)

# N-path filter implementation
N_PATH = 8
CENTER_FREQ = 1e2
PERIOD = 1 / CENTER_FREQ
DUTY_CYCLE = 1 / N_PATH
TIME_RANGE = [0, 50 * PERIOD]
SEED = 428

n_path_filter = CDG()
inp_v = InpV(fn=sinosoidal, r=1.0)

# Capacitors and switches
caps, switches = [None for _ in range(N_PATH)], [None for _ in range(N_PATH)]
for i in range(N_PATH):
    caps[i] = Cap(c=1e-2)
    switches[i] = SwE(
        ctrl=ctrl_clk, offset=PERIOD / N_PATH * i, period=PERIOD, duty_cycle=DUTY_CYCLE
    )
    n_path_filter.connect(switches[i], inp_v, caps[i])

compiler = ArkCompiler(rewrite=RewriteGen())
compiler.compile(cdg=n_path_filter, cdg_spec=spec, help_fn=help_fn, import_lib={})
mapping = compiler.var_mapping
init_states = compiler.map_init_state({node: 0 for node in mapping.keys()})

sol = compiler.prog(
    TIME_RANGE, init_states=init_states, init_seed=SEED, max_step=PERIOD / 100
)
time_points = sol.t

# Plot the output, can observe the voltage across the capacitors converge to
# the value which is the average during the sampling time.
# The actual filtered output will be choosing the corresponding capacitor voltage in its
# duty cycle, which is shown in the last column.
fig, ax = plt.subplots(nrows=N_PATH + 2, figsize=(4, (N_PATH + 2)))
ax[0].plot(time_points, [sinosoidal(t) for t in time_points], label="Input")
ax[0].legend()
output = np.zeros(len(time_points))
for i in range(0, N_PATH):
    print(sol.y[mapping[caps[i]]][-1])
    ax[i + 1].plot(time_points, sol.y[mapping[caps[i]]], label="Cap %d" % i)
    output += sol.y[mapping[caps[i]]] * np.array(
        [ctrl_clk(t, PERIOD, DUTY_CYCLE, PERIOD / N_PATH * i) for t in time_points]
    )
    ax[i + 1].legend()

ax[-1].plot(time_points, output, label="Output")
ax[-1].legend()
plt.tight_layout()
plt.show()
