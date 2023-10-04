"""
Example: N-Path Filter implemented with a switch-capacitor network
Single input single output, assume ideal switches
"""
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np

from ark.ark import Ark
from ark.cdg.cdg import CDG, CDGNode
from ark.reduction import SUM
from ark.specification.attribute_def import AttrDef
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.range import Range
from ark.specification.rule_keyword import DST, EDGE, SRC, TIME, VAR
from ark.specification.specification import CDGSpec

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
# Voltage input with series resistor
InpV = NodeType(
    name="InpV",
    attrs={
        "order": 0,
        "attr_def": {
            "fn": AttrDef(attr_type=FunctionType, nargs=1),
            "r": AttrDef(attr_type=float, attr_range=Range(min=0)),
        },
    },
)

# Switches controlled by clock
SwE = EdgeType(
    name="SwE",
    attrs={
        "attr_def": {
            "ctrl": AttrDef(attr_type=FunctionType, nargs=1),
            "offset": AttrDef(attr_type=float, attr_range=Range(min=0)),
            "period": AttrDef(attr_type=float, attr_range=Range(min=0)),
            "duty_cycle": AttrDef(attr_type=float, attr_range=Range(min=0, max=1)),
        },
    },
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
import_fn = {"ctrl_clk": ctrl_clk, "sinosoidal": sinosoidal}
spec = CDGSpec(cdg_types=cdg_types, production_rules=prod_rules, validation_rules=None)

system = Ark(cdg_spec=spec)

# N-path filter implementation
N_PATH = 8
CENTER_FREQ = 1e2
PERIOD = 1 / CENTER_FREQ
DUTY_CYCLE = 1 / N_PATH
TIME_RANGE = [0, 50 * PERIOD]

n_path_filter = CDG()
inp_v = InpV(fn=sinosoidal, r=1.0)

# Capacitors and switches
caps: list[CDGNode]
caps, switches = [None for _ in range(N_PATH)], [None for _ in range(N_PATH)]
for i in range(N_PATH):
    caps[i] = Cap(c=1e-2)
    switches[i] = SwE(
        ctrl=ctrl_clk, offset=PERIOD / N_PATH * i, period=PERIOD, duty_cycle=DUTY_CYCLE
    )
    n_path_filter.connect(switches[i], inp_v, caps[i])

system.compile(cdg=n_path_filter)
n_path_filter.initialize_all_states(val=0)
time_points = np.linspace(*TIME_RANGE, 1000)
system.execute(
    cdg=n_path_filter,
    time_eval=time_points,
    max_step=PERIOD / 100,
)

# Plot the output, can observe the voltage across the capacitors converge to
# the value which is the average during the sampling time.
# The actual filtered output will be choosing the corresponding capacitor voltage in its
# duty cycle, which is shown in the last column.
fig, ax = plt.subplots(nrows=N_PATH + 2, figsize=(4, (N_PATH + 2)))
ax[0].plot(time_points, [sinosoidal(t) for t in time_points], label="Input")
ax[0].legend()
output = np.zeros(len(time_points))
for i, node in enumerate(caps):
    trace = node.get_trace(n=0)
    ax[i + 1].plot(time_points, trace, label="Cap %d" % i)
    output += trace * np.array(
        [ctrl_clk(t, PERIOD, DUTY_CYCLE, PERIOD / N_PATH * i) for t in time_points]
    )
    ax[i + 1].legend()

ax[-1].plot(time_points, output, label="Output")
ax[-1].legend()
plt.tight_layout()
plt.show()
