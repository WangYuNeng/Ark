"""
Example: TLN (Transmission Line Network) Circuit
Generalize the LC ladder to have potential bandpass structure.
That is, instead of series inductor and shunt capacitor, we have
series inductor with capacitor and shunt inductor with capacitor.
"""
from types import FunctionType

from spec import pulse

from ark.reduction import SUM
from ark.specification.attribute_def import AttrDef
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.range import Range
from ark.specification.rule_keyword import DST, SRC, TIME, VAR
from ark.specification.specification import CDGSpec
from ark.specification.validation_rule import ValPattern, ValRule

bp_tln_spec = CDGSpec("bp-tln")
mm_tln_spec = CDGSpec("mm-bp-tln", inherit=bp_tln_spec)

lc_range, gr_range = Range(min=0.1e-9, max=10e-9), Range(min=0.0)
w_range = Range(min=0.5, max=2.0)

#### Type definitions start ####
# Ideal implementation
# Parallel capacitor(c=capacitance) and inductor(l=inductance)
# TODO: Add parallel resistor
ShuntV = NodeType(
    name="ShuntV",
    attrs={
        "order": 1,
        "reduction": SUM,
        "attr_def": {
            "c": AttrDef(attr_type=float, attr_range=lc_range),
        },
    },
)
ShuntI = NodeType(
    name="ShuntI",
    attrs={
        "order": 1,
        "reduction": SUM,
        "attr_def": {
            "l": AttrDef(attr_type=float, attr_range=lc_range),
        },
    },
)
# Series inductor(l=inductance) and capacitor(c=capacitance)
SeriesI = NodeType(
    name="SeriesI",
    attrs={
        "order": 1,
        "reduction": SUM,
        "attr_def": {
            "l": AttrDef(attr_type=float, attr_range=lc_range),
        },
    },
)
SeriesV = NodeType(
    name="SeriesV",
    attrs={
        "order": 1,
        "reduction": SUM,
        "attr_def": {
            "c": AttrDef(attr_type=float, attr_range=lc_range),
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

cdg_types = [ShuntV, ShuntI, SeriesI, SeriesV, InpV, InpI]
bp_tln_spec.add_cdg_types(cdg_types)
#### Type definitions end ####

#### Production rules start ####
shunt_v_to_series_i_s = ProdRule(IdealE, ShuntV, SeriesI, SRC, -VAR(DST) / SRC.c)
shunt_v_to_series_i_d = ProdRule(IdealE, ShuntV, SeriesI, DST, VAR(SRC) / DST.l)
series_i_to_shunt_v_s = ProdRule(IdealE, SeriesI, ShuntV, SRC, -VAR(DST) / SRC.l)
series_i_to_shunt_v_d = ProdRule(IdealE, SeriesI, ShuntV, DST, VAR(SRC) / DST.c)
series_i_to_series_v_s = ProdRule(IdealE, SeriesI, SeriesV, SRC, -VAR(DST) / SRC.l)
series_i_to_series_v_d = ProdRule(IdealE, SeriesI, SeriesV, DST, VAR(SRC) / DST.c)
shunt_v_to_shunt_i_s = ProdRule(IdealE, ShuntV, ShuntI, SRC, -VAR(DST) / SRC.c)
shunt_v_to_shunt_i_d = ProdRule(IdealE, ShuntV, ShuntI, DST, VAR(SRC) / DST.l)
inpv_to_shunt_v = ProdRule(
    IdealE, InpV, ShuntV, DST, (SRC.fn(TIME) - VAR(DST)) / DST.c / SRC.r
)
inpv_to_series_i = ProdRule(
    IdealE, InpV, SeriesI, DST, (SRC.fn(TIME) - VAR(DST) * SRC.r) / DST.l
)
inpi_to_shunt_v = ProdRule(
    IdealE, InpI, ShuntV, DST, (SRC.fn(TIME) - VAR(DST) * SRC.g) / DST.c
)
inpi_to_series_i = ProdRule(
    IdealE, InpI, SeriesI, DST, (SRC.fn(TIME) - VAR(DST)) / DST.l / SRC.g
)
prod_rules = [
    shunt_v_to_series_i_s,
    shunt_v_to_series_i_d,
    series_i_to_shunt_v_s,
    series_i_to_shunt_v_d,
    series_i_to_series_v_s,
    series_i_to_series_v_d,
    shunt_v_to_shunt_i_s,
    shunt_v_to_shunt_i_d,
    inpv_to_shunt_v,
    inpv_to_series_i,
    inpi_to_shunt_v,
    inpi_to_series_i,
]
bp_tln_spec.add_production_rules(prod_rules)
#### Production rules end ####

#### Validation rules start ####
shunt_v_val = ValRule(
    ShuntV,
    [
        ValPattern(SRC, IdealE, SeriesI, Range(min=0)),
        ValPattern(DST, IdealE, SeriesI, Range(min=0)),
        ValPattern(DST, IdealE, InpV, Range(min=0)),
        ValPattern(DST, IdealE, InpI, Range(min=0)),
        ValPattern(SRC, IdealE, ShuntI, Range(min=0, max=1)),
    ],
)
series_i_val = ValRule(
    SeriesI,
    [
        ValPattern(SRC, IdealE, ShuntV, Range(min=0, max=1)),
        ValPattern(DST, IdealE, [ShuntV, InpV, InpI], Range(min=0, max=1)),
        ValPattern(SRC, IdealE, SeriesV, Range(min=0, max=1)),
    ],
)
series_v_val = ValRule(
    SeriesV,
    [
        ValPattern(DST, IdealE, SeriesI, Range(min=0, max=1)),
    ],
)
shunt_i_val = ValRule(
    ShuntI,
    [
        ValPattern(DST, IdealE, ShuntV, Range(min=0, max=1)),
    ],
)
inpv_val = ValRule(
    InpV,
    [
        ValPattern(SRC, IdealE, ShuntV, Range(min=0, max=1)),
        ValPattern(SRC, IdealE, SeriesI, Range(min=0, max=1)),
    ],
)
inpi_val = ValRule(
    InpI,
    [
        ValPattern(SRC, IdealE, ShuntV, Range(min=0, max=1)),
        ValPattern(SRC, IdealE, SeriesI, Range(min=0, max=1)),
    ],
)
val_rules = [
    shunt_v_val,
    series_i_val,
    series_v_val,
    shunt_i_val,
    inpv_val,
    inpi_val,
]
bp_tln_spec.add_validation_rules(val_rules)
#### Validation rules end ####


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from ark.ark import Ark
    from ark.cdg.cdg import CDG
    from ark.specification.cdg_types import EdgeType, NodeType
    from ark.specification.specification import CDGSpec

    line_len = 4
    time_range = [0, 5e-8]
    time_points = np.linspace(*time_range, 1001, endpoint=True)
    system = Ark(cdg_spec=bp_tln_spec)

    line = CDG()
    series_i_nodes = [SeriesI(l=1e-9) for _ in range(line_len)]
    series_v_nodes = [SeriesV(c=1e-9) for _ in range(line_len)]
    shunt_v_nodes = [ShuntV(c=1e-9) for _ in range(line_len)]
    shunt_i_nodes = [ShuntI(l=1e-9) for _ in range(line_len)]
    # Create the edges
    for i in range(line_len):
        line.connect(IdealE(), series_i_nodes[i], shunt_v_nodes[i])
        line.connect(IdealE(), series_i_nodes[i], series_v_nodes[i])
        line.connect(IdealE(), shunt_v_nodes[i], shunt_i_nodes[i])
        if not i == line_len - 1:
            line.connect(IdealE(), shunt_v_nodes[i], series_i_nodes[i + 1])

    # Create the first shunt lc
    shunt_v_in, shunt_i_in = ShuntV(c=1e-9), ShuntI(l=1e-9)
    line.connect(IdealE(), shunt_v_in, shunt_i_in)
    line.connect(IdealE(), shunt_v_in, series_i_nodes[0])
    # Create the input
    line.connect(IdealE(), InpI(fn=pulse, g=0.0), shunt_v_in)
    line.initialize_all_states(val=0)

    system.validate(line)
    system.compile(line)
    system.execute(cdg=line, time_eval=time_points, max_step=1e-9)
    plt.plot(time_points, shunt_v_nodes[-1].get_trace(0), label="line_end")
    plt.plot(time_points, shunt_v_in.get_trace(0), label="line_start")
    plt.legend()
    plt.show()
