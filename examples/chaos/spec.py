# Implement the MNIST classifier with Lorenz96 chaotic system from https://arxiv.org/pdf/2406.02580
# Lorenz96 system dyanmics: dx_i/dt = (x_{i+1} - x_{i-2})x_{i-1} - x_i + F

from typing import Optional

from ark.cdg.cdg import CDG
from ark.compiler import ArkCompiler
from ark.reduction import PRODUCT, SUM
from ark.specification.attribute_def import AttrDef
from ark.specification.attribute_type import AnalogAttr
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule
from ark.specification.range import Range
from ark.specification.rule_keyword import DST, SELF, SRC, VAR
from ark.specification.specification import CDGSpec
from ark.specification.trainable import TrainableMgr

# Ark specificaton
lorenz96_spec = CDGSpec("lorenz96")

# Node and edge types
# State variable xi
State = NodeType(
    name="State",
    attrs={
        "order": 1,
        "reduction": SUM,
        "attr_def": {
            "F": AttrDef(attr_type=AnalogAttr(Range(-100, 100))),
        },
    },
)
# Auxiliary node for product terms x_{i+1}x_{i-1} and x_{i-2}x_{i-1}
ProdAux = NodeType(
    name="ProdAux",
    attrs={
        "order": 0,
        "reduction": PRODUCT,
        "attr_def": {
            "sign": AttrDef(attr_type=AnalogAttr(Range(-1, 1))),
        },
    },
)

# Edge types
StateSelfEdge = EdgeType(name="SelfEdge")
ProdAuxSelfEdge = EdgeType(name="ProdAuxSelfEdge")
ToProdAuxEdge = EdgeType(name="ToProdAux")
FromProdAuxEdge = EdgeType(name="FromProdAux")

lorenz96_spec.add_cdg_types(
    [State, ProdAux, StateSelfEdge, ProdAuxSelfEdge, ToProdAuxEdge, FromProdAuxEdge]
)

# Production rules
self_conn = ProdRule(StateSelfEdge, State, State, SELF, -VAR(SELF) + SELF.F)
prod_self_conn = ProdRule(ProdAuxSelfEdge, ProdAux, ProdAux, SELF, SELF.sign)
to_prod_conn = ProdRule(ToProdAuxEdge, State, ProdAux, DST, VAR(SRC))
from_prod_conn = ProdRule(FromProdAuxEdge, ProdAux, State, DST, VAR(SRC))

lorenz96_spec.add_production_rules(
    [self_conn, prod_self_conn, to_prod_conn, from_prod_conn]
)


def build_lorenz96_sys(
    n_state_var: int, init_F: float = 8, trainable_mgr: Optional[TrainableMgr] = None
):
    lorenz_sys = CDG()
    if trainable_mgr:
        state_vars = [
            State(F=trainable_mgr.new_analog(init_val=init_F))
            for i in range(n_state_var)
        ]
    else:
        state_vars = [State(F=init_F) for i in range(n_state_var)]
    pos_prod_aux = [ProdAux(sign=1) for i in range(n_state_var)]
    neg_prod_aux = [ProdAux(sign=-1) for i in range(n_state_var)]

    # Connect the state variables to the positive product auxiliaries w/ circular index
    for i in range(n_state_var):
        xi = state_vars[i]  # xi
        ppxi = state_vars[(i - 2) % n_state_var]  # x_{i-2}
        pxi = state_vars[(i - 1) % n_state_var]  # x_{i-1}
        nxi = state_vars[(i + 1) % n_state_var]  # x_{i+1}

        pxnx, ppxpx = pos_prod_aux[i], neg_prod_aux[i]

        lorenz_sys.connect(StateSelfEdge(), xi, xi)
        lorenz_sys.connect(ProdAuxSelfEdge(), ppxpx, ppxpx)
        lorenz_sys.connect(ProdAuxSelfEdge(), pxnx, pxnx)
        lorenz_sys.connect(ToProdAuxEdge(), pxi, pxnx)
        lorenz_sys.connect(ToProdAuxEdge(), nxi, pxnx)
        lorenz_sys.connect(ToProdAuxEdge(), ppxi, ppxpx)
        lorenz_sys.connect(ToProdAuxEdge(), pxi, ppxpx)

        lorenz_sys.connect(FromProdAuxEdge(), pxnx, xi)
        lorenz_sys.connect(FromProdAuxEdge(), ppxpx, xi)

    return lorenz_sys, state_vars


if __name__ == "__main__":
    n_state_var = 5

    lorenz_sys, _ = build_lorenz96_sys(n_state_var)

    compiler = ArkCompiler()
    compiler.compile_odeterm(cdg=lorenz_sys, cdg_spec=lorenz96_spec, vectorize=False)
    compiler.print_odeterm()
