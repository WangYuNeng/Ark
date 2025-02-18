"""Compiler for CDG to dynamical system simulation"""

import ast
import copy
import inspect
from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import sympy
import sympy.codegen
from scipy import integrate

from ark.cdg.cdg import CDG, CDGEdge, CDGElement, CDGNode
from ark.reduction import PRODUCT, Reduction
from ark.rewrite import BaseRewriteGen, SympyRewriteGen, VectorizeRewriteGen
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.production_rule import ProdRule, ProdRuleId
from ark.specification.rule_keyword import DST, SRC, TIME, Target, kw_name
from ark.specification.specification import CDGSpec
from ark.util import concat_expr, mk_jnp_call, mk_list, set_ctx


def ddt(name: str, order: int) -> str:
    """return the name of the derivative of the given order"""
    return f"{'ddt_' * order}{name}"


def n_state(name: str) -> str:
    """return the name of the next state"""
    return f"next_{name}"


def rn_attr(name: str, attr: str) -> str:
    """return the variable name of the named attribute"""
    return f"{name}_{attr}"


def switch_attr(name: str) -> str:
    """return the name of the switch variable"""
    return rn_attr(name, "switch")


def mk_var(name: str, to_sympy: bool = False) -> ast.Name:
    """convert name to an ast.Name"""
    if to_sympy:
        return sympy.Symbol(name)
    return ast.Name(name, ctx=ast.Load())


def mk_arg(name: str) -> ast.arg:
    """convert name to an ast.arg"""
    return ast.arg(arg=name)


def mk_list_assign(targets: list[ast.Name], value: ast.expr) -> ast.Assign:
    return ast.Assign(
        [
            set_ctx(
                ast.Tuple(targets),
                ast.Store(),
            )
        ],
        set_ctx(value, ast.Load()),
    )


def parse_expr(val: "int | float | Callable | str") -> ast.expr:
    """parse value to expression"""
    if isinstance(val, int) or isinstance(val, float):
        val_str = str(val)
    elif isinstance(val, Callable):
        val_str = val.__name__
    elif isinstance(val, str):
        val_str = val
    else:
        raise TypeError(f"Unsupported type {type(val)}")
    mod = ast.parse(val_str)
    return mod.body[0]


def apply_gen_rule(
    rule: ProdRule,
    transformer: BaseRewriteGen,
    to_sympy: bool = False,
    from_noise: bool = False,
) -> ast.expr | sympy.Expr:
    """Return the rewritten ast from the given rule"""
    if not from_noise:
        if not to_sympy:
            gen_expr = copy.deepcopy(rule.fn_ast)
        else:
            gen_expr = rule.fn_sympy
    else:
        if not to_sympy:
            gen_expr = copy.deepcopy(rule.noise_ast)
        else:
            gen_expr = rule.noise_sympy
    rr_expr = transformer.visit(gen_expr)
    return rr_expr if not isinstance(rr_expr, ast.Expression) else rr_expr.body


def match_prod_rule(
    rule_dict: dict[ProdRuleId, ProdRule],
    edge: CDGEdge,
    src: CDGNode,
    dst: CDGNode,
    tgt: Target,
) -> ProdRule | None:
    """Find the production rule that matches the given edge, source node,
    destination node, and rule target.

    Args:
        rule_dict (dict[ProdRuleId, ProdRule]): production rule dictionary
        edge (CDGEdge): edge to match
        src (CDGNode): source node to match
        dst (CDGNode): destination node to match
        tgt (Target): target of the production rule (SRC or DST)
    Returns:
        ProdRule: the matched production rule or None if no match is found
    """

    def check_match(
        edge_type: EdgeType, src_nt: NodeType, dst_nt: NodeType
    ) -> ProdRule | None:
        rule_id = ProdRuleId(edge_type, src_nt, dst_nt, tgt)
        if rule_id in rule_dict:
            return rule_dict[rule_id]
        return None

    edge_type: EdgeType = edge.cdg_type
    src_nt: NodeType = src.cdg_type
    dst_nt: NodeType = dst.cdg_type

    match = check_match(edge_type, src_nt, dst_nt)
    if match is not None:
        return match

    et_base = edge_type.base_cdg_types()
    src_nt_base = src_nt.base_cdg_types()
    dst_nt_base = dst_nt.base_cdg_types()

    # Try to match the node type of the target as close as possible
    # then the edge type and then the node type of the other end.
    if tgt == SRC:
        ordered_types = product(src_nt_base, et_base, dst_nt_base)
    else:
        ordered_types = product(dst_nt_base, et_base, src_nt_base)

    for nt0, et, nt1 in ordered_types:
        if tgt == SRC:
            src_nt, dst_nt, edge_type = nt0, nt1, et
        else:
            src_nt, dst_nt, edge_type = nt1, nt0, et
        match = check_match(edge_type, src_nt, dst_nt)
        if match is not None:
            return match

    return None


@dataclass
class ProdRuleCDGInfo:
    """Coordinates and implented edge of a production rule in the CDG

    Each `coordinate` of the connection is represented by the nodes' state variable
    indices. E.g., if we have [da/dt, db/dt, d^2b/dt^2] as the state variables of
    the nodes, a connection between node a and b that targets a will have coordinate
    (0, 1), a connection between node a and b that targets a will have coordinate
    (2, 0). That is, `x` corresponds to the `x`-th state variable of the node and is also
    the target of the differential equation generated by the connection. `y` corresponds
    to the `y`-th state variable which contributes the x-th state variable derivative.

    If the y node is not stateful, the index is -1.
    """

    x: int
    y: int
    edge: CDGEdge


def multi_hot_vectors(
    dim: int, indices: list[list[int]], row_one_hot: bool
) -> jax.Array:
    """Convert indices to multi-hot vectors.

    Args:
        dim (int): dimension of the multi-hot vectors
        indices (list[int]): indices to convert to multi-hot vectors
        row_one_hot (bool): whether to multi-hot vectors as rows or columns
    Returns:
        jax.Array: multi-hot vectors
    """
    if row_one_hot:
        return jnp.array([np.sum(np.eye(dim)[idx], axis=0) for idx in indices])
    return jnp.array([np.sum(np.eye(dim)[idx], axis=0) for idx in indices]).T


def one_hot_vectors(dim: int, indices: list[int], row_one_hot: bool) -> jax.Array:
    """Convert indices to one-hot vectors.

    Args:
        dim (int): dimension of the one-hot vectors
        indices (list[int]): indices to convert to one-hot vectors
        row_one_hot (bool): whether to one-hot vectors as rows or columns
    Returns:
        jax.Array: one-hot vectors
    """
    if row_one_hot:
        return jnp.array(np.eye(dim)[indices])
    return jnp.array(np.eye(dim)[indices].T)


def mk_one_hot_vector_call(dim: int, indices: list[int], row_one_hot: bool) -> ast.Call:
    """Create a call expression for one_hot_vectors function.

    Args:
        dim (int): dimension of the one-hot vectors
        indices (list[int]): indices to convert to one-hot vectors
        row_one_hot (bool): whether to one-hot vectors as rows or columns

    Returns:
        ast.Call: call expression of one_hot_vectors(dim, indices, row_one_hot)
    """
    return set_ctx(
        ast.Call(
            func=mk_var("one_hot_vectors"),
            args=[
                ast.Constant(value=dim),
                ast.List(elts=[ast.Constant(value=idx) for idx in indices]),
                ast.Constant(value=row_one_hot),
            ],
            keywords=[],
        ),
        ast.Load(),
    )
    """Tried the following but it doesn't improve the performance.
    dim_expr = ast.Constant(value=dim)
    indices_expr = ast.List(elts=[ast.Constant(value=idx) for idx in indices])
    jnp_indices = mk_jnp_call(args=[indices_expr], call_fn="array")
    eye_expr = mk_jnp_call(args=[dim_expr], call_fn="eye")
    one_hot_vec_expr = ast.Subscript(value=eye_expr, slice=jnp_indices)
    if not row_one_hot:
        one_hot_vec_expr = ast.Attribute(value=one_hot_vec_expr, attr="T")
    return one_hot_vec_expr
    """


def mk_binop(expr1: ast.expr, op: ast.operator, expr2: ast.expr) -> ast.BinOp:
    """Create a binary operation expression.

    Args:
        expr1 (ast.expr): left operand of the binary operation
        op (ast.operator): binary operation
        expr2 (ast.expr): right operand of the binary operation

    Returns:
        ast.BinOp: binary operation expression
    """
    return ast.BinOp(left=expr1, op=op, right=expr2)


def mk_matmul(expr1: ast.expr, expr2: ast.expr) -> ast.BinOp:
    """Create a matrix multiplication expression.

    Args:
        expr1 (ast.expr): left operand of the matrix multiplication
        expr2 (ast.expr): right operand of the matrix multiplication

    Returns:
        ast.BinOp: matrix multiplication expression
    """
    return mk_binop(expr1, ast.MatMult(), expr2)


def mk_add(expr1: ast.expr, expr2: ast.expr) -> ast.BinOp:
    """Create an addition expression.

    Args:
        expr1 (ast.expr): left operand of the addition
        expr2 (ast.expr): right operand of the addition

    Returns:
        ast.BinOp: addition expression
    """
    return mk_binop(expr1, ast.Add(), expr2)


class ArkCompiler:
    """Ark compiler for CDG to dynamical system simulation

    Args:
    RewriteGen: ast rewrite class
    """

    SIM_SEED = "sim_seed"
    SIM_RAND_EXPT = parse_expr(f"np.random.rand({SIM_SEED})")
    ODE_FN_NAME, ODE_INPUT_VAR = "__ode_fn", "__variables"
    ODE_0TH_ORDER_VAR = "__0th_order_variables"
    ONE_HOT_VAR = "__one_hot_vectors"
    NOISE_FN_NAME = "__noise_fn"
    INIT_STATE, TIME_RANGE = "init_states", "time_range"
    SWITCH_VAL = "switch_vals"
    ATTR_VAL = "attr_vals"
    ARGS = "args"
    FN_ARGS = "fargs"
    KWARGS = "kwargs"
    SIM_SOL = "__sol"
    SOLVE_IVP_EXPR = parse_expr(
        f"{SIM_SOL} = scipy.integrate.solve_ivp({ODE_FN_NAME}, \
                                {TIME_RANGE}, {INIT_STATE}, **{KWARGS})"
    )
    PROG_NAME = "dynamics"

    def __init__(self) -> None:
        self._node_to_state_var = {}
        self._ode_fn_io_names = []
        self._switch_mapping, self._attr_mapping = {}, {}
        self._num_attr_mapping, self._fn_attr_mapping = {}, {}
        self._node_to_0th_order_var = {}
        self._namespace = {
            "np": np,
            "scipy": scipy,
            "scipy.integrate": integrate,
            "jnp": jnp,
            "one_hot_vectors": one_hot_vectors,
        }
        self._prog_ast = None
        self._ode_term_ast = None
        self._gen_rule_dict = {}
        self._verbose = 0
        self._one_hot_var_id = 0

    @property
    def prog_name(self):
        """name of the program to be generated"""
        return self.PROG_NAME

    @property
    def tmp_file_name(self):
        """name of the temporary file to store the generated program"""
        return f"__tmp_{self.prog_name}__.py"

    def print_prog(self):
        """print the compiled program"""
        assert self._prog_ast is not None, "Program is not compiled yet!"
        print(ast.unparse(self._prog_ast))

    def dump_prog(self, file_name: str):
        """dump the compiled program to the given file"""
        assert self._prog_ast is not None, "Program is not compiled yet!"
        with open(file_name, "w") as f:
            f.write(ast.unparse(self._prog_ast))

    def print_odeterm(self):
        """print the compiled odeterm function"""
        assert self._ode_term_ast is not None, "Program is not compiled yet!"
        print(ast.unparse(self._ode_term_ast))

    def dump_odeterm(self, file_name: str):
        """dump the compiled odeterm function to the given file"""
        assert self._ode_term_ast is not None, "Program is not compiled yet!"
        with open(file_name, "w") as f:
            f.write(ast.unparse(self._ode_term_ast))

    def compile(self, cdg: CDG, cdg_spec: CDGSpec, verbose: int = 0):
        """Compile the cdg to a function for dynamical system simulation.

        Args:
            cdg (CDG): The input graph to compile
            cdg_spec (CDGSpec): Specification containing the production rules
            verbose (int): level of status printing, 0 -- no printing,
            1 -- print the compilation progress.

        Returns:
            prog (Callable): The compiled function.
            node_mapping (dict[str, int]): map the name of CDGNode to the corresponding
            index in the state variables. The node_mapping[name] points to the index
            of the 0th order term and the n-th order deravitves (if applicable) index
            is node_mapping[name] + n.
            switch_mapping (dict[str, int]): map the name of CDGEdge to the corresponding
            index in the switch variables.
            attr_mapping (dict[str, dict[str, int]]): map the name of CDGElement to the
            corresponding index in the attribute variables. The attr_mapping[name][attr]
            points to the index of the attribute.
        """

        self._verbose = verbose

        (
            self._node_to_state_var,
            self._ode_fn_io_names,
            self._switch_mapping,
            self._attr_mapping,
        ) = self._collect_ode_fn_io(cdg)

        attr_var_def = self._compile_attribute_var(
            self._switch_mapping, self._attr_mapping
        )
        ode_fn, _ = self._compile_ode_fn(cdg, cdg_spec, self._ode_fn_io_names)
        solv_ivp_stmts = self._compile_solve_ivp()

        stmts = attr_var_def + [ode_fn] + solv_ivp_stmts
        top_stmt = self._compile_top_stmt(stmts)

        module = ast.Module([top_stmt], type_ignores=[])
        module = ast.fix_missing_locations(module)
        self._prog_ast = module

        if verbose:
            print("Compiling the program...")
        code = compile(source=module, filename=self.tmp_file_name, mode="exec")
        exec(code, self._namespace)

        if verbose:
            print("Compilation finished")

        return (
            self._namespace[self.prog_name],
            self._node_to_state_var,
            self._switch_mapping,
            self._attr_mapping,
        )

    def compile_odeterm(
        self, cdg: CDG, cdg_spec: CDGSpec, vectorize: bool = False, verbose: int = 0
    ) -> tuple[
        tuple[Callable, Callable],
        dict[str, int],
        dict[str, int],
        dict[str, dict[str, int]],
        dict[str, dict[str, int]],
    ]:
        """Compile the cdg to an executable function representing the derivative term.

        The function will in the form of `__ode_fn(time, __variables, args, fargs)`
        where `args` is the list of arguments mapped to
        the float/int attribute values and switch values, and `fargs` is the list of
        function arguments mapped to the Callable attribute values.

        The separation of the two arguments is for compatibility to Diffrax
        1. The function arguments can't be jitted. Need to be treat as static value
        separately.
        2. The ode term in Diffrax is in the form of f(t, y, args).

        Args:
            cdg (CDG): The input graph to compile
            cdg_spec (CDGSpec): Specification containing the production rules
            vectorize (bool): whether to compile the function in a vectorized form
            verbose (int): level of status printing, 0 -- no printing,
            1 -- print the compilation progress.

        Returns:
            ode_term and noise_term ((Callable, Callable)): The compiled function.
            node_mapping (dict[str, int]): map the name of CDGNode to the corresponding
            index in the state variables. The node_mapping[name] points to the index
            of the 0th order term and the n-th order deravitves (if applicable) index
            is node_mapping[name] + n.
            switch_mapping (dict[str, int]): map the name of CDGEdge to the corresponding
            index in the switch variables.
            num_attr_mapping (dict[str, dict[str, int]]): map the name of CDGElement to the
            corresponding index in the numerical attribute variables.
            The num_attr_mapping[name][attr] points to the index of the attribute.
            fn_attr_mapping (dict[str, dict[str, int]]): map the name of CDGElement to the
            corresponding index in the function attribute variables.
        """

        self._verbose = verbose

        stmts = []

        (
            self._node_to_state_var,
            self._ode_fn_io_names,
            self._switch_mapping,
            self._attr_mapping,
        ) = self._collect_ode_fn_io(cdg, separate_fn_attr=True)

        # Count the numerical value attributes index from the switch mapping
        # Isolate the function attributes mapping
        num_attr_mapping, num_attr_idx = {}, len(self._switch_mapping)
        fn_attr_mapping, fn_attr_idx = {}, 0
        for ele_name, attrs in self._attr_mapping.items():
            num_attr_mapping[ele_name], fn_attr_mapping[ele_name] = {}, {}
            for attr_name, is_num in attrs.items():
                if not is_num:
                    fn_attr_mapping[ele_name][attr_name] = fn_attr_idx
                    fn_attr_idx += 1
                else:
                    num_attr_mapping[ele_name][attr_name] = num_attr_idx
                    num_attr_idx += 1
        self._num_attr_mapping, self._fn_attr_mapping = (
            num_attr_mapping,
            fn_attr_mapping,
        )

        # Generate the variable names of the switches and attributes
        args_names = [
            None
            for _ in range(
                len(self._switch_mapping)
                + sum(len(attrs) for attrs in num_attr_mapping.values())
            )
        ]
        fn_args_names = [
            None for _ in range(sum(len(attrs) for attrs in fn_attr_mapping.values()))
        ]

        for name, idx in self._switch_mapping.items():
            args_names[idx] = mk_var(switch_attr(name))
        for ele_name, attrs in num_attr_mapping.items():
            for attr_name, idx in attrs.items():
                args_names[idx] = mk_var(rn_attr(ele_name, attr_name))
        for ele_name, attrs in fn_attr_mapping.items():
            for attr_name, idx in attrs.items():
                fn_args_names[idx] = mk_var(rn_attr(ele_name, attr_name))

        # Generate the Assignment statements
        if args_names and not vectorize:
            stmts.append(
                mk_list_assign(
                    targets=args_names,
                    value=mk_var(self.ARGS),
                )
            )
        if fn_args_names:
            stmts.append(
                mk_list_assign(
                    targets=fn_args_names,
                    value=mk_var(self.FN_ARGS),
                )
            )
        _, ode_stmts = self._compile_ode_fn(
            cdg, cdg_spec, self._ode_fn_io_names, return_jax=True, vectorize=vectorize
        )
        _, noise_ode_stmts = self._compile_ode_fn(
            cdg,
            cdg_spec,
            self._ode_fn_io_names,
            return_jax=True,
            noise_ode=True,
            vectorize=vectorize,
        )
        for fn_name, ode_stmt in zip(
            [self.ODE_FN_NAME, self.NOISE_FN_NAME], [ode_stmts, noise_ode_stmts]
        ):
            fn_stmts = [stmt for stmt in stmts]

            fn_stmts.extend(ode_stmt)
            arguments = ast.arguments(
                posonlyargs=[],
                args=[
                    mk_arg(kw_name(TIME)),
                    mk_arg(self.ODE_INPUT_VAR),
                    mk_arg(self.ARGS),
                    mk_arg(self.FN_ARGS),
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            )
            func_def = ast.FunctionDef(
                name=fn_name, args=arguments, body=fn_stmts, decorator_list=[]
            )
            module = ast.Module([func_def], type_ignores=[])
            module = ast.fix_missing_locations(module)
            self._ode_term_ast = module

            code = compile(source=module, filename=self.tmp_file_name, mode="exec")
            exec(code, self._namespace)

        return (
            (self._namespace[self.ODE_FN_NAME], self._namespace[self.NOISE_FN_NAME]),
            self._node_to_state_var,
            self._switch_mapping,
            self._num_attr_mapping,
            self._fn_attr_mapping,
        )

    def compile_sympy_diffeqs(
        self, cdg: CDG, cdg_spec: CDGSpec, noise_ode: bool = False
    ) -> list[sympy.Equality]:
        """Compile the cdg to the ode function into a list of sympy equations
        describing the dynamical system.

        Args:
            cdg (CDG): The input CDG
            cdg_spec (CDGSpec): Specification
            noise_ode (bool): whether the ode function is for noise term.

        Returns:
            equations (list[sympy.Expr]): list of sympy equations
        """

        node: CDGNode
        src: CDGNode
        dst: CDGNode
        edge: CDGEdge
        gen_rule: ProdRule
        reduction: Reduction

        rule_dict = cdg_spec.prod_rule_dict()
        rewrite = SympyRewriteGen()
        rewrite.set_attr_rn_fn(rn_attr)

        equations = []

        # Generate the ddt_var = f(var) statements
        for order in range(cdg.ds_order + 1):
            nodes = cdg.nodes_in_order(order)

            # 0th order: var = f(vars)
            # 1th order: ddt_var = f(vars)
            # 2th order: ddt_var = ddt_var_cur, ddt_ddt_var = f(vars)
            # ...
            for node in nodes:
                for sub_order in range(1, order):
                    vname = n_state(ddt(node.name, order=sub_order))
                    if noise_ode:
                        # For higher order noise terms, the derivative is 0
                        equations.append(
                            sympy.codegen.Assignment(
                                lhs=mk_var(vname, to_sympy=True),
                                rhs=sympy.Float(0),
                            )
                        )
                    else:
                        cur_vname = ddt(node.name, order=sub_order)
                        equations.append(
                            sympy.codegen.Assignment(
                                lhs=mk_var(vname, to_sympy=True),
                                rhs=mk_var(cur_vname, to_sympy=True),
                            )
                        )
                vname = ddt(node.name, order=order)
                if order != 0:
                    vname = n_state(vname)
                reduction = node.reduction
                rhs = []

                # Indicate whether all productive edges (edge produces expression)
                # are switchable for this node for the PRODUCT reduction with all switchable edge case.
                product_and_all_edge_switchable = reduction == PRODUCT
                switchable_edge_names = []

                for edge in node.edges:
                    src, dst = edge.src, edge.dst
                    gen_rule = match_prod_rule(
                        rule_dict=rule_dict,
                        edge=edge,
                        src=src,
                        dst=dst,
                        tgt=node.which_tgt(edge),
                    )
                    if gen_rule is not None:
                        rewrite.mapping = gen_rule.get_rewrite_mapping(edge=edge)
                        rhs_expr = apply_gen_rule(
                            rule=gen_rule,
                            transformer=rewrite,
                            to_sympy=True,
                            from_noise=noise_ode,
                        )
                        if edge.switchable:
                            rhs_expr = reduction.sympy_switch(
                                rhs_expr, mk_var(switch_attr(edge.name), to_sympy=True)
                            )
                            switchable_edge_names.append(edge.name)
                        else:
                            product_and_all_edge_switchable = False
                        rhs.append(rhs_expr)
                if rhs:
                    # Edge case -- when a node has all swithable edges with a PRODUCT reduction,
                    # current implementation will resolve to 1 when all switches are off but it should be 0.
                    if product_and_all_edge_switchable:
                        # Guard the edge case with a (1-\Pi(1-switches)) term
                        switch_guard_expr = 1 - concat_expr(
                            [
                                1 - mk_var(switch_attr(name), to_sympy=True)
                                for name in switchable_edge_names
                            ],
                            sympy.Mul,
                        )
                        rhs.append(switch_guard_expr)
                    equations.append(
                        sympy.codegen.Assignment(
                            lhs=mk_var(vname, to_sympy=True),
                            rhs=sympy.simplify(concat_expr(rhs, reduction.sympy_op)),
                        )
                    )
        return equations

    def compile_vectorized_diffeqs(
        self, cdg: CDG, cdg_spec: CDGSpec, noise_ode: bool = False
    ) -> ast.Assign:
        """Compile the cdg to the ode function into a vectorized assignment statement
        dx/dt = ... describing the dynamical system.

        Args:
            cdg (CDG): The input CDG
            cdg_spec (CDGSpec): Specification
            noise_ode (bool): whether the ode function is for noise term.

        Returns:
            equations (ast.Assign): system equation
        """

        def find_element_attr_coord(cdg_element: CDGElement):
            """Enumerate the attributes of the CDGElement"""
            ele_attr_to_coord = {}
            for attr_name in cdg_element.attrs.keys():
                # If the attribute is a numerical attribute, record the index
                # of it in `args`
                if attr_name in self._num_attr_mapping[cdg_element.name]:
                    idx = self._num_attr_mapping[cdg_element.name][attr_name]
                    ele_attr_to_coord[attr_name] = [idx]
            return ele_attr_to_coord

        # Collect the index of 0th-order nodes
        self._node_to_0th_order_var = {
            node.name: idx for idx, node in enumerate(cdg.nodes_in_order(0))
        }
        rule_dict = cdg_spec.prod_rule_dict()
        rewrite = VectorizeRewriteGen()
        rewrite.set_attr_rn_fn(rn_attr)
        n_state_var = len(self._node_to_state_var)
        n_0th_order_var = len(self._node_to_0th_order_var)
        n_args = len(self._switch_mapping) + sum(
            len(attrs) for attrs in self._num_attr_mapping.values()
        )

        # Collect the production rule categorized by
        # {redction_types} x (0th-order, higher-order)
        reduction_types = set([node.reduction for node in cdg.nodes])
        rhs_exprs: list[dict[Reduction, list[ast.Expr]]] = [
            {rt: [] for rt in reduction_types} for _ in range(2)
        ]

        # FIXME: Add assignment for all the lower-order terms
        # for high order state variables
        if cdg.ds_order > 1:
            raise NotImplementedError(
                "Vectorized mode does not support high order state variables yet."
            )

        prodrule_to_info = self._get_prodrule_info(cdg, cdg_spec)
        for prod_rule_id, info_list in prodrule_to_info.items():
            # Collect src/dst/self variable and attribute mapping and
            # edge attribute mapping
            # variable -> Matrix @ x
            # attribute -> Matrix @ args
            Tx_coord, Ty_coord = [], []

            # [edge_attr_to_coord, src_attr_to_coord, dst_attr_to_coord]
            attr_to_coord_list = [None, None, None]

            # Store the location and of the switch in the current vector
            # and the index of the switch in the args
            switch_vec_idx_args_idx = []

            # Some variables might be confusing:
            # Tx_coord/Ty_coord: the coordinates of the target/used state variables
            # src/dst: the source/destination node of the edge
            # It could be source is the target or destination is the target

            for i, info in enumerate(info_list):
                x, y, edge, switchable = info.x, info.y, info.edge, info.edge.switchable
                Tx_coord.append(x)
                if y != -1:
                    Ty_coord.append(y)

                if switchable:
                    switch_vec_idx_args_idx.append([i, self._switch_mapping[edge.name]])

                # Enumerate all the attributes of the edge and nodes and record the
                # attribute to index (in `args`) mapping
                for i, ele in enumerate([edge, edge.src, edge.dst]):
                    if not attr_to_coord_list[i]:
                        attr_to_coord_list[i] = find_element_attr_coord(ele)
                    else:
                        attr_to_coord_list: list[dict[str, list[int]]]
                        ele_a2c = find_element_attr_coord(ele)
                        if attr_to_coord_list[i].keys() != ele_a2c.keys():
                            raise RuntimeError(
                                "Element attributes mismatch, previous nodes have"
                                f"{attr_to_coord_list[i].keys()} and current node have {ele_a2c.keys()}"
                            )
                        # Merge the attribute mapping
                        for k, v in ele_a2c.items():
                            attr_to_coord_list[i][k].extend(v)

            # Check if the src/dst are stateful
            src_stateful = info.edge.src.order > 0
            dst_stateful = info.edge.dst.order > 0
            node_to_var = (
                self._node_to_state_var if src_stateful else self._node_to_0th_order_var
            )

            # If x is the src, SRC should map to one-hot vectors of x-th state variable,
            # DST should map to one-hot vectors of y-th state variable, and vice versa.
            x_is_src = info.x == node_to_var[info.edge.src.name]
            (src_coord, dst_coord) = (
                (Tx_coord, Ty_coord) if x_is_src else (Ty_coord, Tx_coord)
            )

            # Set the dimension and associated variable based on whether
            # they are vectorized state variableles or 0th order variables
            (src_dim, src_var) = (
                (n_state_var, mk_var(self.ODE_INPUT_VAR))
                if src_stateful
                else (n_0th_order_var, mk_var(self.ODE_0TH_ORDER_VAR))
            )
            (dst_dim, dst_var) = (
                (n_state_var, mk_var(self.ODE_INPUT_VAR))
                if dst_stateful
                else (n_0th_order_var, mk_var(self.ODE_0TH_ORDER_VAR))
            )
            x_dim = src_dim if x_is_src else dst_dim

            # FIXME: Function attribute cannot be vectorized. In reality,
            # the function attribute for the same production rule are usually
            # the same though. Need to figure out how to handle this. Currently,
            # we simply use the last function attribute of a to represent all the
            fn_name_mapping = rule_dict[prod_rule_id].get_rewrite_mapping(info.edge)

            # Convert the collected coordinates to ast of one-hot vectors and
            # multiplied by the state variables or arguments
            src_mat_ast = mk_matmul(
                self._mk_one_hot_jnp_array_variable(
                    dim=src_dim, indices=src_coord, row_one_hot=True
                ),
                src_var,
            )
            dst_mat_ast = mk_matmul(
                self._mk_one_hot_jnp_array_variable(
                    dim=dst_dim, indices=dst_coord, row_one_hot=True
                ),
                dst_var,
            )

            # edge_attr_to_mat_ast, src_attr_to_mat_ast, dst_attr_to_mat_ast
            attr_to_mat_ast_list = [{}, {}, {}]
            for i, a2c in enumerate(attr_to_coord_list):
                for attr, coord in a2c.items():
                    attr_to_mat_ast_list[i][attr] = mk_matmul(
                        self._mk_one_hot_jnp_array_variable(
                            dim=n_args, indices=coord, row_one_hot=True
                        ),
                        mk_var(self.ARGS),
                    )

            # Set up the mapping for the production rule rewrite
            keyword_name_mapping = rule_dict[prod_rule_id].get_rewrite_mapping(
                info.edge,
                src_val=src_mat_ast,
                dst_val=dst_mat_ast,
                time_val=ast.Name(kw_name(TIME)),
            )
            rewrite.set_mappings(
                fn_mapping=fn_name_mapping,
                keyword_name_mapping=keyword_name_mapping,
                edge_attr_mapping=attr_to_mat_ast_list[0],
                src_attr_mapping=attr_to_mat_ast_list[1],
                dst_attr_mapping=attr_to_mat_ast_list[2],
            )
            rewritten_prod_rule_expr = apply_gen_rule(
                rule=rule_dict[prod_rule_id],
                transformer=rewrite,
                to_sympy=False,
                from_noise=noise_ode,
            )

            # Multiply by an unit column matrix in case the production rule
            # is a scalar
            n_produced_expr = len(info_list)
            rewritten_prod_rule_expr = ast.BinOp(
                left=rewritten_prod_rule_expr,
                op=ast.Mult(),
                right=mk_jnp_call(
                    args=[ast.Constant(value=n_produced_expr)], call_fn="ones"
                ),
            )

            if switch_vec_idx_args_idx:
                # switch expression is Tvecidx @ Tswitchargsidx @ args + col vec w/
                # 0 at Tvecidx and 1 otherwise (non-switchables are considered
                # as always-on switches)
                vec_idx, sw_args_idx = [i[0] for i in switch_vec_idx_args_idx], [
                    i[1] for i in switch_vec_idx_args_idx
                ]
                # Tswitchargs @ args
                # [n_node_switch, n_args] @ [n_args, 1] = [n_node_switch, 1]
                switch_args_expr = mk_matmul(
                    self._mk_one_hot_jnp_array_variable(
                        dim=n_args,
                        indices=sw_args_idx,
                        row_one_hot=True,
                    ),
                    mk_var(self.ARGS),
                )
                # Tvec [n_produced_expr, n_node_switch]
                switch_Tvec_expr = self._mk_one_hot_jnp_array_variable(
                    dim=n_produced_expr,
                    indices=vec_idx,
                    row_one_hot=False,
                )

                col_vec = 1 - np.sum(np.eye(n_produced_expr)[vec_idx], axis=0)
                col_vec_list_expr = mk_list(
                    [ast.Constant(value=int(i)) for i in col_vec]
                )
                switch_col_vec_expr = mk_jnp_call(
                    args=[col_vec_list_expr],
                    call_fn="array",
                )

                # switch expression
                switch_expr = mk_add(
                    mk_matmul(switch_Tvec_expr, switch_args_expr),
                    switch_col_vec_expr,
                )
                tgt_node = edge.src if x_is_src else edge.dst
                if tgt_node.reduction == PRODUCT:
                    # FIXME: Add the leading term for special case of PRODUCT reduction
                    raise NotImplementedError(
                        "PRODUCT reduction with switchable edges is not supported in the vectorized mode yet. "
                        "Please use the non-vectorized mode."
                    )
                rewritten_prod_rule_expr = ast.BinOp(
                    left=rewritten_prod_rule_expr,
                    op=tgt_node.reduction.ast_switch,
                    right=switch_expr,
                )

            Txmat_T_ast = self._mk_one_hot_jnp_array_variable(
                dim=x_dim,
                indices=Tx_coord,
                row_one_hot=False,
            )
            rhs_expr = mk_matmul(Txmat_T_ast, rewritten_prod_rule_expr)

            tgt_reduction = (
                info.edge.src.reduction if x_is_src else info.edge.dst.reduction
            )
            tgt_is_stateful = info.edge.src.order if x_is_src else info.edge.dst.order
            assert (
                tgt_reduction in rhs_exprs[tgt_is_stateful]
            ), "Reduction type mismatch"
            f"Expected reduction in {rhs_exprs[tgt_is_stateful].keys()} but got {tgt_reduction}."
            rhs_exprs[tgt_is_stateful][tgt_reduction].append(rhs_expr)

        # Generate the assignment statement
        stmts = []

        # Sum up the rhs expressions for the 0th order and higher order terms separately
        for is_stateful, lhs in zip(
            [False, True],
            [self.ODE_0TH_ORDER_VAR, ddt(self.ODE_INPUT_VAR, order=1)],
        ):
            stmts_same_order = []
            for reduction, expr_list in rhs_exprs[is_stateful].items():
                if expr_list:
                    stmts_same_order.append(concat_expr(expr_list, reduction.ast_op))

            if stmts_same_order:
                # Sum up the rhs expressions
                stmts.append(
                    ast.Assign(
                        targets=[set_ctx(mk_var(lhs), ast.Store())],
                        value=set_ctx(
                            concat_expr(stmts_same_order, ast.Add()), ast.Load()
                        ),
                        type_ignores=[],
                    )
                )

        return stmts

    def _compile_user_def_fn(self, funcs: list[Callable]) -> list[ast.FunctionDef]:
        """Compile user defined functions to ast.FunctionDef"""
        return [ast.parse(inspect.getsource(fn)).body[0] for fn in funcs]

    def _compile_attribute_var(
        self,
        switch_mapping: dict[str, int],
        attr_mapping: dict[str, dict[str, int]],
    ) -> list[ast.Assign]:
        """Compile the attributes of nodes and edges to variables"""

        stmts = []

        # Map the input vector to the state variables
        if switch_mapping:
            edge_names = [None for _ in range(len(switch_mapping))]
            for name, idx in switch_mapping.items():
                edge_names[idx] = name
            stmts.append(
                mk_list_assign(
                    targets=[
                        mk_var(switch_attr(edge_name)) for edge_name in edge_names
                    ],
                    value=mk_var(self.SWITCH_VAL),
                )
            )

        if attr_mapping:
            attr_names = [
                None for _ in range(sum(len(attrs) for attrs in attr_mapping.values()))
            ]
            for ele_name, attrs in attr_mapping.items():
                for attr_name, idx in attrs.items():
                    attr_names[idx] = rn_attr(ele_name, attr_name)
            stmts.append(
                mk_list_assign(
                    targets=[mk_var(attr_name) for attr_name in attr_names],
                    value=mk_var(self.ATTR_VAL),
                )
            )
        return stmts

    def _collect_ode_fn_io(
        self, cdg: CDG, separate_fn_attr: bool = False
    ) -> tuple[dict[str, int], list[str], dict[str, int], dict[str, dict[str, int]]]:
        """Collect the input/output node mapping and var names of the ode function

        Args:
            cdg (CDG): The input CDG
            separate_fn_attr (bool): Separate the int/float attributes with function
            attributes.

        Returns:
            node_to_state_var (dict[str, int]): map name ofCDGNode to the corresponding
            index in the state variables. The `node_to_state_var[name]` points to the
            index of the 0th order term and the n-th order deravitves (if applicable)
            index is `node_to_state_var[name] + n`.
            ode_input_return_names (list): names of the state variables of the compiled
            ode function for scipy.integrate.solve_ivp simulation.
            switch_mapping (dict[str, int]): map the name of CDGEdge to the corresponding
            index in the switch variables.
            attr_mapping (dict[str, dict[str, int]]): map the name of CDGElement to the
            corresponding index in the attribute variables. The `attr_mapping[name][attr]`
            points to the index of the attribute if `separate_fn_attr` is false. Otherwise,
            it denote whether the attribute is a float/int attribute or not.
        """

        # Go through the nodes to collect the state variables
        # n-th order variable has n-1 state variables
        node_to_state_var = {}
        ode_input_return_names = []
        for node_order in range(1, cdg.ds_order + 1):
            for node in cdg.nodes_in_order(node_order):
                node_to_state_var[node.name] = len(ode_input_return_names)
                for order in range(node_order):
                    ode_input_return_names.append(ddt(node.name, order=order))

        # Go through the edges to collect the switch variables
        switch_mapping = {edge.name: i for i, edge in enumerate(cdg.switches)}
        attr_mapping, attr_idx = {}, 0
        for ele in cdg.nodes + cdg.edges:
            attr_mapping[ele.name] = {}
            for attr_name, attr in ele.attrs.items():
                if separate_fn_attr:
                    if isinstance(attr, Callable) or isinstance(attr, partial):
                        attr_mapping[ele.name][attr_name] = False
                    else:
                        attr_mapping[ele.name][attr_name] = True
                else:
                    attr_mapping[ele.name][attr_name] = attr_idx
                    attr_idx += 1

        return (
            node_to_state_var,
            ode_input_return_names,
            switch_mapping,
            attr_mapping,
        )

    def _compile_ode_fn(
        self,
        cdg: CDG,
        cdg_spec: CDGSpec,
        io_names: list[str],
        return_jax: bool = False,
        noise_ode: bool = False,
        vectorize: bool = False,
    ) -> tuple[ast.FunctionDef, list[ast.stmt]]:
        """Compile the cdg to the ode function for scipy.integrate.solve_ivp simulation

        The compiled function will have the following signature:
        ```
        def __ode_fn(t, y):
            var1, var2, ... = y
            return [ddt_var1, ddt_var2, ...]
        ```

        Args:
            cdg (CDG): The input CDG
            cdg_spec (CDGSpec): Specification
            io_names (list[str]): names of the state variables `var1, var2, ...`
            return_jax (bool): whether the return value is a list or a jax array.
            noise_ode (bool): whether the ode function is for noise term.
            vectorize (bool): whether to compile the function in a vectorized form
        """

        stmts = []
        input_vec = self.ODE_INPUT_VAR
        ode_fn_name = self.ODE_FN_NAME
        input_return_names = io_names

        if not vectorize:
            # Unpack the state variables
            stmts.append(
                mk_list_assign(
                    targets=[mk_var(name) for name in input_return_names],
                    value=set_ctx(mk_var(input_vec), ast.Load()),
                )
            )
            # Generate the ddt_var = f(var) statements
            equations = self.compile_sympy_diffeqs(cdg, cdg_spec, noise_ode)

            # Convert sympy equations to ast statements
            for eq in equations:
                eq_str = sympy.pycode(eq)
                stmts.append(ast.parse(eq_str).body[0])

            # Return statement of the ode function
            if not return_jax:
                stmts.append(
                    set_ctx(
                        ast.Return(
                            ast.List(
                                [
                                    mk_var(n_state(ddt(name, order=1)))
                                    for name in input_return_names
                                ]
                            )
                        ),
                        ast.Load(),
                    )
                )
            else:
                stmts.append(
                    set_ctx(
                        ast.Return(
                            ast.Call(
                                ast.Attribute(
                                    value=mk_var("jnp"),
                                    attr="array",
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.List(
                                        [
                                            mk_var(n_state(ddt(name, order=1)))
                                            for name in input_return_names
                                        ]
                                    )
                                ],
                                keywords=[],
                            )
                        ),
                        ast.Load(),
                    )
                )

        else:
            dxdt_assignment_stmts = self.compile_vectorized_diffeqs(
                cdg, cdg_spec, noise_ode
            )
            return_stmts = set_ctx(
                ast.Return(mk_var(ddt(self.ODE_INPUT_VAR, order=1))), ast.Load()
            )
            stmts.extend(dxdt_assignment_stmts + [return_stmts])

        arguments = ast.arguments(
            posonlyargs=[],
            args=[mk_arg(kw_name(TIME)), mk_arg(input_vec)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )
        return ast.FunctionDef(ode_fn_name, arguments, stmts, decorator_list=[]), stmts

    def _compile_solve_ivp(self) -> tuple[ast.Assign, ast.Return]:
        stmts = [
            self.SIM_RAND_EXPT,
            self.SOLVE_IVP_EXPR,
            set_ctx(ast.Return(mk_var(self.SIM_SOL)), ast.Load()),
        ]
        return stmts

    def _compile_top_stmt(self, stmts) -> ast.FunctionDef:
        """Make a top level statement"""
        args = [
            mk_arg(self.TIME_RANGE),
            mk_arg(self.INIT_STATE),
            mk_arg(self.SWITCH_VAL),
            mk_arg(self.ATTR_VAL),
            mk_arg(self.SIM_SEED),
        ]
        kwarg = mk_arg(self.KWARGS)
        defaults = [ast.Constant(None), ast.Constant(0), ast.Constant(0)]
        return ast.FunctionDef(
            name=self.prog_name,
            args=ast.arguments(
                posonlyargs=[],
                args=args,
                kwonlyargs=[],
                kwarg=kwarg,
                kw_defaults=[],
                defaults=defaults,
            ),
            body=stmts,
            decorator_list=[],
        )

    def _compile_top_stmt_odeterm(self, stmts: list) -> ast.FunctionDef:
        """Top level statement for ode term

        Args:
            stmts (list): list of statements

        Returns:
            ast.FunctionDef: top level statement
        """

        args = [
            mk_arg(self.TIME_RANGE),
            mk_arg(self.INIT_STATE),
            mk_arg(self.SWITCH_VAL),
            mk_arg(self.ATTR_VAL),
            mk_arg(self.SIM_SEED),
            mk_arg(self.FN_ARGS),
        ]
        kwarg = mk_arg(self.KWARGS)
        defaults = [ast.Constant(None), ast.Constant(0), ast.Constant(0)]
        return ast.FunctionDef(
            name=self.prog_name,
            args=ast.arguments(
                posonlyargs=[],
                args=args,
                kwonlyargs=[],
                kwarg=kwarg,
                kw_defaults=[],
                defaults=defaults,
            ),
            body=stmts,
            decorator_list=[],
        )

    def _get_prodrule_info(
        self, cdg: CDG, cdg_spec: CDGSpec
    ) -> dict[ProdRuleId, list[ProdRuleCDGInfo]]:
        """Return coordinate information of production rule in the graph.

        Each `connection` is identified by the production rule id, i.e., the edge type,
        source node type, destination node type, and the target of the production rule.

        Returns:
            dict[ProdRuleId, list[ProdRuleCDGInfo]]: the coordinates of connections
        """

        rule_dict = cdg_spec.prod_rule_dict()
        prod_id_to_info = {}

        # Iterate all stateful nodes and their edges
        for node in cdg.nodes:
            for edge in node.edges:
                # Find the production rule that matches the edge
                src, dst = edge.src, edge.dst
                gen_rule = match_prod_rule(
                    rule_dict=rule_dict,
                    edge=edge,
                    src=src,
                    dst=dst,
                    tgt=node.which_tgt(edge),
                )
                if gen_rule is not None:
                    if node.which_tgt(edge) == SRC:
                        x_node, y_node = src, dst
                    else:
                        x_node, y_node = dst, src
                    # Prod rule applies to the highest order derivative
                    x = (
                        self._node_to_state_var[x_node.name] + x_node.order - 1
                        if x_node.order > 0
                        else self._node_to_0th_order_var[x_node.name]
                    )
                    # Prod rule uses the lower order derivative
                    y = (
                        self._node_to_state_var[y_node.name]
                        if y_node.order > 0
                        else self._node_to_0th_order_var[y_node.name]
                    )
                    if gen_rule.identifier not in prod_id_to_info:
                        prod_id_to_info[gen_rule.identifier] = []
                    prod_id_to_info[gen_rule.identifier].append(
                        ProdRuleCDGInfo(x, y, edge)
                    )

        return prod_id_to_info

    def _mk_one_hot_jnp_array_variable(
        self, indices: list[int], dim: int, row_one_hot: bool
    ) -> ast.Name:
        """Initialize a jax array variable of one-hot vectors and return the variable name

        The name will be f"{self.ONE_HOT_VAR}_{self._one_hot_var_id}".

        Args:
            indices (list[int]): indices to set to 1
            dim (int): dimension of the one-hot vector
            row_one_hot (bool): whether the one-hot vector is row-wise or column-wise

        Returns:
            ast.Name: Namee of the one-hot-jax array
        """
        one_hot_arr = one_hot_vectors(dim, indices, row_one_hot)
        var_name = f"{self.ONE_HOT_VAR}_{self._one_hot_var_id}"
        self._one_hot_var_id += 1
        self._namespace[var_name] = one_hot_arr
        return mk_var(var_name)
