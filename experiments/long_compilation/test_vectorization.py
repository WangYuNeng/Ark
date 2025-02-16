"""Experiment to test how much vectorization improvs compilation time.

Simplified graph model -- ode_fn only depends on the edge type and 
the arguments only takes the node state.
"""

import ast
from typing import Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from test_long_compile import test_backward_pass, test_forward_pass

from ark.compiler import ddt, mk_arg, mk_list_assign, mk_var, n_state, set_ctx
from ark.optimization.base_module import TimeInfo

# Functions to be called in ode_fn
edge_fn = [
    None,
    lambda x, xp, y, yp: y + xp + jnp.sin(x + yp),
    lambda x, xp, y, yp: jnp.cos(x + xp + y + yp),
]
N_EDGE_TYPE = len(edge_fn)

ODE_FN_NAME = "ode_fn"
TIME = "t"
ODE_INPUT_VAR = "x"
ODE_ARGS = "args"
ODE_FN_ARGS = "fn_args"


# Create random adjacency matrix with size N
def rand_adjacency_matrix(size: int) -> np.ndarray:
    return np.random.randint(0, N_EDGE_TYPE, (size, size))


def adj_mat_to_ode_stmts(adj_mat: np.ndarray, printing: bool = False) -> Callable:
    """Convert adjacency matrix to ODEs.

    ```
    def __ode_fn(t, y, args, fn_args):
        var1, var2, ... = y
        return [ddt_var1, ddt_var2, ...]
    ```
    """

    stmts = []
    input_return_names = [f"x{i}" for i in range(adj_mat.shape[0])]
    args_name = [f"xp{i}" for i in range(adj_mat.shape[0])]
    fn_args_names = [f"fn{i}" for i in range(N_EDGE_TYPE)]

    # x0, x1, ... = x
    stmts.append(
        mk_list_assign(
            targets=[mk_var(name) for name in input_return_names],
            value=set_ctx(mk_var(ODE_INPUT_VAR), ast.Load()),
        )
    )

    # xp0, xp1, ... = args
    stmts.append(
        mk_list_assign(
            targets=[mk_var(name) for name in args_name],
            value=set_ctx(mk_var(ODE_ARGS), ast.Load()),
        )
    )

    # fn0, fn1, ... = fn_args
    stmts.append(
        mk_list_assign(
            targets=[mk_var(name) for name in fn_args_names],
            value=set_ctx(mk_var(ODE_FN_ARGS), ast.Load()),
        )
    )

    # Generate the ddt_var = f(var) statements
    for i, row in enumerate(adj_mat):
        # ddt_xi = sum_j(fn_args[adj_mat[i][j]](xi, xj))
        sub_stmts = []
        for j, edge_type in enumerate(row):
            # fn_args[adj_mat[i][j]](xi, xj)
            if edge_type == 0:
                continue
            sub_stmts.append(
                ast.Call(
                    func=mk_var(fn_args_names[edge_type]),
                    args=[
                        mk_var(input_return_names[i]),
                        mk_var(args_name[i]),
                        mk_var(input_return_names[j]),
                        mk_var(args_name[j]),
                    ],
                    keywords=[],
                )
            )
        if sub_stmts:
            stmts.append(
                set_ctx(
                    ast.Assign(
                        targets=[mk_var(n_state(ddt(input_return_names[i], order=1)))],
                        value=concat_call_expr(sub_stmts, ast.Add()),
                    ),
                    ast.Store(),
                )
            )
            for call in sub_stmts:
                set_ctx(call, ast.Load())
        else:
            # If there is no edge, ddt_xi = 0
            stmts.append(
                set_ctx(
                    ast.Assign(
                        targets=[mk_var(n_state(ddt(input_return_names[i], order=1)))],
                        value=ast.Constant(value=0),
                    ),
                    ast.Store(),
                )
            )

    # Return statement of the ode function
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

    return wrap_ode_fn(stmts, printing=printing)


def one_hot_vectors(dim: int, indices: list[int], row_one_hot: bool) -> jax.Array:
    """Convert indices to one-hot vectors."""
    if row_one_hot:
        return jnp.array(np.eye(dim)[indices])
    return jnp.array(np.eye(dim)[indices].T)


def adj_mat_to_ode_stmts_vectorized(
    adj_mat: np.ndarray, printing: bool = False
) -> Callable:

    def mk_one_hot_vector_call(
        dim: int, indices: list[int], row_one_hot: bool
    ) -> ast.Call:
        """Create a call expression for one_hot_vectors function."""
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

    stmts = []
    n_state_var = adj_mat.shape[0]
    n_args = adj_mat.shape[0]

    # fn0, fn1, ... = fn_args
    fn_args_names = [f"fn{i}" for i in range(N_EDGE_TYPE)]
    stmts.append(
        mk_list_assign(
            targets=[mk_var(name) for name in fn_args_names],
            value=set_ctx(mk_var(ODE_FN_ARGS), ast.Load()),
        )
    )

    # Collect the location of each fn_args
    fn_to_coords = {fn_idx: [] for fn_idx in range(1, N_EDGE_TYPE)}
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i, j] != 0:
                fn_to_coords[adj_mat[i, j]].append((i, j))

    # vectorize arguments of functions Tx@x, Tx@xp, Ty@x, Ty@xp
    # ddt_x = Tf @ f(Tx@x, Tx@xp, Ty@x, Ty@xp) + ...
    ddt_rhs_exprs = []
    for fn_idx, coords in fn_to_coords.items():
        if coords:
            x_coords, y_coords = zip(*coords)
            x_coords, y_coords = list(x_coords), list(y_coords)
            Tx = mk_one_hot_vector_call(n_args, x_coords, row_one_hot=True)
            Ty = mk_one_hot_vector_call(n_args, y_coords, row_one_hot=True)
            Tf = mk_one_hot_vector_call(n_state_var, x_coords, row_one_hot=False)

            # Tf @ fn(Tx@x, Tx@xp, Ty@x, Ty@xp)
            ddt_rhs_exprs.append(
                set_ctx(
                    ast.BinOp(
                        left=Tf,
                        op=ast.MatMult(),
                        right=ast.Call(
                            func=mk_var(fn_args_names[fn_idx]),
                            args=[
                                ast.BinOp(
                                    left=Tx,
                                    op=ast.MatMult(),
                                    right=mk_var(ODE_INPUT_VAR),
                                ),
                                ast.BinOp(
                                    left=Tx, op=ast.MatMult(), right=mk_var(ODE_ARGS)
                                ),
                                ast.BinOp(
                                    left=Ty,
                                    op=ast.MatMult(),
                                    right=mk_var(ODE_INPUT_VAR),
                                ),
                                ast.BinOp(
                                    left=Ty, op=ast.MatMult(), right=mk_var(ODE_ARGS)
                                ),
                            ],
                            keywords=[],
                        ),
                    ),
                    ast.Load(),
                )
            )
    if ddt_rhs_exprs:
        stmts.append(
            set_ctx(
                ast.Assign(
                    targets=[mk_var(n_state(ddt(ODE_INPUT_VAR, order=1)))],
                    value=concat_call_expr(ddt_rhs_exprs, ast.Add()),
                ),
                ast.Store(),
            )
        )
        for call in ddt_rhs_exprs:
            set_ctx(call, ast.Load())
    else:
        stmts.append(
            set_ctx(
                ast.Assign(
                    targets=[mk_var(n_state(ddt(ODE_INPUT_VAR, order=1)))],
                    value=ast.Constant(value=0),
                ),
                ast.Store(),
            )
        )

    stmts.append(
        set_ctx(
            ast.Return(
                value=mk_var(n_state(ddt(ODE_INPUT_VAR, order=1))),
            ),
            ast.Load(),
        )
    )
    return wrap_ode_fn(stmts, printing=printing)


def concat_call_expr(exprs: list[ast.expr], operator: ast.operator) -> ast.operator:
    """concatenate ast.Call expressions with the given operator"""
    if len(exprs) == 1:
        return exprs[0]
    if len(exprs) == 2:
        return ast.BinOp(left=exprs[0], op=operator, right=exprs[1])
    return ast.BinOp(
        left=exprs[0], op=operator, right=concat_call_expr(exprs[1:], operator)
    )


def wrap_ode_fn(ode_stmts: list[ast.stmt], printing: bool = False) -> ast.FunctionDef:

    arguments = ast.arguments(
        posonlyargs=[],
        args=[
            mk_arg(TIME),
            mk_arg(ODE_INPUT_VAR),
            mk_arg(ODE_ARGS),
            mk_arg(ODE_FN_ARGS),
        ],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )
    func_def = ast.FunctionDef(
        name=ODE_FN_NAME, args=arguments, body=ode_stmts, decorator_list=[]
    )
    module = ast.Module([func_def], type_ignores=[])
    module = ast.fix_missing_locations(module)

    if printing:
        print(ast.unparse(module))

    code = compile(source=module, filename="__tmp__.py", mode="exec")
    namespace = {"jnp": jnp, "one_hot_vectors": one_hot_vectors}
    exec(code, namespace)

    return namespace[ODE_FN_NAME]


class TestModule(eqx.Module):

    trainable: jax.Array

    def __init__(self, init_trainable: jax.Array):
        self.trainable = init_trainable

    def __call__(
        self,
        time_info: TimeInfo,
        initial_state: jax.Array,
        switch: jax.Array,  # dummy for io consistency
        args_seed: jax.typing.DTypeLike,  # dummy for io consistency
        noise_seed: jax.typing.DTypeLike,  # dummy for io consistency
    ):
        """The differentiable forward pass of the circuit simulation."""
        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.ode_fn),
            solver=diffrax.Heun(),
            t0=time_info.t0,
            t1=time_info.t1,
            dt0=time_info.dt0,
            y0=initial_state,
            saveat=diffrax.SaveAt(ts=time_info.saveat),
            args=self.trainable,
        )

        return solution.ys

    def ode_fn(self, t, y, args):
        raise NotImplementedError


n_inner_loop = 20
n_state_vars = [i for i in range(2, 35, 4)]
orig_fw_times, orig_bw_times = [[] for _ in n_state_vars], [[] for _ in n_state_vars]
vect_fw_times, vect_bw_times = [[] for _ in n_state_vars], [[] for _ in n_state_vars]
for i, n_state_var in enumerate(n_state_vars):
    for _ in range(n_inner_loop):
        mat = rand_adjacency_matrix(n_state_var)
        ode_fn = adj_mat_to_ode_stmts(mat, printing=False)
        ode_fn_vectorized = adj_mat_to_ode_stmts_vectorized(mat, printing=False)
        TestModule.ode_fn = lambda self, t, y, args: ode_fn(t, y, args, edge_fn)

        test_obj = TestModule(init_trainable=jnp.zeros(n_state_var))
        orig_runtime, orig_trace = test_forward_pass(
            test_obj, n_state_var=n_state_var, printing=False
        )

        TestModule.ode_fn = lambda self, t, y, args: ode_fn_vectorized(
            t, y, args, edge_fn
        )
        test_obj = TestModule(init_trainable=jnp.zeros(n_state_var))
        vect_runtime, vect_trace = test_forward_pass(
            test_obj, n_state_var=n_state_var, printing=False
        )
        assert jnp.allclose(orig_trace, vect_trace)

        orig_fw_times[i].append(orig_runtime)
        vect_fw_times[i].append(vect_runtime)
        orig_bw_times[i].append(
            test_backward_pass(test_obj, n_state_var=n_state_var, printing=False)
        )
        vect_bw_times[i].append(
            test_backward_pass(test_obj, n_state_var=n_state_var, printing=False)
        )

    print(
        f"n_state_var: {n_state_var},\n"
        f"\torig_fw_time: {np.mean(orig_fw_times[i]):.4f}, orig_bw_time (compile, exec): {np.mean(orig_bw_times[i], axis=0)}\n"
        f"\tvect_fw_time: {np.mean(vect_fw_times[i]):.4f}, vect_bw_time (compile, exec): {np.mean(vect_bw_times[i], axis=0)}"
    )

# Save the results
with open("compile_time.npz", "wb") as f:
    np.savez(
        f,
        n_state_vars=n_state_vars,
        orig_fw_times=orig_fw_times,
        orig_bw_times=orig_bw_times,
        vect_fw_times=vect_fw_times,
        vect_bw_times=vect_bw_times,
    )

mean_orig, mean_vectorized = np.mean(orig_bw_times, axis=1), np.mean(
    vect_bw_times, axis=1
)
std_orig, std_vectorized = np.std(orig_bw_times, axis=1), np.std(vect_bw_times, axis=1)
print(mean_orig)

# Create figure and plot mean and error bars
plt.plot(n_state_vars, mean_orig[:, 0], label="Original")
plt.errorbar(n_state_vars, mean_orig[:, 0], yerr=std_orig[:, 0], fmt="o", capsize=5)
plt.plot(n_state_vars, mean_vectorized[:, 0], label="Vectorized")
plt.errorbar(
    n_state_vars, mean_vectorized[:, 0], yerr=std_vectorized[:, 0], fmt="o", capsize=5
)
plt.xlabel("Number of state variables")
plt.ylabel("Time (s)")
plt.title("Time to compile the backward pass")
plt.legend()
plt.grid()
plt.show()

# Create figure and plot mean and error bars
plt.plot(n_state_vars, mean_orig[:, 1], label="Original")
plt.errorbar(n_state_vars, mean_orig[:, 1], yerr=std_orig[:, 1], fmt="o", capsize=5)
plt.plot(n_state_vars, mean_vectorized[:, 1], label="Vectorized")
plt.errorbar(
    n_state_vars, mean_vectorized[:, 1], yerr=std_vectorized[:, 1], fmt="o", capsize=5
)
plt.xlabel("Number of state variables")
plt.ylabel("Time (s)")
plt.title("Time to execute the 10 backward pass")
plt.legend()
plt.grid()
plt.show()
