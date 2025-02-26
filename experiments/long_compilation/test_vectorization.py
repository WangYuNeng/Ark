"""Experiment to test how much vectorization improvs compilation time.

Simplified graph model -- ode_fn only depends on the edge type and 
the arguments only takes the node state.
"""

import ast
from typing import Any, Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental import sparse
from test_long_compile import test_backward_pass, test_forward_pass

from ark.compiler import (  # mk_one_hot_vector_call,
    ddt,
    mk_arg,
    mk_list_assign,
    mk_var,
    n_state,
    set_ctx,
)
from ark.optimization.base_module import TimeInfo
from ark.util import mk_arr_access, mk_jnp_call, mk_jnp_scatter_gather

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
        return sparse.BCOO.fromdense(jnp.array(np.eye(dim)[indices]))
    return sparse.BCOO.fromdense(jnp.array(np.eye(dim)[indices].T))


def adj_mat_to_ode_stmts_vectorized(
    adj_mat: np.ndarray, printing: bool = False
) -> Callable:

    namespace = {}

    def mk_one_hot_vector_call(
        dim: int, indices: list[int], row_one_hot: bool
    ) -> ast.Name:
        """Create a call expression for one_hot_vectors function."""
        one_hot_arr = one_hot_vectors(dim, indices, row_one_hot)
        print(one_hot_arr)
        var_name = f"one_hot_vectors_{len(namespace)}"
        namespace[var_name] = one_hot_arr
        return mk_var(var_name)

    def mk_named_jnp_idx_arr(indices: list[int]) -> ast.Name:
        """Create a jnp array of the indices and return the name of the array."""
        var_name = f"jnp_idx_arr_{len(namespace)}"
        namespace[var_name] = jnp.array(indices)
        return mk_var(var_name)

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

    # vectorize arguments of functions x[xcoord], xp[xcoord], x[ycoord], xp[ycoord]
    # ddt_x = jnp.zeros(n_state_var).at[xcoord].add(f(x[xcoord], xp[xcoord], x[ycoord], Ty@xp)) + ...
    ddt_rhs_exprs = []
    for fn_idx, coords in fn_to_coords.items():
        if coords:
            x_coords, y_coords = zip(*coords)
            x_coord_ast = mk_named_jnp_idx_arr(list(x_coords))
            y_coord_ast = mk_named_jnp_idx_arr(list(y_coords))

            ddt_rhs_exprs.append(
                set_ctx(
                    mk_jnp_scatter_gather(
                        arr_size=n_state_var,
                        idx=x_coord_ast,
                        val=ast.Call(
                            func=mk_var(fn_args_names[fn_idx]),
                            args=[
                                mk_arr_access(
                                    lst=mk_var(ODE_INPUT_VAR), idx=x_coord_ast
                                ),
                                mk_arr_access(lst=mk_var(ODE_ARGS), idx=x_coord_ast),
                                mk_arr_access(
                                    lst=mk_var(ODE_INPUT_VAR), idx=y_coord_ast
                                ),
                                mk_arr_access(lst=mk_var(ODE_ARGS), idx=y_coord_ast),
                            ],
                            keywords=[],
                        ),
                        gather="add",
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
    return wrap_ode_fn(stmts, namespace=namespace, printing=printing)


def concat_call_expr(exprs: list[ast.expr], operator: ast.operator) -> ast.operator:
    """concatenate ast.Call expressions with the given operator"""
    if len(exprs) == 1:
        return exprs[0]
    if len(exprs) == 2:
        return ast.BinOp(left=exprs[0], op=operator, right=exprs[1])
    return ast.BinOp(
        left=exprs[0], op=operator, right=concat_call_expr(exprs[1:], operator)
    )


def wrap_ode_fn(
    ode_stmts: list[ast.stmt], namespace: dict[str, Any] = {}, printing: bool = False
) -> ast.FunctionDef:

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
    namespace = {"jnp": jnp, "one_hot_vectors": one_hot_vectors} | namespace
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


def plot_cmp_trace(
    n_state_vars: int, orig_data: np.ndarray, vect_data: np.ndarray, forward: bool
):
    mean_orig, mean_vectorized = np.mean(orig_data, axis=1), np.mean(vect_data, axis=1)
    std_orig, std_vectorized = np.std(orig_data, axis=1), np.std(vect_data, axis=1)
    # Create figure and plot mean and error bars
    pass_type = "forward" if forward else "backward"
    plt.plot(n_state_vars, mean_orig[:, 0], label="Original")
    plt.errorbar(n_state_vars, mean_orig[:, 0], yerr=std_orig[:, 0], fmt="o", capsize=5)
    plt.plot(n_state_vars, mean_vectorized[:, 0], label="Vectorized")
    plt.errorbar(
        n_state_vars,
        mean_vectorized[:, 0],
        yerr=std_vectorized[:, 0],
        fmt="o",
        capsize=5,
    )
    plt.xlabel("Number of state variables")
    plt.ylabel("Time (s)")
    plt.title(f"Time to compile the {pass_type} pass")
    plt.legend()
    plt.grid()
    plt.show()

    # Create figure and plot mean and error bars
    plt.plot(n_state_vars, mean_orig[:, 1], label="Original")
    plt.errorbar(n_state_vars, mean_orig[:, 1], yerr=std_orig[:, 1], fmt="o", capsize=5)
    plt.plot(n_state_vars, mean_vectorized[:, 1], label="Vectorized")
    plt.errorbar(
        n_state_vars,
        mean_vectorized[:, 1],
        yerr=std_vectorized[:, 1],
        fmt="o",
        capsize=5,
    )
    plt.xlabel("Number of state variables")
    plt.ylabel("Time (s)")
    plt.title(f"Time to execute the 10 {pass_type} pass")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    n_inner_loop = 4
    n_state_vars = [i for i in range(2, 35, 4)]
    orig_fw_times, orig_bw_times = [[] for _ in n_state_vars], [
        [] for _ in n_state_vars
    ]
    vect_fw_times, vect_bw_times = [[] for _ in n_state_vars], [
        [] for _ in n_state_vars
    ]
    for i, n_state_var in enumerate(n_state_vars):
        for _ in range(n_inner_loop):
            mat = rand_adjacency_matrix(n_state_var)
            ode_fn = adj_mat_to_ode_stmts(mat, printing=False)
            ode_fn_vectorized = adj_mat_to_ode_stmts_vectorized(mat, printing=False)
            TestModule.ode_fn = lambda self, t, y, args: ode_fn(t, y, args, edge_fn)

            test_obj = TestModule(init_trainable=jnp.zeros(n_state_var))
            orig_compile_time, orig_exec_time, orig_trace = test_forward_pass(
                test_obj, n_state_var=n_state_var, printing=False
            )

            TestModule.ode_fn = lambda self, t, y, args: ode_fn_vectorized(
                t, y, args, edge_fn
            )
            test_obj = TestModule(init_trainable=jnp.zeros(n_state_var))
            vect_compile_time, vec_exec_time, vect_trace = test_forward_pass(
                test_obj, n_state_var=n_state_var, printing=False
            )
            assert jnp.allclose(orig_trace, vect_trace)

            orig_fw_times[i].append([orig_compile_time, orig_exec_time])
            vect_fw_times[i].append([vect_compile_time, vec_exec_time])
            orig_bw_times[i].append(
                test_backward_pass(test_obj, n_state_var=n_state_var, printing=False)
            )
            vect_bw_times[i].append(
                test_backward_pass(test_obj, n_state_var=n_state_var, printing=False)
            )

        print(
            f"n_state_var: {n_state_var},\n"
            f"\torig_fw_time (compile, exec): {np.mean(orig_fw_times[i], axis=0)}, orig_bw_time (compile, exec): {np.mean(orig_bw_times[i], axis=0)}\n"
            f"\tvect_fw_time (compile, exec): {np.mean(vect_fw_times[i], axis=0)}, vect_bw_time (compile, exec): {np.mean(vect_bw_times[i], axis=0)}"
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

    plot_cmp_trace(n_state_vars, orig_fw_times, vect_fw_times, forward=True)
    plot_cmp_trace(n_state_vars, orig_bw_times, vect_bw_times, forward=False)
