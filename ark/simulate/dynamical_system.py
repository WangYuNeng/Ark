from dataclasses import dataclass
from functools import wraps, partial
from typing import Callable, Self
from warnings import warn

import numpy as np
import diffrax as dr
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pint
import sympy as sp

from ark.cdg.cdg import CDG

from ark.compiler import ArkCompiler

from ark.rewrite import SympyRewriteGen

from ark.simulate.sample_mismatch import get_attr_spec

from ark.specification.attribute_def import AttrDef

from ark.specification.rule_keyword import sympy_function

from ark.specification.specification import CDGSpec
from ark.simulate._equation_transforms import collapse_derivative, make_symbols_positive


units = pint.get_application_registry()

class AugmentedState(eqx.Module):
    y: jax.Array
    "Main state variable for system, contains the transient variables. Shape (num_transient, )"

    y_error: jax.Array
    "A single term (scalar) for the sensitivity of the transient variables at the current time with respect to equal noise in all transient parameters."

    theta_error: jax.Array
    "A matrix of sensitivities of the transient variables at the current time with respect to all parameters. Shape (num_transient, num_parameters)"

    def __init__(self, y, y_error, theta_error):
        self.y = y
        self.y_error = y_error
        self.theta_error = theta_error

    def __iter__(self):
        return iter((self.y, self.y_error, self.theta_error))


def augment_with_error(
    *,
    f: callable,
    initial_state: jax.Array,
    num_parameters: int,
    constant_error_rates: jax.Array,
) -> callable:
    """
    Construct a new dy/dt that also captures the propagation of error information.
    """

    @wraps(f)
    def new_f(t, y_aug: AugmentedState, args) -> AugmentedState:
        # Cast to jnp.bfloat16 for better performance
        t = jnp.array(t, dtype=jnp.bfloat16)
        n_theta = len(args)

        y, _y_error_val, theta_error_val = y_aug
        y = jnp.array(y, dtype=jnp.bfloat16)
        theta_error_val = jnp.array(theta_error_val, dtype=jnp.bfloat16)
        args = jnp.array(args, dtype=jnp.bfloat16)

        # Previous value for dy/dtheta
        z = theta_error_val  # (transients, parameters)

        # y, error_val = jnp.split(y_aug, [y_aug.shape[0] // 2])

        # args = theta
        def f_of_y(y, t, args):
            return f(t, y, args)

        def f_of_args(args, t, y):
            return f(t, y, args)
        

        # ∂f/∂theta
        partial_theta_f = jax.jacobian(f_of_args, argnums=0)(args, t, y)
        # ∂f/∂y * dy/dtheta
        partial_y_f = jax.vmap(lambda z_param_j: eqx.filter_jvp(f_of_y, (y,), (z_param_j,), t=t, args=args), in_axes=1, out_axes=1)(z)[1]

        # We have f = dy/dt and z = dy/dtheta, so dz/dt = df/dtheta
        # dz/dt = df/dtheta = (∂f/∂y * dy/dtheta) + ∂f/∂theta
        dz_dt = partial_y_f + partial_theta_f

        # Computing the sensitivity of the state with respect noise in transient state
        # Compute tr(J @ Sigma) which turns out to be just tr(J)
        def compute_diag_y(x, inp, fn, i, t, args):
            return fn(inp.at[i].set(x), t, args)[i]

        @partial(jax.vmap, in_axes=(None, None, 0, None, None))
        def compute_grad_diag_y(inp, fn, i, t, args):
            return eqx.filter_grad(compute_diag_y)(inp[i], inp, fn, i, t, args)

        idcs = jnp.arange(y.shape[0])

        # Trace of the Jacobian multiplied with the covariance matrix
        y_additional_error = jnp.sum(compute_grad_diag_y(y, f_of_y, idcs, t, args))

        f_value = f(t, y, args)

        print('ode augmented', f_value.shape, y_additional_error.shape, dz_dt.shape)
        diff_state = AugmentedState(
            y=f_value,  # (num_transient, )
            y_error=y_additional_error,  # ()
            theta_error=dz_dt,  # (num_transient, num_parameters)
        )
        assert diff_state.y.shape == y.shape
        assert diff_state.y_error.shape == ()
        assert diff_state.theta_error.shape == (y.shape[0], num_parameters)

        return diff_state

    num_transient = initial_state.shape[0]
    augmented_initial_state = AugmentedState(
        y=initial_state,
        y_error=jnp.zeros(shape=()),
        theta_error=jnp.zeros(shape=(num_transient, num_parameters,)),
    )

    return new_f, augmented_initial_state


@sympy_function
def _pulse_sympy(
    t,
    amplitude=sp.S.One,
    delay=sp.S.Zero,
    rise_time=sp.nsimplify(5e-9),
    fall_time=sp.nsimplify(5e-9),
    pulse_width=sp.nsimplify(10e-9),
):
    t = t - delay
    # Use a sympy piecewise function to represent the pulse
    return sp.Piecewise(
        (amplitude * t / rise_time, t < rise_time),
        (amplitude, t < rise_time + pulse_width),
        (
            amplitude * (sp.S.One - (t - pulse_width - rise_time) / fall_time),
            t < rise_time + pulse_width + fall_time,
        ),
        (sp.S.Zero, True),
    )


# For every 4 terms on a line, add a line break
# _ADD_BREAKS_RE = re.compile(r"(([+-][^\n+-]+){5})")


@dataclass
class NoiseDescription:
    pass


class SymbolType:
    pass
    # def __repr__(self):
    #     attrs = self.__dict__
    #     inner = ', '.join(map(lambda e: f'{e[0]}={e[1]!r}', attrs.items()))
    #     return f'{self.__class__.__name__}({inner})'


@dataclass
class Time(SymbolType):
    time_scale: float


@dataclass
class Transient(SymbolType):
    name: str
    unit: pint.Unit
    noise: NoiseDescription


@dataclass
class Parameter(SymbolType):
    name: str
    unit: pint.Unit
    variability: AttrDef


SymbolType.Time = Time
SymbolType.Transient = Transient
SymbolType.Parameter = Parameter


class DynamicalSystem:
    diffeqs: list[sp.Eq]
    symbol_types: dict[sp.Symbol, SymbolType]

    def __init__(
        self,
        diffeqs: list[sp.Eq],
        symbol_types: dict[sp.Symbol, SymbolType] | None = None,
    ):
        self.diffeqs = diffeqs
        if symbol_types is None:
            symbol_types = {}
        self.symbol_types = symbol_types

    @classmethod
    def from_ark(cls, cdg: CDG, spec: CDGSpec, time_scale: float = 1e-8) -> Self:
        compiler = ArkCompiler(rewrite=SympyRewriteGen())
        sympy_pairs = compiler.compile_sympy(cdg=cdg, cdg_spec=spec, help_fn=[])
        sympy_diffeqs = [
            make_symbols_positive(collapse_derivative(pair)) for pair in sympy_pairs
        ]

        InpI_0_fn = sp.Function("InpI_0_fn")
        t = sp.symbols("t", positive=True)

        # Automatically replace the input pulse if it exists
        if InpI_0_fn(t) in sympy_diffeqs[0].atoms(sp.Function):
            warn("Replacing InpI_0_fn with default expression")
            sympy_diffeqs[0] = sympy_diffeqs[0].subs(InpI_0_fn(t), _pulse_sympy(t))

        self = cls(diffeqs=sympy_diffeqs)

        def get_symbol_type(symbol_name: str):
            if symbol_name == "t":
                return SymbolType.Time(time_scale=time_scale)
            elif symbol_name[-1].isnumeric() or symbol_name[-1] == "V":
                # Find last non-numeric non-underscore character
                ci = len(symbol_name) - 1
                while symbol_name[ci].isnumeric() or symbol_name[ci] == "_":
                    ci -= 1
                c = symbol_name[ci]
                if c == "V":
                    unit = units.volt
                elif c == "I":
                    unit = units.ampere
                else:
                    raise ValueError(f"Unknown unit character {c}")
                return SymbolType.Transient(
                    name=symbol_name, unit=unit, noise=NoiseDescription()
                )
            else:
                attr_spec = get_attr_spec(graph=cdg, variable_name=symbol_name)
                if symbol_name.endswith("g"):
                    unit = units.siemens
                elif symbol_name.endswith("c"):
                    unit = units.farad
                elif symbol_name.endswith("r"):
                    unit = units.ohm
                elif symbol_name.endswith("l"):
                    unit = units.henry
                elif symbol_name.endswith("ws") or symbol_name.endswith("wt"):
                    unit = units.dimensionless
                else:
                    raise ValueError(f"Unknown unit character {symbol_name}")
                return SymbolType.Parameter(
                    name=symbol_name, variability=attr_spec, unit=unit
                )

        free_symbols = self._free_symbols()

        # Determine the types of the symbols
        symbol_types = {symbol: get_symbol_type(symbol.name) for symbol in free_symbols}
        sorted_symbol_types = dict(
            sorted(
                symbol_types.items(),
                key=lambda x: tuple(
                    int(e) if e.isnumeric() else e for e in x[0].name.split("_")
                ),
            )
        )
        self.symbol_types = {
            **{
                k: v
                for k, v in sorted_symbol_types.items()
                if isinstance(v, SymbolType.Time)
            },
            **{
                k: v
                for k, v in sorted_symbol_types.items()
                if isinstance(v, SymbolType.Transient)
            },
            **{
                k: v
                for k, v in sorted_symbol_types.items()
                if isinstance(v, SymbolType.Parameter)
            },
        }

        symbol_order = {s: i for i, s in enumerate(self.symbol_types.keys())}
        self.diffeqs.sort(
            key=lambda x: symbol_order.get(next(iter(x.lhs.free_symbols)))
        )

        self = self.rescale_system(time_scale_factor=time_scale)

        return self

    @property
    def f(self):
        transient_symbols = self.transient_symbols()
        parameter_symbols = self.parameter_symbols()
        (time_symbol, _) = self.time_symbol()

        all_symbols = (
            [time_symbol]
            + list(transient_symbols.keys())
            + list(parameter_symbols.keys())
        )
        rhs_functions = [
            sp.lambdify(all_symbols, expr=eq.rhs, modules="jax") for eq in self.diffeqs
        ]

        def f(t, y, args):
            # args cannot change over the course of the integration
            # y_diff = jnp.zeros_like(y)
            # print('From f', {k: args[i] for i, k in enumerate(parameter_symbols.keys())})
            y_diff = jnp.array(
                [rhs_function(t, *y, *args) for rhs_function in rhs_functions],
                dtype=y.dtype,
            )
            # for i, (transient_symbol, _transient_type) in enumerate(transient_symbols.items()):
            #     transient_gradient = rhs_functions[i](t, *y, *args)
            #     print(transient_gradient)
            #     y_diff = y_diff.at[i].set(transient_gradient)
            return y_diff

        return f

    def solve_system(
        self, initial_values: jax.Array, parameter_values: jax.Array, saveat: dr.SaveAt
    ) -> dr.Solution:
        jax.debug.print("Starting system solve...")

        time_scale = self.time_symbol()[1].time_scale
        # f(t, y, args)
        f = self.f

        max_steps = 1_000
        t1 = 75e-9 / time_scale  # End time is scaled by time_scale
        dt0 = t1 / max_steps

        term = dr.ODETerm(f)
        solver = dr.Tsit5()
        # adjoint = dr.BacksolveAdjoint(solver=solver)
        adjoint = dr.RecursiveCheckpointAdjoint()
        # adjoint = dr.DirectAdjoint()

        system_solution = dr.diffeqsolve(
            terms=term,
            solver=solver,
            t0=0.0,
            t1=t1,
            dt0=dt0,
            y0=initial_values,
            args=parameter_values,
            saveat=saveat,
            max_steps=max_steps,
            adjoint=adjoint,
        )
        # Transform the timescale back to the original scale
        system_solution = eqx.tree_at(
            where=lambda x: x.ts,
            pytree=system_solution,
            replace_fn=lambda ts: ts * time_scale,
        )
        return system_solution

    def solve_system_with_sensitivity(
        self, initial_values: jax.Array, parameter_values: jax.Array, saveat: dr.SaveAt
    ) -> dr.Solution:
        jax.debug.print("Starting system solve with sensitivity...")

        # f = dy/dt for the dynamical system
        f = self.f

        # Augment with sensitivity information
        SENSITIVITY_SCALE = 1.0
        f_aug, augmented_initial_values = augment_with_error(
            f=f,
            initial_state=initial_values,
            num_parameters=parameter_values.shape[0],
            constant_error_rates=jnp.full_like(
                initial_values, fill_value=SENSITIVITY_SCALE
            ),
        )

        time_scale = self.time_symbol()[1].time_scale

        max_steps = 1_000
        t1 = 75e-9 / time_scale
        dt0 = t1 / max_steps

        term = dr.ODETerm(f_aug)
        solver = dr.Tsit5(
            # scan_kind='lax',
        )
        adjoint = dr.RecursiveCheckpointAdjoint()
        # adjoint = dr.BacksolveAdjoint()
        print(f'Using adjoint {adjoint}')

        system_solution = dr.diffeqsolve(
            terms=term,
            solver=solver,
            t0=0.0,
            t1=t1,
            dt0=dt0,
            y0=augmented_initial_values,
            args=parameter_values,
            saveat=saveat,
            max_steps=max_steps,
            adjoint=adjoint,
            # progress_meter=dr.TqdmProgressMeter(),
        )
        system_solution = eqx.tree_at(
            where=lambda x: x.ts,
            pytree=system_solution,
            replace_fn=lambda ts: ts * time_scale,
        )
        return system_solution

    def parse_parameters(self, parameters: jax.Array) -> dict[sp.Symbol, jax.Array]:
        assert parameters.shape[-1] == len(self.symbol_types)
        return {k: parameters[..., i] for i, k in enumerate(self.symbol_types.keys())}

    def encode_parameters(
        self, parameter_dict: dict[sp.Symbol, jax.Array]
    ) -> jax.Array:
        assert self.parameter_symbols().keys() == parameter_dict.keys()
        return jnp.array([parameter_dict[k] for k in self.parameter_symbols().keys()])

    def zero_state(self) -> jax.Array:
        return jnp.zeros(len(self.transient_symbols()), dtype=float)

    def map_equations(self, fn: Callable[[sp.Eq], sp.Eq]) -> Self:
        new_diffeqs = [fn(eq) for eq in self.diffeqs]
        return DynamicalSystem(new_diffeqs, symbol_types=self.symbol_types)

    def map_rhs(self, fn: Callable[[sp.Expr], sp.Expr]) -> Self:
        new_diffeqs = [sp.Eq(eq.lhs, fn(eq.rhs)) for eq in self.diffeqs]
        return DynamicalSystem(new_diffeqs, symbol_types=self.symbol_types)

    def transient_symbols(self) -> dict[sp.Symbol, SymbolType.Transient]:
        return {
            symbol: type
            for symbol, type in self.symbol_types.items()
            if isinstance(type, SymbolType.Transient)
        }

    def parameter_symbols(self) -> dict[sp.Symbol, SymbolType.Parameter]:
        return {
            symbol: type
            for symbol, type in self.symbol_types.items()
            if isinstance(type, SymbolType.Parameter)
        }

    def time_symbol(self) -> tuple[sp.Symbol, SymbolType.Time]:
        return next(
            (symbol, type)
            for symbol, type in self.symbol_types.items()
            if isinstance(type, SymbolType.Time)
        )

    def sample_parameters(self, nominal_values: jax.Array, key: jax.Array) -> jax.Array:
        # Each parameter is equal to its nominal value multiplied by a normal distribution centered in 1 with stddev 0.1 (given by AttrDef)
        n_parameters = len(self.parameter_symbols())
        rel_error = jrandom.truncated_normal(
            key=key, lower=1e-7, upper=100, shape=(n_parameters,)
        )
        return nominal_values * (1 + rel_error * 0.1)

    def _repr_latex_(self):
        equation_strs = [
            sp.latex(eq, diff_operator="rd").replace("=", "&=") for eq in self.diffeqs
        ]
        joiner = "\\\\\n"
        align_str = f"\\begin{{align*}}\n{joiner.join(equation_strs)}\n\\end{{align*}}"
        # align_str = _ADD_BREAKS_RE.sub(r"\g<1>\\\\\n&", align_str, count=1)
        return align_str

    def rescale_system(self, time_scale_factor=1e-8):
        # Time rescaling
        # self.symbol_types[self.time_symbol()]
        k = sp.symbols("k", positive=True)
        (t, _) = self.time_symbol()

        def fix_equation(equation):
            rhs = equation.rhs
            rhs = rhs.subs(t, t * k)

            def fix_piecewise_ts(*args):
                fixed_args = [(a[0], sp.solve(a[1], t, minimal=True)) for a in args]
                new_args = [
                    (a[0].simplify(), a[1] if not isinstance(a[1], list) else True)
                    for a in fixed_args
                ]
                return sp.Piecewise(*new_args)

            rhs = (
                rhs.replace(sp.Piecewise, fix_piecewise_ts) * time_scale_factor
            ).factor()
            equation = equation.__class__(lhs=equation.lhs, rhs=rhs)
            equation = equation.subs(k, time_scale_factor)
            rhs = equation.rhs.factor()
            equation = equation.__class__(lhs=equation.lhs, rhs=rhs)
            return equation

        return self.map_equations(fix_equation)

    def _free_symbols(self):
        return set().union(*[eq.free_symbols for eq in self.diffeqs])


