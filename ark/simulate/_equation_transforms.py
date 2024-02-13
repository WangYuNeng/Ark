
import sympy as sp

def collapse_derivative(pair: tuple[sp.Symbol, sp.Expr]) -> sp.Eq:
    """Turns tuple of derivative + sympy expression into a single sympy equation."""
    if (var_name := pair[0].name).startswith("ddt_"):
        symbol = sp.symbols(var_name[4:])
        equation = sp.Eq(sp.Derivative(symbol, sp.symbols("time")), pair[1])
        return equation.subs(sp.symbols("time"), sp.symbols("t"))
    else:
        raise ValueError(f"Not a derivative expression: {pair}")


def make_symbols_positive(eq: sp.Eq) -> sp.Eq:
    """Turn all sympy symbols positive"""
    to_replace = {}
    for symbol in eq.free_symbols:
        if symbol.is_Symbol:
            to_replace[symbol] = sp.Symbol(name=symbol.name, positive=True)
    return eq.subs(to_replace)
