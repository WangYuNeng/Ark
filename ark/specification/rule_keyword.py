from dataclasses import dataclass
import sympy


class Expression:
    """Expression for writing production rules
    Ref: https://github.com/ncsys-lab/analog-verification/blob/new-modelspec/core/expr.py
    """

    def __add__(self, other: "Expression | float | int") -> "Sum":
        if isinstance(other, int):
            other = Constant(float(other))
        if isinstance(other, float):
            other = Constant(other)
        return Sum(self, other)

    def __sub__(self, other: "Expression | float | int") -> "Difference":
        if isinstance(other, int):
            other = Constant(float(other))
        if isinstance(other, float):
            other = Constant(other)
        return Difference(self, other)

    def __mul__(self, other: "Expression | float | int") -> "Product":
        if isinstance(other, int):
            other = Constant(float(other))
        if isinstance(other, float):
            other = Constant(other)
        return Product(self, other)

    def __truediv__(self, other: "Expression | float | int") -> "Quotient":
        if isinstance(other, int):
            other = Constant(float(other))
        if isinstance(other, float):
            other = Constant(other)
        return Quotient(self, other)

    def __neg__(self) -> "Negation":
        return Negation(self)

    def __pow__(self, other: "Expression | float | int") -> "Power":
        if isinstance(other, int):
            other = Constant(float(other))
        if isinstance(other, float):
            other = Constant(other)
        return Power(self, other)

    def __call__(self, *args: "List[Expression | float | int]") -> "FunctionCall":
        converted_args = []
        for arg in args:
            if isinstance(arg, int):
                arg = Constant(float(arg))
            if isinstance(arg, float):
                arg = Constant(arg)
            converted_args.append(arg)
        return FunctionCall(self, converted_args)

    @property
    def sympy(self) -> sympy.Expr:
        """Convert to sympy expression"""
        raise NotImplementedError


@dataclass
class Variable(Expression):
    """Variable expression"""

    name: str

    def __str__(self) -> str:
        return self.name

    @property
    def sympy(self) -> sympy.Expr:
        return sympy.Symbol(self.name, real=True)


@dataclass
class Constant(Expression):
    """Constant expression"""

    value: float

    def __str__(self) -> str:
        return str(self.value)

    @property
    def sympy(self) -> sympy.Expr:
        return sympy.RealNumber(self.value)


@dataclass
class Sum(Expression):
    """Sum expression"""

    left: Expression
    right: Expression

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"

    @property
    def sympy(self) -> sympy.Expr:
        return self.left.sympy + self.right.sympy


@dataclass
class Difference(Expression):
    """Difference expression"""

    left: Expression
    right: Expression

    def __str__(self) -> str:
        return f"({self.left} - {self.right})"

    @property
    def sympy(self) -> sympy.Expr:
        return self.left.sympy - self.right.sympy


@dataclass
class Product(Expression):
    """Product expression"""

    left: Expression
    right: Expression

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"

    @property
    def sympy(self) -> sympy.Expr:
        return self.left.sympy * self.right.sympy


@dataclass
class Quotient(Expression):
    """Quotient expression"""

    left: Expression
    right: Expression

    def __str__(self) -> str:
        return f"({self.left} / {self.right})"

    @property
    def sympy(self) -> sympy.Expr:
        return self.left.sympy / self.right.sympy


@dataclass
class Negation(Expression):
    """Negation expression"""

    expr: Expression

    def __str__(self) -> str:
        return f"(-{self.expr})"

    @property
    def sympy(self) -> sympy.Expr:
        return -self.expr.sympy


@dataclass
class Power(Expression):
    """Power expression"""

    base: Expression
    exp: Expression

    def __str__(self) -> str:
        return f"({self.base} ** {self.exp})"

    @property
    def sympy(self) -> sympy.Expr:
        return self.base.sympy**self.exp.sympy


@dataclass
class FunctionCall(Expression):
    """Function call expression"""

    fn: Expression
    args: list[Expression]

    def __str__(self) -> str:
        return f'{self.fn}({", ".join(str(arg) for arg in self.args)})'

    @property
    def sympy(self) -> sympy.Expr:
        sympy_fn = sympy.Function(self.fn.sympy, real=True)
        return sympy_fn(*[arg.sympy for arg in self.args])


@dataclass
class RuleKeyword:
    """Keyword class for writing production rules"""

    name: str

    def __getattribute__(self, __name: str) -> Variable:
        if __name.startswith("__") and __name.endswith("__"):
            # Handle python default attributes
            return object.__getattribute__(self, __name)
        else:
            self_name = object.__getattribute__(self, "name")
            named_attr = f"{self_name}.{__name}"
            return Variable(named_attr)

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return str(self.name)


class Target(RuleKeyword):
    """Target keyword denoting the posistion of a node in the edge"""

    def is_source(self):
        return self.__name__ == "s"

    def is_dest(self):
        return self.__name__ == "t"


def var(rule_keyword: RuleKeyword) -> Variable:
    """Returns a state variable for writing production rules"""
    name = object.__getattribute__(rule_keyword, "name")
    return Variable(name)


SRC, DST = (
    Target("s"),
    Target("t"),
)
EDGE, SELF = RuleKeyword("e"), Target("SELF")
TIME = Variable("time")
VAR = var


def kw_name(keyword: RuleKeyword):
    """Returns the name of the keyword"""
    return object.__getattribute__(keyword, "name")
