
class Constraint:
    
    def __init__(self) -> None:
        pass

class DegreeConstraint(Constraint):

    def __init__(self, expr: str) -> None:
        super().__init__()
        self._expr = expr

    @property
    def expr(self):
        return self._expr

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, self.expr)