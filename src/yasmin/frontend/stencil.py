from dataclasses import dataclass

from yasmin.frontend.expr import Expr, SymbolicExpr
from yasmin.ir import stencil as stencil_ir


@dataclass(frozen=True, slots=True)
class Stencil(SymbolicExpr):
    expr: Expr

    def _as_ir(self) -> stencil_ir.Expr:
        return self.expr._as_ir()