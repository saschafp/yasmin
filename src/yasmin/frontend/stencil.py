from dataclasses import dataclass

from yasmin.frontend.expr import Expr


@dataclass(frozen=True, slots=True)
class Stencil:
    expr: Expr
