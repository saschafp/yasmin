from dataclasses import dataclass
from typing import cast

from yasmin.frontend.expr import Expr
from yasmin.frontend.stencil import Stencil
from yasmin.ir import stencil as stencil_ir


@dataclass(frozen=True, slots=True)
class Operator:
    target: Expr
    value: Expr | Stencil

    def __post_init__(self) -> None:
        if not isinstance(self.target.ir, stencil_ir.FieldAccess):
            raise TypeError(
                f"Operator target must be a FieldAccess, "
                f"got {type(self.target.ir).__name__} instead"
            )

    @property
    def ir(self) -> stencil_ir.Operator:
        # target is guaranteed to be a FieldAccess
        # this is checked in __post_init__
        target = cast(stencil_ir.FieldAccess, self.target.ir)

        value = self.value.expr.ir if isinstance(self.value, Stencil) else self.value.ir

        return stencil_ir.Operator(
            statements=(
                stencil_ir.Assign(
                    target=target,
                    value=value,
                ),
            )
        )
