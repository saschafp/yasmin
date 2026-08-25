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
        target = self.target._as_ir()

        if not isinstance(target, stencil_ir.FieldAccess):
            raise TypeError(
                "Operator target must be a FieldAccess, "
                f"got {type(target).__name__} instead"
            )

    def _as_ir(self) -> stencil_ir.Operator:
        target = cast(
            stencil_ir.FieldAccess,
            self.target._as_ir(),
        )

        value = self.value._as_ir()

        return stencil_ir.Operator(
            statements=(
                stencil_ir.Assign(
                    target=target,
                    value=value,
                ),
            )
        )