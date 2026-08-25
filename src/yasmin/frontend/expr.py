from __future__ import annotations

from dataclasses import dataclass

from yasmin.core import Field
from yasmin.ir import stencil


@dataclass(frozen=True, slots=True)
class Expr:
    ir: stencil.Expr

    def __add__(self, other: Expr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.ADD,
                self.ir,
                as_ir_expr(other),
            )
        )

    def __radd__(self, other: Expr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.ADD,
                as_ir_expr(other),
                self.ir,
            )
        )

    def __sub__(self, other: Expr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.SUB,
                self.ir,
                as_ir_expr(other),
            )
        )

    def __rsub__(self, other: Expr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.SUB,
                as_ir_expr(other),
                self.ir,
            )
        )

    def __mul__(self, other: Expr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.MUL,
                self.ir,
                as_ir_expr(other),
            )
        )

    def __rmul__(self, other: Expr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.MUL,
                as_ir_expr(other),
                self.ir,
            )
        )

    def __truediv__(self, other: Expr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.DIV,
                self.ir,
                as_ir_expr(other),
            )
        )

    def __rtruediv__(self, other: Expr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.DIV,
                as_ir_expr(other),
                self.ir,
            )
        )


@dataclass(frozen=True, slots=True)
class FieldSymbol:
    field: Field

    def __getitem__(self, offsets: int | tuple[int, ...]) -> Expr:
        normalized = (offsets,) if isinstance(offsets, int) else offsets

        if len(normalized) != len(self.field.dims):
            raise ValueError(
                f"Field {self.field.name!r} expects "
                f"{len(self.field.dims)} offsets, got {len(normalized)}"
            )

        return Expr(
            stencil.FieldAccess(
                field=self.field,
                offsets=normalized,
            )
        )


def as_ir_expr(value: Expr | int | float) -> stencil.Expr:
    if isinstance(value, Expr):
        return value.ir

    return stencil.Literal(value)


def symbol(field: Field) -> FieldSymbol:
    return FieldSymbol(field)
