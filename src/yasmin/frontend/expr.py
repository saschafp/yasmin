from __future__ import annotations

from dataclasses import dataclass

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


def as_ir_expr(value: Expr | int | float) -> stencil.Expr:
    if isinstance(value, Expr):
        return value.ir

    return stencil.Literal(value)
