from dataclasses import dataclass
from enum import Enum

from yasmin.core import Field, Scalar


class Expr:
    pass


@dataclass(frozen=True, slots=True)
class Literal(Expr):
    value: float | int


@dataclass(frozen=True, slots=True)
class ScalarRef(Expr):
    scalar: Scalar


@dataclass(frozen=True, slots=True)
class FieldAccess(Expr):
    field: Field
    offsets: tuple[int, ...]


class BinaryOperator(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


@dataclass(frozen=True, slots=True)
class BinaryExpr(Expr):
    op: BinaryOperator
    lhs: Expr
    rhs: Expr
