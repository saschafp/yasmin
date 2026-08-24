from yasmin.analysis import collect_accesses
from yasmin.core import Dimension, DType, Field
from yasmin.ir.stencil import BinaryExpr, BinaryOperator, FieldAccess, Literal


def test_collect_accesses() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=DType.FLOAT64)

    left = FieldAccess(u, (-1, 0))
    right = FieldAccess(u, (1, 0))

    expr = BinaryExpr(BinaryOperator.ADD, left, right)

    accesses = collect_accesses(expr)

    assert accesses == (left, right)


def test_collect_accesses_from_laplacian() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=DType.FLOAT64)

    center = FieldAccess(u, (0, 0))
    left = FieldAccess(u, (-1, 0))
    right = FieldAccess(u, (1, 0))
    up = FieldAccess(u, (0, 1))
    down = FieldAccess(u, (0, -1))

    sum_horizontal = BinaryExpr(BinaryOperator.ADD, left, right)
    sum_vertical = BinaryExpr(BinaryOperator.ADD, up, down)

    sum_all = BinaryExpr(BinaryOperator.ADD, sum_horizontal, sum_vertical)
    four_center = BinaryExpr(BinaryOperator.ADD, Literal(4.0), center)

    laplacian = BinaryExpr(BinaryOperator.SUB, sum_all, four_center)

    accesses = collect_accesses(laplacian)

    assert accesses == (left, right, up, down, center)
