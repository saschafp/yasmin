from yasmin.analysis import infer_offset_bounds
from yasmin.core import Dimension, DType, Field
from yasmin.ir.stencil import BinaryExpr, BinaryOperator, FieldAccess


def test_infer_offset_bounds() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=DType.FLOAT64)

    a = FieldAccess(u, (-2, 0))
    b = FieldAccess(u, (1, 0))
    c = FieldAccess(u, (0, 3))

    expr = BinaryExpr(
        BinaryOperator.ADD,
        BinaryExpr(BinaryOperator.ADD, a, b),
        c,
    )

    bounds = infer_offset_bounds(expr)

    assert bounds[u] == ((-2, 1), (0, 3))
