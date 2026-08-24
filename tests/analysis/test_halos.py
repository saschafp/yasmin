from yasmin.analysis import infer_halo, infer_offset_bounds
from yasmin.core import Dimension, DType, Field
from yasmin.ir.stencil import BinaryExpr, BinaryOperator, FieldAccess


def test_infer_halo() -> None:
    bounds = ((-2, 1), (0, 3))

    assert infer_halo(bounds) == ((2, 1), (0, 3))


def test_asymmetric_stencil_analysis() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=DType.FLOAT64)

    expr = BinaryExpr(
        BinaryOperator.ADD,
        BinaryExpr(
            BinaryOperator.ADD,
            FieldAccess(u, (-2, 0)),
            FieldAccess(u, (1, 0)),
        ),
        FieldAccess(u, (0, 3)),
    )

    bounds = infer_offset_bounds(expr)
    halo = infer_halo(bounds[u])

    assert bounds[u] == ((-2, 1), (0, 3))
    assert halo == ((2, 1), (0, 3))
