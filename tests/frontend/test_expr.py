import pytest

from yasmin.core import Dimension, Field, float64
from yasmin.frontend import Expr, symbol
from yasmin.ir import stencil


def test_build_field_expression() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)
    u_symbol = symbol(u)

    laplacian = (
        u_symbol[-1, 0]
        + u_symbol[1, 0]
        + u_symbol[0, -1]
        + u_symbol[0, 1]
        - 4 * u_symbol[0, 0]
    )

    assert isinstance(laplacian, Expr)
    assert isinstance(laplacian.ir, stencil.BinaryExpr)


def test_field_access_validate_dimensionality() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)
    u_symbol = symbol(u)

    with pytest.raises(ValueError):
        _ = u_symbol[0]
