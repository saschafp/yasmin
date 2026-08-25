from yasmin.core import Dimension, Field, float64
from yasmin.frontend import Stencil, symbol
from yasmin.ir import stencil as ir


def test_stencil_wraps_expression() -> None:
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

    stencil = Stencil(laplacian)

    assert stencil.expr == laplacian
    assert isinstance(stencil.expr.ir, ir.BinaryExpr)
