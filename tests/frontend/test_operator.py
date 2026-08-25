import pytest

from yasmin.core import Dimension, Field, float64
from yasmin.frontend import Operator, Stencil, symbol
from yasmin.ir import stencil as ir


def test_operator_builds_stencil_ir() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)
    out = Field("out", dims=(x, y), dtype=float64)

    u_symbol = symbol(u)
    out_symbol = symbol(out)

    laplacian = Stencil(
        u_symbol[-1, 0]
        + u_symbol[1, 0]
        + u_symbol[0, -1]
        + u_symbol[0, 1]
        - 4 * u_symbol[0, 0]
    )

    operator = Operator(target=out_symbol[0, 0], value=laplacian)

    assert isinstance(operator.ir, ir.Operator)
    assert len(operator.ir.statements) == 1


def test_operator_target_must_be_field_access() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    u_symbol = symbol(u)

    target = u_symbol[0] + 1.0
    value = Stencil(u_symbol[0])

    with pytest.raises(TypeError):
        Operator(target=target, value=value)
