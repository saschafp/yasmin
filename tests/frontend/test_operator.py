import pytest

from yasmin.core import Dimension, float64
from yasmin.frontend import Field, Operator, Stencil
from yasmin.ir import stencil as ir


def test_operator_builds_stencil_ir() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)
    out = Field("out", dims=(x, y), dtype=float64)

    laplacian = Stencil(u[-1, 0] + u[1, 0] + u[0, -1] + u[0, 1] - 4 * u[0, 0])

    operator = Operator(target=out[0, 0], value=laplacian)

    assert isinstance(operator._as_ir(), ir.Operator)
    assert len(operator._as_ir().statements) == 1


def test_operator_target_must_be_field_access() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)

    target = u[0] + 1.0
    value = Stencil(u[0])

    with pytest.raises(TypeError):
        Operator(target=target, value=value)
