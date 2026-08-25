import numpy as np

from yasmin.backends import NumPyBackend
from yasmin.core import Dimension, Field, float64
from yasmin.frontend import Operator, Stencil, symbol


def test_frontend_operator_with_numpy() -> None:
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

    u_data = np.arange(64, dtype=np.float64).reshape((8, 8))
    u_out = np.zeros_like(u_data)

    NumPyBackend().execute(
        operator.ir,
        fields={
            u: u_data,
            out: u_out,
        },
    )

    expected = np.zeros_like(u_data)
    expected[1:-1, 1:-1] = (
        u_data[:-2, 1:-1]
        + u_data[2:, 1:-1]
        + u_data[1:-1, :-2]
        + u_data[1:-1, 2:]
        - 4 * u_data[1:-1, 1:-1]
    )

    np.testing.assert_allclose(u_out, expected)
