import numpy as np

from yasmin.backends import NumPyBackend
from yasmin.core import Dimension, float64
from yasmin.frontend import Field, Operator, Stencil


def test_frontend_operator_with_numpy() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)
    out = Field("out", dims=(x, y), dtype=float64)

    laplacian = Stencil(u[-1, 0] + u[1, 0] + u[0, -1] + u[0, 1] - 4 * u[0, 0])

    operator = Operator(target=out[0, 0], value=laplacian)

    u_data = np.arange(64, dtype=np.float64).reshape((8, 8))
    u_out = np.zeros_like(u_data)

    NumPyBackend().execute(
        operator.ir,
        fields={
            u.core: u_data,
            out.core: u_out,
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
