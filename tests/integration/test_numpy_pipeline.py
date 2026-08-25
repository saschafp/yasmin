import numpy as np

import yasmin as yasi


def test_frontend_operator_with_numpy() -> None:
    x = yasi.Dimension("x")
    y = yasi.Dimension("y")

    u = yasi.Field("u", dims=(x, y), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x, y), dtype=yasi.float64)

    laplacian = yasi.Stencil(u[-1, 0] + u[1, 0] + u[0, -1] + u[0, 1] - 4 * u[0, 0])

    operator = yasi.Operator(target=out[0, 0], value=laplacian)

    u_data = np.arange(64, dtype=np.float64).reshape((8, 8))
    u_out = np.zeros_like(u_data)

    yasi.execute(
        operator,
        backend="numpy",
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
