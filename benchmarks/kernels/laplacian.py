"""Laplacian benchmark kernel templates.

Provides:
- make_initial(nx): returns initial field (numpy array)
- numpy_reference(u): returns laplacian applied to u (interior only)
- run_smoke(): minimal smoke runner used by the benchmark script
"""

import math
import numpy as np


def make_initial(nx: int) -> np.ndarray:
    x = np.arange(nx)
    xv = np.sin(2 * math.pi * x / (nx - 1))
    grid = np.outer(xv, xv)
    return grid.astype(np.float64)


def numpy_reference(u: np.ndarray) -> np.ndarray:
    out = np.zeros_like(u)
    out[1:-1, 1:-1] = (
        u[:-2, 1:-1]
        + u[2:, 1:-1]
        + u[1:-1, :-2]
        + u[1:-1, 2:]
        - 4.0 * u[1:-1, 1:-1]
    )
    return out


def run_smoke():
    nx = 64
    u = make_initial(nx)
    out = numpy_reference(u)
    print("smoke: laplacian produced array with shape", out.shape)


if __name__ == "__main__":
    run_smoke()


# TODO: check correctness of implementation, dimensions (change to 3D?)
def run_yasmin(backend: str, nx: int):
    """Run the Laplacian using yasmin on the specified backend.

    Returns the output array produced by the backend (numpy array).
    """
    import numpy as _np
    import yasmin as yasi

    x = yasi.Dimension("x")
    y = yasi.Dimension("y")

    u = yasi.Field("u", dims=(x, y), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x, y), dtype=yasi.float64)

    @yasi.stencil
    def laplace(f):
        return f[-1, 0] + f[1, 0] + f[0, -1] + f[0, 1] - 4.0 * f[0, 0]

    @yasi.operator
    def diffuse(u_f, out_f):
        out_f[0, 0] = laplace(u_f)

    op = diffuse(u, out)

    u_data = make_initial(nx)
    out_data = _np.zeros_like(u_data)

    # execute fills out_data in-place
    yasi.execute(
        op,
        backend=backend,
        fields={
            u: u_data,
            out: out_data,
        },
        scalars={},
    )

    return out_data
