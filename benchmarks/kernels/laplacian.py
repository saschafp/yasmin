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
