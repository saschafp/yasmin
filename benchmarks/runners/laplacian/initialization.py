"""Laplacian kernel definitions."""

import math
import numpy as np


def make_initial(nx: int) -> np.ndarray:
    """Generate deterministic 2D sine grid."""
    x = np.arange(nx)
    xv = np.sin(2 * math.pi * x / (nx - 1))
    grid = np.outer(xv, xv)
    return grid.astype(np.float64)


def numpy_reference(u: np.ndarray) -> np.ndarray:
    """Compute laplacian stencil on interior points only."""
    out = np.zeros_like(u)
    out[1:-1, 1:-1] = (
        u[:-2, 1:-1]
        + u[2:, 1:-1]
        + u[1:-1, :-2]
        + u[1:-1, 2:]
        - 4.0 * u[1:-1, 1:-1]
    )
    return out
