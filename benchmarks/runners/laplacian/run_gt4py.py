#!/usr/bin/env python3
"""
Benchmark the gt4py reference implementation of the Laplacian.
From gt4py/examples/cartesian/demo_horizontal_diffusion.ipynb on GitHub

Usage: python3 run_gt4py.py <nx> [nruns]

Outputs:
  NX=<nx>
  BUILD_TIME_S=<time to define the stencil>
  GEN_TIME_S=<time to prepare storages / emit a call>
  COMPILE_TIME_S=<time to first GT4Py compilation / first call>
  RUNTIME_MS=<mean runtime in milliseconds>
"""

import sys
import time
import numpy as np
import gt4py.storage
import gt4py.cartesian.gtscript as gtscript

from initialization import make_initial

backend = "numpy"
dtype = np.float64


def make_lap_stencil():
    @gtscript.stencil(backend=backend)
    def lap(
        u: gtscript.Field[dtype],
        out: gtscript.Field[dtype],
    ):
        with computation(PARALLEL), interval(...):
            out = (
                u[1, 0, 0]
                + u[-1, 0, 0]
                + u[0, 1, 0]
                + u[0, -1, 0]
                - 4.0 * u[0, 0, 0]
            )

    return lap


def gt4py_laplacian(u, lap):
    """Compute the 2D Laplacian stencil via GT4Py in a 3D Cartesian layout."""
    u3d = u[:, :, None]
    out3d = np.zeros_like(u3d)
    origin = (1, 1, 0)

    t0 = time.perf_counter()
    in_storage = gt4py.storage.from_array(u3d, dtype, backend=backend, aligned_index=origin)
    out_storage = gt4py.storage.from_array(out3d, dtype, backend=backend, aligned_index=origin)
    t_gen = time.perf_counter() - t0

    t0 = time.perf_counter()
    lap(in_storage, out_storage, origin=origin)
    t_compile = time.perf_counter() - t0

    return t_gen, t_compile


def benchmark_gt4py(nx: int, nruns: int = 5):
    """Run gt4py implementation and measure time."""
    u = make_initial(nx)

    t0 = time.perf_counter()
    lap = make_lap_stencil()
    t_build = time.perf_counter() - t0

    # First call triggers GT4Py JIT / code generation
    t_gen, t_compile = gt4py_laplacian(u, lap)

    # Warmup after first compilation
    for _ in range(2):
        gt4py_laplacian(u, lap)

    times = []
    for _ in range(nruns):
        t0 = time.perf_counter()
        gt4py_laplacian(u, lap)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    runtime_ms = sum(times) / len(times)

    print(f"NX={nx}")
    print(f"BUILD_TIME_S={t_build:.6f}")
    print(f"GEN_TIME_S={t_gen:.6f}")
    print(f"COMPILE_TIME_S={t_compile:.6f}")
    print(f"RUNTIME_MS={runtime_ms:.6f}")

    return t_build, t_gen, t_compile, runtime_ms


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_gt4py.py <nx> [nruns]", file=sys.stderr)
        sys.exit(1)

    nx = int(sys.argv[1])
    nruns = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    benchmark_gt4py(nx, nruns)
