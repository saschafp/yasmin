#!/usr/bin/env python3
"""
Benchmark the direct NumPy advection-diffusion reference.

Usage: python3 run_numpy.py <nx> [nruns]

Outputs:
  NX=<nx>
  RUNTIME_MS=<mean runtime in milliseconds>
"""

import sys
import time

from initialization import (
    MU,
    enforce_boundary_conditions,
    make_initial,
    numpy_reference,
    set_initial_solution,
    solution_factory,
)


def benchmark_numpy(nx: int, nruns: int = 5):
    """Run the direct NumPy reference for the GT4Py advection-diffusion benchmark."""
    cfl = 1.0
    timestep = cfl / (nx - 1) ** 2

    x = __import__("numpy").linspace(0.0, 1.0, nx)
    y = __import__("numpy").linspace(0.0, 1.0, nx)
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (nx - 1)

    u, v = make_initial(nx)
    u = u.copy()
    v = v.copy()

    for _ in range(2):
        u_now, v_now = u.copy(), v.copy()
        for stage_weight in (1.0 / 3.0, 0.5, 1.0):
            dt = stage_weight * timestep
            u_new, v_new = numpy_reference(u_now, v_now, dt, dx, dy, MU)
            u_now, v_now = u_new, v_new
            enforce_boundary_conditions(dt, x, y, u_now, v_now)
        u, v = u_now, v_now

    times = []
    for _ in range(nruns):
        u_now, v_now = u.copy(), v.copy()
        t0 = time.perf_counter()
        for stage_weight in (1.0 / 3.0, 0.5, 1.0):
            dt = stage_weight * timestep
            u_new, v_new = numpy_reference(u_now, v_now, dt, dx, dy, MU)
            u_now, v_now = u_new, v_new
            enforce_boundary_conditions(dt, x, y, u_now, v_now)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    runtime_ms = sum(times) / len(times)
    print(f"NX={nx}")
    print(f"RUNTIME_MS={runtime_ms:.6f}")
    return runtime_ms


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_numpy.py <nx> [nruns]", file=sys.stderr)
        sys.exit(1)

    nx = int(sys.argv[1])
    nruns = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    benchmark_numpy(nx, nruns)
