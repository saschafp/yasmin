#!/usr/bin/env python3
"""
Benchmark the GT4Py advection-diffusion implementation.
From examples/cartesian/demo_burgers.ipynb on gt4py GitHub.

Usage: python3 run_gt4py.py <nx> [nruns]

The grid is square, so ny = nx.

Outputs:
  NX=<nx>
  BUILD_TIME_S=<time to build stencil objects>
  GEN_TIME_S=<time to prepare storage>
  COMPILE_TIME_S=<time to first JIT / compile>
  RUNTIME_MS=<mean runtime in milliseconds>
"""

import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import gt4py.storage
import gt4py.cartesian.gtscript as gtscript

backend = "numpy"
dtype = np.float64
mu = 0.1


@gtscript.function
def absolute_value(phi):
    abs_phi = phi[0, 0, 0] * (phi[0, 0, 0] >= 0.0) - phi[0, 0, 0] * (phi[0, 0, 0] < 0.0)
    return abs_phi


@gtscript.function
def advection_x(dx, u, abs_u, phi):
    adv_phi_x = u[0, 0, 0] / (60.0 * dx) * (
        +45.0 * (phi[1, 0, 0] - phi[-1, 0, 0])
        - 9.0 * (phi[2, 0, 0] - phi[-2, 0, 0])
        + (phi[3, 0, 0] - phi[-3, 0, 0])
    ) - abs_u[0, 0, 0] / (60.0 * dx) * (
        +(phi[3, 0, 0] + phi[-3, 0, 0])
        - 6.0 * (phi[2, 0, 0] + phi[-2, 0, 0])
        + 15.0 * (phi[1, 0, 0] + phi[-1, 0, 0])
        - 20.0 * phi[0, 0, 0]
    )
    return adv_phi_x


@gtscript.function
def advection_y(dy, v, abs_v, phi):
    adv_phi_y = v[0, 0, 0] / (60.0 * dy) * (
        +45.0 * (phi[0, 1, 0] - phi[0, -1, 0])
        - 9.0 * (phi[0, 2, 0] - phi[0, -2, 0])
        + (phi[0, 3, 0] - phi[0, -3, 0])
    ) - abs_v[0, 0, 0] / (60.0 * dy) * (
        +(phi[0, 3, 0] + phi[0, -3, 0])
        - 6.0 * (phi[0, 2, 0] + phi[0, -2, 0])
        + 15.0 * (phi[0, 1, 0] + phi[0, -1, 0])
        - 20.0 * phi[0, 0, 0]
    )
    return adv_phi_y


@gtscript.function
def advection(dx, dy, u, v):
    abs_u = absolute_value(phi=u)
    abs_v = absolute_value(phi=v)

    adv_u_x = advection_x(dx=dx, u=u, abs_u=abs_u, phi=u)
    adv_u_y = advection_y(dy=dy, v=v, abs_v=abs_v, phi=u)
    adv_u = adv_u_x[0, 0, 0] + adv_u_y[0, 0, 0]

    adv_v_x = advection_x(dx=dx, u=u, abs_u=abs_u, phi=v)
    adv_v_y = advection_y(dy=dy, v=v, abs_v=abs_v, phi=v)
    adv_v = adv_v_x[0, 0, 0] + adv_v_y[0, 0, 0]

    return adv_u, adv_v


@gtscript.function
def diffusion_x(dx, phi):
    diff_phi = (
        -phi[-2, 0, 0]
        + 16.0 * phi[-1, 0, 0]
        - 30.0 * phi[0, 0, 0]
        + 16.0 * phi[1, 0, 0]
        - phi[2, 0, 0]
    ) / (12.0 * dx**2)
    return diff_phi


@gtscript.function
def diffusion_y(dy, phi):
    diff_phi = (
        -phi[0, -2, 0]
        + 16.0 * phi[0, -1, 0]
        - 30.0 * phi[0, 0, 0]
        + 16.0 * phi[0, 1, 0]
        - phi[0, 2, 0]
    ) / (12.0 * dy**2)
    return diff_phi


@gtscript.function
def diffusion(dx, dy, u, v):
    diff_u_x = diffusion_x(dx=dx, phi=u)
    diff_u_y = diffusion_y(dy=dy, phi=u)
    diff_u = diff_u_x[0, 0, 0] + diff_u_y[0, 0, 0]

    diff_v_x = diffusion_x(dx=dx, phi=v)
    diff_v_y = diffusion_y(dy=dy, phi=v)
    diff_v = diff_v_x[0, 0, 0] + diff_v_y[0, 0, 0]

    return diff_u, diff_v

# Stencils definition and compilation

# gt4py settings
origin = (3, 3, 0)
rebuild = False

externals = {
    "absolute_value": absolute_value,
    "advection_x": advection_x,
    "advection_y": advection_y,
    "advection": advection,
    "diffusion_x": diffusion_x,
    "diffusion_y": diffusion_y,
    "diffusion": diffusion,
}


@gtscript.stencil(backend=backend, externals=externals, rebuild=rebuild)
def rk_stage(
    in_u_now: gtscript.Field[dtype],
    in_v_now: gtscript.Field[dtype],
    in_u_tmp: gtscript.Field[dtype],
    in_v_tmp: gtscript.Field[dtype],
    out_u: gtscript.Field[dtype],
    out_v: gtscript.Field[dtype],
    *,
    dt: float,
    dx: float,
    dy: float,
    mu: float,
):
    with computation(PARALLEL), interval(...):
        adv_u, adv_v = advection(dx=dx, dy=dy, u=in_u_tmp, v=in_v_tmp)
        diff_u, diff_v = diffusion(dx=dx, dy=dy, u=in_u_tmp, v=in_v_tmp)
        out_u = in_u_now[0, 0, 0] + dt * (-adv_u[0, 0, 0] + mu * diff_u[0, 0, 0])
        out_v = in_v_now[0, 0, 0] + dt * (-adv_v[0, 0, 0] + mu * diff_v[0, 0, 0])


@gtscript.stencil(backend=backend)
def copy(in_phi: gtscript.Field[dtype], out_phi: gtscript.Field[dtype]):
    with computation(PARALLEL), interval(...):
        out_phi = in_phi[0, 0, 0]

# Initial and boundary conditions

def solution_factory(t, x, y, slice_x=None, slice_y=None):
    nx, ny = x.shape[0], y.shape[0]

    slice_x = slice_x or slice(0, nx)
    slice_y = slice_y or slice(0, ny)

    mi = slice_x.stop - slice_x.start
    mj = slice_y.stop - slice_y.start

    x2d = np.tile(x[slice_x, np.newaxis, np.newaxis], (1, mj, 1))
    y2d = np.tile(y[np.newaxis, slice_y, np.newaxis], (mi, 1, 1))

    u = 0.75 - 1.0 / (4.0 * (1.0 + np.exp(-t - 4.0 * x2d + 4.0 * y2d) / (32.0 * mu)))
    v = 0.75 + 1.0 / (4.0 * (1.0 + np.exp(-t - 4.0 * x2d + 4.0 * y2d) / (32.0 * mu)))

    return u, v


def set_initial_solution(x, y, u, v):
    u[...], v[...] = solution_factory(0.0, x, y)


def enforce_boundary_conditions(t, x, y, u, v):
    nx, ny = x.shape[0], y.shape[0]

    slice_x, slice_y = slice(0, 3), slice(0, ny)
    u[slice_x, slice_y], v[slice_x, slice_y] = solution_factory(t, x, y, slice_x, slice_y)

    slice_x, slice_y = slice(nx - 3, nx), slice(0, ny)
    u[slice_x, slice_y], v[slice_x, slice_y] = solution_factory(t, x, y, slice_x, slice_y)

    slice_x, slice_y = slice(3, nx - 3), slice(0, 3)
    u[slice_x, slice_y], v[slice_x, slice_y] = solution_factory(t, x, y, slice_x, slice_y)

    slice_x, slice_y = slice(3, nx - 3), slice(ny - 3, ny)
    u[slice_x, slice_y], v[slice_x, slice_y] = solution_factory(t, x, y, slice_x, slice_y)


def benchmark_gt4py(nx: int, nruns: int = 5):
    """Run the GT4Py advection-diffusion kernel on a square grid of size nx x nx."""
    ny = nx
    cfl = 1.0
    timestep = cfl / (nx - 1) ** 2
    niter = 20

    x = np.linspace(0.0, 1.0, nx)
    dx = 1.0 / (nx - 1)
    y = np.linspace(0.0, 1.0, ny)
    dy = 1.0 / (ny - 1)

    t0 = time.perf_counter()
    u_now = gt4py.storage.zeros((nx, ny, 1), dtype, backend=backend, aligned_index=origin)
    v_now = gt4py.storage.zeros((nx, ny, 1), dtype, backend=backend, aligned_index=origin)
    u_new = gt4py.storage.zeros((nx, ny, 1), dtype, backend=backend, aligned_index=origin)
    v_new = gt4py.storage.zeros((nx, ny, 1), dtype, backend=backend, aligned_index=origin)
    set_initial_solution(x, y, u_new, v_new)
    t_build = time.perf_counter() - t0

    # First call triggers GT4Py JIT / code-generation.
    t0 = time.perf_counter()
    for _ in range(1):
        copy(in_phi=u_new, out_phi=u_now, origin=(0, 0, 0), domain=(nx, ny, 1))
        copy(in_phi=v_new, out_phi=v_now, origin=(0, 0, 0), domain=(nx, ny, 1))
        for k in range(3):
            dt = (1.0 / 3.0, 0.5, 1.0)[k] * timestep
            rk_stage(
                in_u_now=u_now,
                in_v_now=v_now,
                in_u_tmp=u_new,
                in_v_tmp=v_new,
                out_u=u_new,
                out_v=v_new,
                dt=dt,
                dx=dx,
                dy=dy,
                mu=mu,
                origin=(3, 3, 0),
                domain=(nx - 6, ny - 6, 1),
            )
            enforce_boundary_conditions(dt, x, y, u_new, v_new)
    t_compile = time.perf_counter() - t0

    # Warmup after first compilation
    for _ in range(2):
        copy(in_phi=u_new, out_phi=u_now, origin=(0, 0, 0), domain=(nx, ny, 1))
        copy(in_phi=v_new, out_phi=v_now, origin=(0, 0, 0), domain=(nx, ny, 1))
        for k in range(3):
            dt = (1.0 / 3.0, 0.5, 1.0)[k] * timestep
            rk_stage(
                in_u_now=u_now,
                in_v_now=v_now,
                in_u_tmp=u_new,
                in_v_tmp=v_new,
                out_u=u_new,
                out_v=v_new,
                dt=dt,
                dx=dx,
                dy=dy,
                mu=mu,
                origin=(3, 3, 0),
                domain=(nx - 6, ny - 6, 1),
            )
            enforce_boundary_conditions(dt, x, y, u_new, v_new)

    times = []
    for _ in range(nruns):
        t0 = time.perf_counter()
        for _ in range(1):
            copy(in_phi=u_new, out_phi=u_now, origin=(0, 0, 0), domain=(nx, ny, 1))
            copy(in_phi=v_new, out_phi=v_now, origin=(0, 0, 0), domain=(nx, ny, 1))
            for k in range(3):
                dt = (1.0 / 3.0, 0.5, 1.0)[k] * timestep
                rk_stage(
                    in_u_now=u_now,
                    in_v_now=v_now,
                    in_u_tmp=u_new,
                    in_v_tmp=v_new,
                    out_u=u_new,
                    out_v=v_new,
                    dt=dt,
                    dx=dx,
                    dy=dy,
                    mu=mu,
                    origin=(3, 3, 0),
                    domain=(nx - 6, ny - 6, 1),
                )
                enforce_boundary_conditions(dt, x, y, u_new, v_new)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    runtime_ms = sum(times) / len(times)

    print(f"NX={nx}")
    print(f"BUILD_TIME_S={t_build:.6f}")
    print(f"GEN_TIME_S=0.000000")
    print(f"COMPILE_TIME_S={t_compile:.6f}")
    print(f"RUNTIME_MS={runtime_ms:.6f}")

    return t_build, 0.0, t_compile, runtime_ms


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_gt4py.py <nx> [nruns]", file=sys.stderr)
        sys.exit(1)

    nx = int(sys.argv[1])
    nruns = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    benchmark_gt4py(nx, nruns)
