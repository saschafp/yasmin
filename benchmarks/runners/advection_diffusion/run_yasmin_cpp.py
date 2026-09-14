#!/usr/bin/env python3
"""
Benchmark yasmin with the C++ backend for the advection-diffusion problem.

Usage: python3 run_yasmin_cpp.py <nx> [nruns]

The grid is square, so ny = nx.

Outputs:
  NX=<nx>
  BUILD_TIME_S=<time to construct the operator and fields>
  GEN_TIME_S=<time to lower to IR>
  COMPILE_TIME_S=<time to compile the C++ backend>
  RUNTIME_MS=<mean execution time in milliseconds>
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yasmin as yasi
from yasmin.backends import CppBackend
from yasmin.lowering import lower

mu = 0.1


def solution_factory(t, x, y, slice_x=None, slice_y=None):
    nx, ny = x.shape[0], y.shape[0]

    slice_x = slice_x or slice(0, nx)
    slice_y = slice_y or slice(0, ny)

    mi = slice_x.stop - slice_x.start
    mj = slice_y.stop - slice_y.start

    x2d = np.tile(x[slice_x, np.newaxis], (1, mj))
    y2d = np.tile(y[np.newaxis, slice_y], (mi, 1))

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


def benchmark_yasmin_cpp(nx: int, nruns: int = 5):
    """Run the same advection-diffusion problem as the GT4Py reference using the C++ backend."""
    ny = nx
    cfl = 1.0
    timestep = cfl / (nx - 1) ** 2

    x = np.linspace(0.0, 1.0, nx)
    dx = 1.0 / (nx - 1)
    y = np.linspace(0.0, 1.0, ny)
    dy = 1.0 / (ny - 1)
    dx2 = dx * dx
    dy2 = dy * dy

    t0 = time.perf_counter()
    x_dim = yasi.Dimension("x")
    y_dim = yasi.Dimension("y")

    u_now = yasi.Field("u_now", dims=(x_dim, y_dim), dtype=yasi.float64)
    v_now = yasi.Field("v_now", dims=(x_dim, y_dim), dtype=yasi.float64)
    u_new = yasi.Field("u_new", dims=(x_dim, y_dim), dtype=yasi.float64)
    v_new = yasi.Field("v_new", dims=(x_dim, y_dim), dtype=yasi.float64)
    abs_u = yasi.Field("abs_u", dims=(x_dim, y_dim), dtype=yasi.float64)
    abs_v = yasi.Field("abs_v", dims=(x_dim, y_dim), dtype=yasi.float64)

    dt_s = yasi.Scalar("dt", dtype=yasi.float64)
    dx_s = yasi.Scalar("dx", dtype=yasi.float64)
    dy_s = yasi.Scalar("dy", dtype=yasi.float64)
    dx2_s = yasi.Scalar("dx2", dtype=yasi.float64)
    dy2_s = yasi.Scalar("dy2", dtype=yasi.float64)
    mu_s = yasi.Scalar("mu", dtype=yasi.float64)

    @yasi.operator
    def copy_op(in_phi, out_phi):
        out_phi[0, 0] = in_phi[0, 0]

    @yasi.operator
    def rk_stage(in_u_now, in_v_now, in_u_tmp, in_v_tmp, abs_u_f, abs_v_f, out_u, out_v):
        adv_u_x = in_u_tmp[0, 0] / (60.0 * dx_s) * (
            45.0 * (in_u_tmp[1, 0] - in_u_tmp[-1, 0])
            - 9.0 * (in_u_tmp[2, 0] - in_u_tmp[-2, 0])
            + (in_u_tmp[3, 0] - in_u_tmp[-3, 0])
        ) - abs_u_f[0, 0] / (60.0 * dx_s) * (
            (in_u_tmp[3, 0] + in_u_tmp[-3, 0])
            - 6.0 * (in_u_tmp[2, 0] + in_u_tmp[-2, 0])
            + 15.0 * (in_u_tmp[1, 0] + in_u_tmp[-1, 0])
            - 20.0 * in_u_tmp[0, 0]
        )

        adv_u_y = in_v_tmp[0, 0] / (60.0 * dy_s) * (
            45.0 * (in_u_tmp[0, 1] - in_u_tmp[0, -1])
            - 9.0 * (in_u_tmp[0, 2] - in_u_tmp[0, -2])
            + (in_u_tmp[0, 3] - in_u_tmp[0, -3])
        ) - abs_v_f[0, 0] / (60.0 * dy_s) * (
            (in_u_tmp[0, 3] + in_u_tmp[0, -3])
            - 6.0 * (in_u_tmp[0, 2] + in_u_tmp[0, -2])
            + 15.0 * (in_u_tmp[0, 1] + in_u_tmp[0, -1])
            - 20.0 * in_u_tmp[0, 0]
        )

        adv_v_x = in_u_tmp[0, 0] / (60.0 * dx_s) * (
            45.0 * (in_v_tmp[1, 0] - in_v_tmp[-1, 0])
            - 9.0 * (in_v_tmp[2, 0] - in_v_tmp[-2, 0])
            + (in_v_tmp[3, 0] - in_v_tmp[-3, 0])
        ) - abs_u_f[0, 0] / (60.0 * dx_s) * (
            (in_v_tmp[3, 0] + in_v_tmp[-3, 0])
            - 6.0 * (in_v_tmp[2, 0] + in_v_tmp[-2, 0])
            + 15.0 * (in_v_tmp[1, 0] + in_v_tmp[-1, 0])
            - 20.0 * in_v_tmp[0, 0]
        )

        adv_v_y = in_v_tmp[0, 0] / (60.0 * dy_s) * (
            45.0 * (in_v_tmp[0, 1] - in_v_tmp[0, -1])
            - 9.0 * (in_v_tmp[0, 2] - in_v_tmp[0, -2])
            + (in_v_tmp[0, 3] - in_v_tmp[0, -3])
        ) - abs_v_f[0, 0] / (60.0 * dy_s) * (
            (in_v_tmp[0, 3] + in_v_tmp[0, -3])
            - 6.0 * (in_v_tmp[0, 2] + in_v_tmp[0, -2])
            + 15.0 * (in_v_tmp[0, 1] + in_v_tmp[0, -1])
            - 20.0 * in_v_tmp[0, 0]
        )

        adv_u = adv_u_x + adv_u_y
        adv_v = adv_v_x + adv_v_y

        diff_u_x = (
            0.0 - in_u_tmp[-2, 0]
            + 16.0 * in_u_tmp[-1, 0]
            - 30.0 * in_u_tmp[0, 0]
            + 16.0 * in_u_tmp[1, 0]
            - in_u_tmp[2, 0]
        ) / (12.0 * dx2_s)

        diff_u_y = (
            0.0 - in_u_tmp[0, -2]
            + 16.0 * in_u_tmp[0, -1]
            - 30.0 * in_u_tmp[0, 0]
            + 16.0 * in_u_tmp[0, 1]
            - in_u_tmp[0, 2]
        ) / (12.0 * dy2_s)

        diff_v_x = (
            0.0 - in_v_tmp[-2, 0]
            + 16.0 * in_v_tmp[-1, 0]
            - 30.0 * in_v_tmp[0, 0]
            + 16.0 * in_v_tmp[1, 0]
            - in_v_tmp[2, 0]
        ) / (12.0 * dx2_s)

        diff_v_y = (
            0.0 - in_v_tmp[0, -2]
            + 16.0 * in_v_tmp[0, -1]
            - 30.0 * in_v_tmp[0, 0]
            + 16.0 * in_v_tmp[0, 1]
            - in_v_tmp[0, 2]
        ) / (12.0 * dy2_s)

        diff_u = diff_u_x + diff_u_y
        diff_v = diff_v_x + diff_v_y

        out_u[0, 0] = in_u_now[0, 0] + dt_s * (0.0 - adv_u + mu_s * diff_u)
        out_v[0, 0] = in_v_now[0, 0] + dt_s * (0.0 - adv_v + mu_s * diff_v)

    copy_u_base = copy_op(u_new, u_now)
    copy_v_base = copy_op(v_new, v_now)
    rk_base = rk_stage(u_now, v_now, u_new, v_new, abs_u, abs_v, u_new, v_new)

    u_now_arr = np.zeros((nx, ny), dtype=np.float64)
    v_now_arr = np.zeros((nx, ny), dtype=np.float64)
    u_new_arr = np.zeros((nx, ny), dtype=np.float64)
    v_new_arr = np.zeros((nx, ny), dtype=np.float64)
    set_initial_solution(x, y, u_new_arr, v_new_arr)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    copy_u_ir = copy_u_base._as_ir()
    copy_v_ir = copy_v_base._as_ir()
    rk_ir = rk_base._as_ir()
    t_gen = time.perf_counter() - t0

    compile_t0 = time.perf_counter()
    CppBackend().compile(lower(operator=copy_u_ir, name="copy_u"))
    CppBackend().compile(lower(operator=copy_v_ir, name="copy_v"))
    CppBackend().compile(lower(operator=rk_ir, name="rk_stage"))
    t_compile = time.perf_counter() - compile_t0

    for _ in range(2):
        yasi.execute(copy_u_base, backend="cpp", fields={u_new: u_new_arr.copy(), u_now: np.zeros_like(u_now_arr)}, scalars={})
        yasi.execute(copy_v_base, backend="cpp", fields={v_new: v_new_arr.copy(), v_now: np.zeros_like(v_now_arr)}, scalars={})
        for k in range(3):
            dt = (1.0 / 3.0, 0.5, 1.0)[k] * timestep
            abs_u_arr = np.abs(u_new_arr)
            abs_v_arr = np.abs(v_new_arr)
            yasi.execute(
                rk_base,
                backend="cpp",
                fields={
                    u_now: u_now_arr,
                    v_now: v_now_arr,
                    u_new: u_new_arr,
                    v_new: v_new_arr,
                    abs_u: abs_u_arr,
                    abs_v: abs_v_arr,
                },
                scalars={
                    dt_s: dt,
                    dx_s: dx,
                    dy_s: dy,
                    dx2_s: dx2,
                    dy2_s: dy2,
                    mu_s: mu,
                },
            )
            enforce_boundary_conditions(dt, x, y, u_new_arr, v_new_arr)

    times = []
    for _ in range(nruns):
        t0 = time.perf_counter()
        yasi.execute(copy_u_base, backend="cpp", fields={u_new: u_new_arr.copy(), u_now: np.zeros_like(u_now_arr)}, scalars={})
        yasi.execute(copy_v_base, backend="cpp", fields={v_new: v_new_arr.copy(), v_now: np.zeros_like(v_now_arr)}, scalars={})
        for k in range(3):
            dt = (1.0 / 3.0, 0.5, 1.0)[k] * timestep
            abs_u_arr = np.abs(u_new_arr)
            abs_v_arr = np.abs(v_new_arr)
            yasi.execute(
                rk_base,
                backend="cpp",
                fields={
                    u_now: u_now_arr,
                    v_now: v_now_arr,
                    u_new: u_new_arr,
                    v_new: v_new_arr,
                    abs_u: abs_u_arr,
                    abs_v: abs_v_arr,
                },
                scalars={
                    dt_s: dt,
                    dx_s: dx,
                    dy_s: dy,
                    dx2_s: dx2,
                    dy2_s: dy2,
                    mu_s: mu,
                },
            )
            enforce_boundary_conditions(dt, x, y, u_new_arr, v_new_arr)
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
        print("Usage: run_yasmin_cpp.py <nx> [nruns]", file=sys.stderr)
        sys.exit(1)

    nx = int(sys.argv[1])
    nruns = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    benchmark_yasmin_cpp(nx, nruns)
