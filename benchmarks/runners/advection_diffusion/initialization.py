"""Direct advection-diffusion reference implementation matching the GT4Py benchmark."""

import numpy as np

MU = 0.1


def solution_factory(t, x, y, slice_x=None, slice_y=None):
    nx = x.shape[0]
    ny = y.shape[0]

    slice_x = slice_x or slice(0, nx)
    slice_y = slice_y or slice(0, ny)

    mi = slice_x.stop - slice_x.start
    mj = slice_y.stop - slice_y.start

    x2d = np.tile(x[slice_x, np.newaxis], (1, mj))
    y2d = np.tile(y[np.newaxis, slice_y], (mi, 1))

    u = 0.75 - 1.0 / (4.0 * (1.0 + np.exp(-t - 4.0 * x2d + 4.0 * y2d) / (32.0 * MU)))
    v = 0.75 + 1.0 / (4.0 * (1.0 + np.exp(-t - 4.0 * x2d + 4.0 * y2d) / (32.0 * MU)))

    return u, v


def set_initial_solution(x, y, u, v):
    u[...], v[...] = solution_factory(0.0, x, y)


def enforce_boundary_conditions(t, x, y, u, v):
    nx = x.shape[0]
    ny = y.shape[0]

    slice_x, slice_y = slice(0, 3), slice(0, ny)
    u[slice_x, slice_y], v[slice_x, slice_y] = solution_factory(t, x, y, slice_x, slice_y)

    slice_x, slice_y = slice(nx - 3, nx), slice(0, ny)
    u[slice_x, slice_y], v[slice_x, slice_y] = solution_factory(t, x, y, slice_x, slice_y)

    slice_x, slice_y = slice(3, nx - 3), slice(0, 3)
    u[slice_x, slice_y], v[slice_x, slice_y] = solution_factory(t, x, y, slice_x, slice_y)

    slice_x, slice_y = slice(3, nx - 3), slice(ny - 3, ny)
    u[slice_x, slice_y], v[slice_x, slice_y] = solution_factory(t, x, y, slice_x, slice_y)


def _rhs(u, v, dt, dx, dy, mu):
    u_pad = np.pad(u, 3, mode="edge")
    v_pad = np.pad(v, 3, mode="edge")

    u_c = u_pad[3:-3, 3:-3]
    v_c = v_pad[3:-3, 3:-3]

    abs_u = np.abs(u_c)
    abs_v = np.abs(v_c)

    u_p1x = u_pad[4:-2, 3:-3]
    u_m1x = u_pad[2:-4, 3:-3]
    u_p2x = u_pad[5:-1, 3:-3]
    u_m2x = u_pad[1:-5, 3:-3]
    u_p3x = u_pad[6:, 3:-3]
    u_m3x = u_pad[:-6, 3:-3]

    u_p1y = u_pad[3:-3, 4:-2]
    u_m1y = u_pad[3:-3, 2:-4]
    u_p2y = u_pad[3:-3, 5:-1]
    u_m2y = u_pad[3:-3, 1:-5]
    u_p3y = u_pad[3:-3, 6:]
    u_m3y = u_pad[3:-3, :-6]

    v_p1x = v_pad[4:-2, 3:-3]
    v_m1x = v_pad[2:-4, 3:-3]
    v_p2x = v_pad[5:-1, 3:-3]
    v_m2x = v_pad[1:-5, 3:-3]
    v_p3x = v_pad[6:, 3:-3]
    v_m3x = v_pad[:-6, 3:-3]

    v_p1y = v_pad[3:-3, 4:-2]
    v_m1y = v_pad[3:-3, 2:-4]
    v_p2y = v_pad[3:-3, 5:-1]
    v_m2y = v_pad[3:-3, 1:-5]
    v_p3y = v_pad[3:-3, 6:]
    v_m3y = v_pad[3:-3, :-6]

    adv_u_x = u_c / (60.0 * dx) * (
        45.0 * (u_p1x - u_m1x)
        - 9.0 * (u_p2x - u_m2x)
        + (u_p3x - u_m3x)
    ) - abs_u / (60.0 * dx) * (
        (u_p3x + u_m3x)
        - 6.0 * (u_p2x + u_m2x)
        + 15.0 * (u_p1x + u_m1x)
        - 20.0 * u_c
    )

    adv_u_y = v_c / (60.0 * dy) * (
        45.0 * (u_p1y - u_m1y)
        - 9.0 * (u_p2y - u_m2y)
        + (u_p3y - u_m3y)
    ) - abs_v / (60.0 * dy) * (
        (u_p3y + u_m3y)
        - 6.0 * (u_p2y + u_m2y)
        + 15.0 * (u_p1y + u_m1y)
        - 20.0 * u_c
    )

    adv_v_x = u_c / (60.0 * dx) * (
        45.0 * (v_p1x - v_m1x)
        - 9.0 * (v_p2x - v_m2x)
        + (v_p3x - v_m3x)
    ) - abs_u / (60.0 * dx) * (
        (v_p3x + v_m3x)
        - 6.0 * (v_p2x + v_m2x)
        + 15.0 * (v_p1x + v_m1x)
        - 20.0 * v_c
    )

    adv_v_y = v_c / (60.0 * dy) * (
        45.0 * (v_p1y - v_m1y)
        - 9.0 * (v_p2y - v_m2y)
        + (v_p3y - v_m3y)
    ) - abs_v / (60.0 * dy) * (
        (v_p3y + v_m3y)
        - 6.0 * (v_p2y + v_m2y)
        + 15.0 * (v_p1y + v_m1y)
        - 20.0 * v_c
    )

    diff_u_x = (
        -u_pad[1:-5, 3:-3]
        + 16.0 * u_pad[2:-4, 3:-3]
        - 30.0 * u_c
        + 16.0 * u_pad[4:-2, 3:-3]
        - u_pad[5:-1, 3:-3]
    ) / (12.0 * dx * dx)

    diff_u_y = (
        -u_pad[3:-3, 1:-5]
        + 16.0 * u_pad[3:-3, 2:-4]
        - 30.0 * u_c
        + 16.0 * u_pad[3:-3, 4:-2]
        - u_pad[3:-3, 5:-1]
    ) / (12.0 * dy * dy)

    diff_v_x = (
        -v_pad[1:-5, 3:-3]
        + 16.0 * v_pad[2:-4, 3:-3]
        - 30.0 * v_c
        + 16.0 * v_pad[4:-2, 3:-3]
        - v_pad[5:-1, 3:-3]
    ) / (12.0 * dx * dx)

    diff_v_y = (
        -v_pad[3:-3, 1:-5]
        + 16.0 * v_pad[3:-3, 2:-4]
        - 30.0 * v_c
        + 16.0 * v_pad[3:-3, 4:-2]
        - v_pad[3:-3, 5:-1]
    ) / (12.0 * dy * dy)

    rhs_u = -(adv_u_x + adv_u_y) + mu * (diff_u_x + diff_u_y)
    rhs_v = -(adv_v_x + adv_v_y) + mu * (diff_v_x + diff_v_y)

    return rhs_u, rhs_v


def numpy_reference(u, v, dt, dx, dy, mu=MU):
    """Advance one RK stage using the same stencil used by the GT4Py benchmark."""
    rhs_u, rhs_v = _rhs(u, v, dt, dx, dy, mu)
    u_new = u + dt * rhs_u
    v_new = v + dt * rhs_v
    return u_new, v_new


def make_initial(nx: int):
    """Construct the same initial condition used in the GT4Py benchmark."""
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, nx)
    u = np.zeros((nx, nx), dtype=np.float64)
    v = np.zeros((nx, nx), dtype=np.float64)
    set_initial_solution(x, y, u, v)
    return u, v
