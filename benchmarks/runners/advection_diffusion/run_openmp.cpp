#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <utility>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

constexpr double MU = 0.1;

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> solution_factory(
    double t,
    const std::vector<double>& x,
    const std::vector<double>& y,
    std::size_t x0 = 0,
    std::size_t x1 = std::numeric_limits<std::size_t>::max(),
    std::size_t y0 = 0,
    std::size_t y1 = std::numeric_limits<std::size_t>::max()) {
    std::size_t nx = x.size();
    std::size_t ny = y.size();
    if (x1 == std::numeric_limits<std::size_t>::max()) x1 = nx;
    if (y1 == std::numeric_limits<std::size_t>::max()) y1 = ny;

    std::size_t mi = x1 - x0;
    std::size_t mj = y1 - y0;

    std::vector<std::vector<double>> u(mi, std::vector<double>(mj));
    std::vector<std::vector<double>> v(mi, std::vector<double>(mj));

    for (std::size_t i = 0; i < mi; ++i) {
        for (std::size_t j = 0; j < mj; ++j) {
            double xx = x[x0 + i];
            double yy = y[y0 + j];
            double arg = -t - 4.0 * xx + 4.0 * yy;
            double denom = 1.0 + std::exp(arg) / (32.0 * MU);
            u[i][j] = 0.75 - 1.0 / (4.0 * denom);
            v[i][j] = 0.75 + 1.0 / (4.0 * denom);
        }
    }
    return std::make_pair(u, v);
}

void set_initial_solution(const std::vector<double>& x, const std::vector<double>& y,
                          std::vector<std::vector<double>>& u,
                          std::vector<std::vector<double>>& v) {
    auto sol = solution_factory(0.0, x, y);
    u = std::move(sol.first);
    v = std::move(sol.second);
}

void enforce_boundary_conditions(double t,
                                const std::vector<double>& x,
                                const std::vector<double>& y,
                                std::vector<std::vector<double>>& u,
                                std::vector<std::vector<double>>& v) {
    std::size_t nx = x.size();
    std::size_t ny = y.size();

    auto set_slice = [&](std::size_t i0, std::size_t i1, std::size_t j0, std::size_t j1) {
        auto sol = solution_factory(t, x, y, i0, i1, j0, j1);
        const auto& u0 = sol.first;
        const auto& v0 = sol.second;
        for (std::size_t i = i0; i < i1; ++i) {
            for (std::size_t j = j0; j < j1; ++j) {
                u[i][j] = u0[i - i0][j - j0];
                v[i][j] = v0[i - i0][j - j0];
            }
        }
    };

    set_slice(0, 3, 0, ny);
    set_slice(nx - 3, nx, 0, ny);
    set_slice(3, nx - 3, 0, 3);
    set_slice(3, nx - 3, ny - 3, ny);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> rhs(
    const std::vector<std::vector<double>>& u,
    const std::vector<std::vector<double>>& v,
    double dx,
    double dy,
    double mu) {
    std::size_t nx = u.size();
    std::size_t ny = u[0].size();
    std::vector<std::vector<double>> u_pad(nx + 6, std::vector<double>(ny + 6, 0.0));
    std::vector<std::vector<double>> v_pad(nx + 6, std::vector<double>(ny + 6, 0.0));

    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            u_pad[i + 3][j + 3] = u[i][j];
            v_pad[i + 3][j + 3] = v[i][j];
        }
    }
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            u_pad[i + 3][j + 3] = u[i][j];
            v_pad[i + 3][j + 3] = v[i][j];
            u_pad[i + 3][0] = u[i][0];
            u_pad[i + 3][1] = u[i][0];
            u_pad[i + 3][2] = u[i][0];
            u_pad[i + 3][ny + 3] = u[i][ny - 1];
            u_pad[i + 3][ny + 4] = u[i][ny - 1];
            u_pad[i + 3][ny + 5] = u[i][ny - 1];
        }
    }
    for (std::size_t j = 0; j < ny + 6; ++j) {
        for (std::size_t i = 0; i < 3; ++i) {
            u_pad[i][j] = u[0][std::max<int>(0, j - 3)];
            u_pad[nx + 3 + i][j] = u[nx - 1][std::max<int>(0, j - 3)];
            v_pad[i][j] = v[0][std::max<int>(0, j - 3)];
            v_pad[nx + 3 + i][j] = v[nx - 1][std::max<int>(0, j - 3)];
        }
    }

    auto ABS = [](double x) { return std::fabs(x); };
    std::vector<std::vector<double>> rhs_u(nx, std::vector<double>(ny, 0.0));
    std::vector<std::vector<double>> rhs_v(nx, std::vector<double>(ny, 0.0));

#pragma omp parallel for collapse(2)
    for (std::size_t i = 3; i < nx + 3; ++i) {
        for (std::size_t j = 3; j < ny + 3; ++j) {
            double u_c = u_pad[i][j];
            double v_c = v_pad[i][j];
            double abs_u = ABS(u_c);
            double abs_v = ABS(v_c);

            double u_p1x = u_pad[i + 1][j];
            double u_m1x = u_pad[i - 1][j];
            double u_p2x = u_pad[i + 2][j];
            double u_m2x = u_pad[i - 2][j];
            double u_p3x = u_pad[i + 3][j];
            double u_m3x = u_pad[i - 3][j];

            double u_p1y = u_pad[i][j + 1];
            double u_m1y = u_pad[i][j - 1];
            double u_p2y = u_pad[i][j + 2];
            double u_m2y = u_pad[i][j - 2];
            double u_p3y = u_pad[i][j + 3];
            double u_m3y = u_pad[i][j - 3];

            double v_p1x = v_pad[i + 1][j];
            double v_m1x = v_pad[i - 1][j];
            double v_p2x = v_pad[i + 2][j];
            double v_m2x = v_pad[i - 2][j];
            double v_p3x = v_pad[i + 3][j];
            double v_m3x = v_pad[i - 3][j];

            double v_p1y = v_pad[i][j + 1];
            double v_m1y = v_pad[i][j - 1];
            double v_p2y = v_pad[i][j + 2];
            double v_m2y = v_pad[i][j - 2];
            double v_p3y = v_pad[i][j + 3];
            double v_m3y = v_pad[i][j - 3];

            double adv_u_x = u_c / (60.0 * dx) * (
                45.0 * (u_p1x - u_m1x)
                - 9.0 * (u_p2x - u_m2x)
                + (u_p3x - u_m3x)
            ) - abs_u / (60.0 * dx) * (
                (u_p3x + u_m3x)
                - 6.0 * (u_p2x + u_m2x)
                + 15.0 * (u_p1x + u_m1x)
                - 20.0 * u_c
            );

            double adv_u_y = v_c / (60.0 * dy) * (
                45.0 * (u_p1y - u_m1y)
                - 9.0 * (u_p2y - u_m2y)
                + (u_p3y - u_m3y)
            ) - abs_v / (60.0 * dy) * (
                (u_p3y + u_m3y)
                - 6.0 * (u_p2y + u_m2y)
                + 15.0 * (u_p1y + u_m1y)
                - 20.0 * u_c
            );

            double adv_v_x = u_c / (60.0 * dx) * (
                45.0 * (v_p1x - v_m1x)
                - 9.0 * (v_p2x - v_m2x)
                + (v_p3x - v_m3x)
            ) - abs_u / (60.0 * dx) * (
                (v_p3x + v_m3x)
                - 6.0 * (v_p2x + v_m2x)
                + 15.0 * (v_p1x + v_m1x)
                - 20.0 * v_c
            );

            double adv_v_y = v_c / (60.0 * dy) * (
                45.0 * (v_p1y - v_m1y)
                - 9.0 * (v_p2y - v_m2y)
                + (v_p3y - v_m3y)
            ) - abs_v / (60.0 * dy) * (
                (v_p3y + v_m3y)
                - 6.0 * (v_p2y + v_m2y)
                + 15.0 * (v_p1y + v_m1y)
                - 20.0 * v_c
            );

            double diff_u_x = (
                -u_pad[i - 3][j]
                + 16.0 * u_pad[i - 2][j]
                - 30.0 * u_c
                + 16.0 * u_pad[i + 1][j]
                - u_pad[i + 2][j]
            ) / (12.0 * dx * dx);

            double diff_u_y = (
                -u_pad[i][j - 3]
                + 16.0 * u_pad[i][j - 2]
                - 30.0 * u_c
                + 16.0 * u_pad[i][j + 1]
                - u_pad[i][j + 2]
            ) / (12.0 * dy * dy);

            double diff_v_x = (
                -v_pad[i - 3][j]
                + 16.0 * v_pad[i - 2][j]
                - 30.0 * v_c
                + 16.0 * v_pad[i + 1][j]
                - v_pad[i + 2][j]
            ) / (12.0 * dx * dx);

            double diff_v_y = (
                -v_pad[i][j - 3]
                + 16.0 * v_pad[i][j - 2]
                - 30.0 * v_c
                + 16.0 * v_pad[i][j + 1]
                - v_pad[i][j + 2]
            ) / (12.0 * dy * dy);

            std::size_t ii = i - 3;
            std::size_t jj = j - 3;
            rhs_u[ii][jj] = -(adv_u_x + adv_u_y) + mu * (diff_u_x + diff_u_y);
            rhs_v[ii][jj] = -(adv_v_x + adv_v_y) + mu * (diff_v_x + diff_v_y);
        }
    }

    return std::make_pair(rhs_u, rhs_v);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> step(
    const std::vector<std::vector<double>>& u,
    const std::vector<std::vector<double>>& v,
    double dt,
    double dx,
    double dy,
    double mu) {
    auto rhs_pair = rhs(u, v, dx, dy, mu);
    const auto& du = rhs_pair.first;
    const auto& dv = rhs_pair.second;
    std::vector<std::vector<double>> u_new = u;
    std::vector<std::vector<double>> v_new = v;
#pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < u.size(); ++i) {
        for (std::size_t j = 0; j < u[0].size(); ++j) {
            u_new[i][j] = u[i][j] + dt * du[i][j];
            v_new[i][j] = v[i][j] + dt * dv[i][j];
        }
    }
    return std::make_pair(u_new, v_new);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: advection_diffusion <nx> [nruns]\n";
        return 1;
    }

    int nx = std::atoi(argv[1]);
    int nruns = (argc > 2) ? std::atoi(argv[2]) : 5;
    double cfl = 1.0;
    double timestep = cfl / std::pow(nx - 1, 2);
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (nx - 1);

    std::vector<double> x(nx), y(nx);
    for (int i = 0; i < nx; ++i) {
        x[i] = static_cast<double>(i) / (nx - 1);
        y[i] = static_cast<double>(i) / (nx - 1);
    }

    std::vector<std::vector<double>> u(nx, std::vector<double>(nx, 0.0));
    std::vector<std::vector<double>> v(nx, std::vector<double>(nx, 0.0));
    set_initial_solution(x, y, u, v);

    for (int r = 0; r < 2; ++r) {
        auto u_now = u;
        auto v_now = v;
        for (int stage = 0; stage < 3; ++stage) {
            double dt = ((stage == 0) ? (1.0 / 3.0) : ((stage == 1) ? 0.5 : 1.0)) * timestep;
            auto step_pair = step(u_now, v_now, dt, dx, dy, MU);
            u_now = step_pair.first;
            v_now = step_pair.second;
            enforce_boundary_conditions(dt, x, y, u_now, v_now);
        }
        u = u_now;
        v = v_now;
    }

    std::vector<double> times;
    for (int r = 0; r < nruns; ++r) {
        auto u_now = u;
        auto v_now = v;
        auto t0 = Clock::now();
        for (int stage = 0; stage < 3; ++stage) {
            double dt = ((stage == 0) ? (1.0 / 3.0) : ((stage == 1) ? 0.5 : 1.0)) * timestep;
            auto step_pair = step(u_now, v_now, dt, dx, dy, MU);
            u_now = step_pair.first;
            v_now = step_pair.second;
            enforce_boundary_conditions(dt, x, y, u_now, v_now);
        }
        auto t1 = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double mean = 0.0;
    for (double t : times) mean += t;
    mean /= static_cast<double>(times.size());

    std::cout << "NX=" << nx << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "RUNTIME_MS=" << mean << "\n";
    return 0;
}
