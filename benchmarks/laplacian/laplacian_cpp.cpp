#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "../cpp_utils.hpp"

using Clock = std::chrono::high_resolution_clock;

void make_initial(double* grid, int nx) {
    const double pi = std::acos(-1.0);

    for (int i = 0; i < nx; ++i) {
        const double x = std::sin(2.0 * pi * (double)i / (double)(nx - 1));
        for (int j = 0; j < nx; ++j) {
            const double y = std::sin(2.0 * pi * (double)j / (double)(nx - 1));
            grid[i * nx + j] = x * y;
        }
    }
}

void laplacian(const double* u, double* out, int nx) {
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < nx - 1; ++j) {
            const int center = i * nx + j;
            out[center] =
                u[(i - 1) * nx + j]
                + u[(i + 1) * nx + j]
                + u[i * nx + (j - 1)]
                + u[i * nx + (j + 1)]
                - 4.0 * u[center];
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: laplacian <nx> <warmups> <repeats> <output_path>\n";
        return 1;
    }

    const int nx = std::atoi(argv[1]);
    const int warmups = std::atoi(argv[2]);
    const int repeats = std::atoi(argv[3]);

    if (nx < 2 || warmups < 0 || repeats < 1) {
        std::cerr << "Expected nx >= 2, warmups >= 0, repeats >= 1\n";
        return 1;
    }

    double* u = (double*)std::malloc(nx * nx * sizeof(double));
    double* out = (double*)std::malloc(nx * nx * sizeof(double));
    double* timings = (double*)std::malloc((std::size_t)repeats * sizeof(double));

    if (u == nullptr || out == nullptr || timings == nullptr) {
        std::cerr << "Failed to allocate buffers\n";
        return 1;
    }

    make_initial(u, nx);
    for (int k = 0; k < nx * nx; ++k) {
        out[k] = 0.0;
    }
    for (int run = 0; run < warmups; ++run) {
        laplacian(u, out, nx);
    }

    for (int run = 0; run < repeats; ++run) {
        const auto start = Clock::now();
        laplacian(u, out, nx);
        const auto end = Clock::now();

        timings[run] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    if (!benchmark::write_array(argv[4], out, nx * nx)) {
        std::cerr << "Failed to write output array\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(12)
              << "NX=" << nx << "\n"
              << "RUNTIME_MS=" << benchmark::median(timings, repeats) << "\n";

    std::free(u);
    std::free(out);
    std::free(timings);

    return 0;
}
