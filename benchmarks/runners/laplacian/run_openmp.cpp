#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <omp.h>

using Clock = std::chrono::high_resolution_clock;

std::vector<std::vector<double>> make_initial(int nx) {
    std::vector<std::vector<double>> grid(nx, std::vector<double>(nx));
    for (int i = 0; i < nx; i++) {
        double xi = std::sin(2.0 * M_PI * i / (nx - 1));
        for (int j = 0; j < nx; j++) {
            double xj = std::sin(2.0 * M_PI * j / (nx - 1));
            grid[i][j] = xi * xj;
        }
    }
    return grid;
}

std::vector<std::vector<double>> laplacian_openmp(const std::vector<std::vector<double>>& u) {
    int nx = u.size();
    std::vector<std::vector<double>> out(nx, std::vector<double>(nx, 0.0));

    #pragma omp parallel for
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < nx - 1; j++) {
                out[i][j] = u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - 4.0 * u[i][j];
            }
        }

    return out;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: run_openmp <nx> [nruns]\n";
        return 1;
    }

    int nx = std::atoi(argv[1]);
    int nruns = (argc > 2) ? std::atoi(argv[2]) : 5;

    auto u = make_initial(nx);
    auto out = laplacian_openmp(u);

    std::vector<double> times;
    for (int r = 0; r < nruns; r++) {
        auto t0 = Clock::now();
        out = laplacian_openmp(u);
        auto t1 = Clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(elapsed_ms);
    }

    double mean_runtime = 0.0;
    for (double t : times) {
        mean_runtime += t;
    }
    mean_runtime /= times.size();

    std::cout << "NX=" << nx << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "RUNTIME_MS=" << mean_runtime << "\n";

    return 0;
}
