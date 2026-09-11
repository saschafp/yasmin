#!/usr/bin/env python3
"""
Benchmark yasmin with the NumPy backend for the Laplacian.

Usage: python3 run_yasmin.py <backend> <nx> [nruns]

Outputs:
  NX=<nx>
  BUILD_TIME_S=<time to construct operator>
  GEN_TIME_S=<time to lower to IR>
  COMPILE_TIME_S=<time to compile (0 for numpy)>
  RUNTIME_MS=<mean execution time in milliseconds>
"""

import sys
import time
from pathlib import Path

# Add repo root and src to path for yasmin import
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yasmin as yasi
from initialization import make_initial


def benchmark_yasmin(backend: str, nx: int, nruns: int = 5):
    """Build and benchmark yasmin laplacian on specified backend."""
    
    # Build: construct objects
    t0 = time.perf_counter()
    
    x = yasi.Dimension("x")
    y = yasi.Dimension("y")
    u = yasi.Field("u", dims=(x, y), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x, y), dtype=yasi.float64)

    @yasi.stencil
    def laplace(f):
        return f[-1, 0] + f[1, 0] + f[0, -1] + f[0, 1] - 4.0 * f[0, 0]

    @yasi.operator
    def diffuse(u_f, out_f):
        out_f[0, 0] = laplace(u_f)

    op = diffuse(u, out)
    u_data = make_initial(nx)
    out_data = np.zeros_like(u_data)
    
    t_build = time.perf_counter() - t0
    
    # Generation: lower to IR
    t0 = time.perf_counter()
    op_ir = op._as_ir()
    t_gen = time.perf_counter() - t0
    
    # Compile time (0 for numpy)
    t_compile = 0.0
    
    # Execution: warmup + timed runs
    for _ in range(2):
        yasi.execute(
            op,
            backend=backend,
            fields={u: u_data.copy(), out: np.zeros_like(out_data)},
            scalars={},
        )
    
    times = []
    for _ in range(nruns):
        t0 = time.perf_counter()
        yasi.execute(
            op,
            backend=backend,
            fields={u: u_data.copy(), out: np.zeros_like(out_data)},
            scalars={},
        )
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # convert to ms
    
    runtime_ms = sum(times) / len(times)
    
    print(f"NX={nx}")
    print(f"BUILD_TIME_S={t_build:.6f}")
    print(f"GEN_TIME_S={t_gen:.6f}")
    print(f"COMPILE_TIME_S={t_compile:.6f}")
    print(f"RUNTIME_MS={runtime_ms:.6f}")
    
    return t_build, t_gen, t_compile, runtime_ms


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: run_yasmin.py <backend> <nx> [nruns]", file=sys.stderr)
        sys.exit(1)
    
    backend = sys.argv[1]
    nx = int(sys.argv[2])
    nruns = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    benchmark_yasmin(backend, nx, nruns)
