#!/usr/bin/env python3
"""
Benchmark the NumPy reference implementation of the Laplacian.

Usage: python3 run_numpy.py <nx> <nruns>

Outputs:
  NX=<nx>
  RUNTIME_MS=<mean runtime in milliseconds>
"""

import sys
import time
from pathlib import Path

from initialization import make_initial, numpy_reference


def benchmark_numpy(nx: int, nruns: int = 5):
    """Run numpy reference and measure time."""
    u = make_initial(nx)
    
    # Warmup
    numpy_reference(u)
    
    # Timed runs
    times = []
    for _ in range(nruns):
        t0 = time.perf_counter()
        out = numpy_reference(u)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # convert to ms
    
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
