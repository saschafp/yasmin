"""Validate yasmin backend outputs against NumPy reference for kernels.

This script is runnable directly; it ensures the repository root and `src/`
are on `sys.path` so the local `benchmarks` package and `yasmin` (in `src/`)
can be imported without extra environment setup.
"""

from pathlib import Path
import sys

# Ensure repo root and src/ are on sys.path for direct execution
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import argparse
import numpy as np

from benchmarks.kernels import laplacian


def validate_laplacian(backend: str, nx: int, atol: float, rtol: float) -> int:
    u = laplacian.make_initial(nx)
    ref = laplacian.numpy_reference(u)

    out = laplacian.run_yasmin(backend=backend, nx=nx)

    try:
        np.testing.assert_allclose(out, ref, atol=atol, rtol=rtol)
    except AssertionError as e:
        print("Validation FAILED for backend=", backend, "nx=", nx)
        print(e)
        # report max abs error
        max_err = float(np.max(np.abs(out - ref)))
        print("max abs error:", max_err)
        return 1

    print("Validation PASSED for backend=", backend, "nx=", nx)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["laplacian"], default="laplacian")
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-10)
    args = parser.parse_args()

    if args.kernel == "laplacian":
        rc = validate_laplacian(args.backend, args.nx, args.atol, args.rtol)
        sys.exit(rc)


if __name__ == "__main__":
    main()
