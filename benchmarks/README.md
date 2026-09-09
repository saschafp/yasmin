Benchmarks for yasmin

Structure:
- config/: YAML configuration files for runs
- kernels/: problem kernels (yasmin + reference runners)
- reference/: hand-written or external-tool reference implementations
- runners/: runner scripts and timing utilities
- outputs/: raw CSV/JSON and plots

Usage:

Run a smoke test of the Laplacian kernel (NumPy reference):

python3 runners/benchmark.py --action smoke --kernel laplacian

Run validation of the yasi Laplace kernel:

python3 runners/validate.py --backend numpy --nx 64
