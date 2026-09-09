Benchmarks for yasmin

Structure:
- config/: YAML configuration files for runs
- kernels/: problem kernels (yasmin + reference runners)
- reference/: hand-written or external-tool reference implementations
- runners/: runner scripts and timing utilities
- outputs/: raw CSV/JSON and plots

Usage (starter):

Run a smoke test of the Laplacian kernel (NumPy reference):

python3 benchmarks/runners/benchmark.py --action smoke --kernel laplacian
