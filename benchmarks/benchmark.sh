#!/bin/bash
# Benchmark orchestrator for yasmin kernels.
#
# Usage: ./benchmark.sh [--problem <name>] <nx> [nruns]
#   --problem: problem name (laplacian, advection_diffusion; default: laplacian)
#   nx:        grid size (e.g., 256)
#   nruns:     number of timed runs per benchmark (default: 5)
#
# Example:
#   ./benchmark.sh 64 3
#   ./benchmark.sh --problem advection_diffusion 128 5

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNERS_DIR="$SCRIPT_DIR/runners"

# Parse arguments
PROBLEM="laplacian"
NX="256"
NRUNS="5"

if [[ "$1" == "--problem" ]]; then
    PROBLEM="$2"
    shift 2
fi

if [[ -n "$1" ]]; then
    NX="$1"
fi

if [[ -n "$2" ]]; then
    NRUNS="$2"
fi

PROBLEM_DIR="$RUNNERS_DIR/$PROBLEM"

if [[ ! -d "$PROBLEM_DIR" ]]; then
    echo "Error: problem directory not found: $PROBLEM_DIR" >&2
    echo "Available problems: $(ls -1 "$RUNNERS_DIR" | grep -v '^[^a-z]' | tr '\n' ', ')" >&2
    exit 1
fi

echo "=== $PROBLEM Benchmark ==="
echo "Grid size: ${NX}x${NX}"
echo "Runs per benchmark: ${NRUNS}"
echo ""

# Create output directory
mkdir -p "$SCRIPT_DIR/outputs/raw/csv"

# Run benchmarks and collect output
echo "Running NumPy reference..."
numpy_out=$(cd "$PROBLEM_DIR" && python3 run_numpy.py "$NX" "$NRUNS")
echo "$numpy_out"
echo ""

echo "Running yasmin (NumPy backend)..."
yasmin_out=$(cd "$PROBLEM_DIR" && python3 run_yasmin.py "numpy" "$NX" "$NRUNS")
echo "$yasmin_out"
echo ""

# Save to CSV
CSV_PATH="$SCRIPT_DIR/outputs/raw/csv/${PROBLEM}_${NX}.csv"
if [ ! -f "$CSV_PATH" ]; then
    echo "implementation,nx,build_s,gen_s,compile_s,runtime_ms" > "$CSV_PATH"
fi

numpy_runtime=$(echo "$numpy_out" | grep "^RUNTIME_MS=" | cut -d= -f2)
yasmin_build=$(echo "$yasmin_out" | grep "^BUILD_TIME_S=" | cut -d= -f2)
yasmin_gen=$(echo "$yasmin_out" | grep "^GEN_TIME_S=" | cut -d= -f2)
yasmin_compile=$(echo "$yasmin_out" | grep "^COMPILE_TIME_S=" | cut -d= -f2)
yasmin_runtime=$(echo "$yasmin_out" | grep "^RUNTIME_MS=" | cut -d= -f2)

echo "numpy,$NX,0,0,0,$numpy_runtime" >> "$CSV_PATH"
echo "yasmin,$NX,$yasmin_build,$yasmin_gen,$yasmin_compile,$yasmin_runtime" >> "$CSV_PATH"

echo "Results saved to: $CSV_PATH"
