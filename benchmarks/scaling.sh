#!/bin/bash
# Scaling script for yasmin kernels.
#
# Usage: ./scaling.sh [--problem <name>] [--nruns <n>] [--max_threads <n>]
#   --problem: problem name (laplacian, advection_diffusion; default: laplacian)
#   --nruns:   number of timed runs per benchmark (default: 5)
#   --max_threads: maximum number of threads to use (default: 72)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNERS_DIR="$SCRIPT_DIR/runners"

# Parse arguments
PROBLEM="laplacian"
NRUNS=5
MAX_THREADS=72

while [[ $# -gt 0 ]]; do
    case "$1" in
        --problem)
            PROBLEM="$2"
            shift 2
            ;;
        --nruns)
            NRUNS="$2"
            shift 2
            ;;
        --max_threads)
            MAX_THREADS="$2"
            shift 2
            ;;
        *)  
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done


SIZE_PER_THREAD=$((256 * 256))
STRONG_SCALE_SIZE=$((256 * MAX_THREADS))

PROBLEM_DIR="$RUNNERS_DIR/$PROBLEM"

if [[ ! -d "$PROBLEM_DIR" ]]; then
    echo "Error: problem directory not found: $PROBLEM_DIR" >&2
    exit 1
fi

echo "=== $PROBLEM Scaling ==="
echo "Runs per size: ${NRUNS}"
echo ""

# Create output directory
mkdir -p "$SCRIPT_DIR/outputs/raw/csv"

# Compile the OpenMP C++ reference implementation
echo "Compiling OpenMP C++ reference..."
OPENMP_CPP_SOURCE="$PROBLEM_DIR/run_openmp.cpp"
OPENMP_CPP_BINARY="$PROBLEM_DIR/run_openmp"

if [[ -f "$OPENMP_CPP_SOURCE" ]]; then
    g++ -O3 -std=c++11 -fopenmp "$OPENMP_CPP_SOURCE" -o "$OPENMP_CPP_BINARY" -lm
    echo "OpenMP C++ binary compiled: $OPENMP_CPP_BINARY"
else
    echo "Warning: OpenMP C++ source not found at $OPENMP_CPP_SOURCE"
    OPENMP_CPP_BINARY=""
fi
echo ""

# Save to CSV. Always overwrite file
STRONG_PATH="$SCRIPT_DIR/outputs/raw/csv/${PROBLEM}_strong_scaling.csv"
echo "implementation,nx,threads,runtime_ms" > "$STRONG_PATH"
WEAK_PATH="$SCRIPT_DIR/outputs/raw/csv/${PROBLEM}_weak_scaling.csv"
echo "implementation,nx,threads,runtime_ms" > "$WEAK_PATH"

NTHREADS=1
while [[ $NTHREADS -le $MAX_THREADS ]]; do
    echo "--- number of threads: $NTHREADS ---"
    
    export OMP_NUM_THREADS=$NTHREADS

    WEAK_SCALE_SIZE=$(python3 - "$NTHREADS" "$SIZE_PER_THREAD" <<'PY'
import math
import sys
nthreads = int(sys.argv[1])
base = int(sys.argv[2])
print(int(round(math.sqrt(base*nthreads))))
PY
)

    # Yasmin OpenMP backend
    echo "Running yasmin (OpenMP backend)..."
    yasmin_openmp_out=$(cd "$PROBLEM_DIR" && OMP_NUM_THREADS=$NTHREADS python3 run_yasmin_openmp.py "$STRONG_SCALE_SIZE" "$NRUNS")
    echo "$yasmin_openmp_out"
    echo ""
    yasmin_openmp_runtime=$(echo "$yasmin_openmp_out" | grep "^RUNTIME_MS=" | cut -d= -f2)
    echo "yasmin_openmp,$STRONG_SCALE_SIZE,$NTHREADS,$yasmin_openmp_runtime" >> "$STRONG_PATH"

    yasmin_openmp_out=$(cd "$PROBLEM_DIR" && OMP_NUM_THREADS=$NTHREADS python3 run_yasmin_openmp.py "$WEAK_SCALE_SIZE" "$NRUNS")
    echo "$yasmin_openmp_out"
    echo ""
    yasmin_openmp_runtime=$(echo "$yasmin_openmp_out" | grep "^RUNTIME_MS=" | cut -d= -f2)
    echo "yasmin_openmp,$WEAK_SCALE_SIZE,$NTHREADS,$yasmin_openmp_runtime" >> "$WEAK_PATH"

    # OpenMP C++ reference (if available)
    if [[ -n "$OPENMP_CPP_BINARY" && -f "$OPENMP_CPP_BINARY" ]]; then
        echo "Running OpenMP C++ reference..."
        openmp_cpp_out=$(OMP_NUM_THREADS=$NTHREADS "$OPENMP_CPP_BINARY" "$STRONG_SCALE_SIZE" "$NRUNS")
        echo "$openmp_cpp_out"
        echo ""
        openmp_cpp_runtime=$(echo "$openmp_cpp_out" | grep "^RUNTIME_MS=" | cut -d= -f2)
        echo "cpp_openmp,$STRONG_SCALE_SIZE,$NTHREADS,$openmp_cpp_runtime" >> "$STRONG_PATH"

        openmp_cpp_out=$(OMP_NUM_THREADS=$NTHREADS "$OPENMP_CPP_BINARY" "$WEAK_SCALE_SIZE" "$NRUNS")
        echo "$openmp_cpp_out"
        echo ""
        openmp_cpp_runtime=$(echo "$openmp_cpp_out" | grep "^RUNTIME_MS=" | cut -d= -f2)
        echo "cpp_openmp,$WEAK_SCALE_SIZE,$NTHREADS,$openmp_cpp_runtime" >> "$WEAK_PATH"
    fi
    
    echo "Results saved to: $WEAK_PATH and $STRONG_PATH"
    echo ""
    NTHREADS=$((NTHREADS+1))
done