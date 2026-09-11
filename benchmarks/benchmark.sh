#!/bin/bash
# Benchmark orchestrator for yasmin kernels.
#
# Runs all grid sizes specified in docs/benchmarking.md for the selected problem.
#
# Usage: ./benchmark.sh [--problem <name>] [--nruns <n>]
#   --problem: problem name (laplacian, advection_diffusion; default: laplacian)
#   --nruns:   number of timed runs per benchmark (default: 5, per spec)
#
# Example:
#   ./benchmark.sh
#   ./benchmark.sh --problem laplacian --nruns 5
#   ./benchmark.sh --problem advection_diffusion

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNERS_DIR="$SCRIPT_DIR/runners"

# Grid sizes
declare -a LAPLACIAN_SIZES=(64 128 256 512 1024 2048)
declare -a ADVECTION_DIFFUSION_SIZES=(64 128 256 512 1024 2048)

# Parse arguments
PROBLEM="laplacian"
NRUNS=5

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
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Determine sizes for this problem
case "$PROBLEM" in
    laplacian)
        SIZES=("${LAPLACIAN_SIZES[@]}")
        ;;
    advection_diffusion)
        SIZES=("${ADVECTION_DIFFUSION_SIZES[@]}")
        ;;
    *)
        echo "Error: unknown problem: $PROBLEM" >&2
        exit 1
        ;;
esac

PROBLEM_DIR="$RUNNERS_DIR/$PROBLEM"

if [[ ! -d "$PROBLEM_DIR" ]]; then
    echo "Error: problem directory not found: $PROBLEM_DIR" >&2
    exit 1
fi

echo "=== $PROBLEM Benchmark ==="
echo "Grid sizes: ${SIZES[*]}"
echo "Runs per size: ${NRUNS}"
echo ""

# Create output directory
mkdir -p "$SCRIPT_DIR/outputs/raw/csv"

# Compile C++ reference if benchmarking laplacian
CPP_BINARY=""
if [[ "$PROBLEM" == "laplacian" ]]; then
    echo "Compiling C++ reference..."
    CPP_SOURCE="$PROBLEM_DIR/run_cpp.cpp"
    CPP_BINARY="$PROBLEM_DIR/run_cpp"
    
    if [[ -f "$CPP_SOURCE" ]]; then
        g++ -O3 -std=c++11 "$CPP_SOURCE" -o "$CPP_BINARY" -lm
        echo "C++ binary compiled: $CPP_BINARY"
    else
        echo "Warning: C++ source not found at $CPP_SOURCE"
        CPP_BINARY=""
    fi
    echo ""
fi

# Run benchmarks for each size
for NX in "${SIZES[@]}"; do
    echo "--- Grid size: ${NX}x${NX} ---"

    # Save to CSV. Always overwrite file
    CSV_PATH="$SCRIPT_DIR/outputs/raw/csv/${PROBLEM}_${NX}.csv"
    echo "implementation,nx,build_s,gen_s,compile_s,runtime_ms" > "$CSV_PATH"
    
    # Yasmin numpy backend
    echo "Running yasmin (NumPy backend)..."
    yasmin_out=$(cd "$PROBLEM_DIR" && python3 run_yasmin_np.py "numpy" "$NX" "$NRUNS")
    echo "$yasmin_out"
    echo ""
    yasmin_build=$(echo "$yasmin_out" | grep "^BUILD_TIME_S=" | cut -d= -f2)
    yasmin_gen=$(echo "$yasmin_out" | grep "^GEN_TIME_S=" | cut -d= -f2)
    yasmin_compile=$(echo "$yasmin_out" | grep "^COMPILE_TIME_S=" | cut -d= -f2)
    yasmin_runtime=$(echo "$yasmin_out" | grep "^RUNTIME_MS=" | cut -d= -f2)
    echo "yasmin_numpy,$NX,$yasmin_build,$yasmin_gen,$yasmin_compile,$yasmin_runtime" >> "$CSV_PATH"

    # Yasmin C++ backend
    echo "Running yasmin (C++ backend)..."
    yasmin_cpp_out=$(cd "$PROBLEM_DIR" && python3 run_yasmin_cpp.py "$NX" "$NRUNS")
    echo "$yasmin_cpp_out"
    echo ""
    yasmin_cpp_build=$(echo "$yasmin_cpp_out" | grep "^BUILD_TIME_S=" | cut -d= -f2)
    yasmin_cpp_gen=$(echo "$yasmin_cpp_out" | grep "^GEN_TIME_S=" | cut -d= -f2)
    yasmin_cpp_compile=$(echo "$yasmin_cpp_out" | grep "^COMPILE_TIME_S=" | cut -d= -f2)
    yasmin_cpp_runtime=$(echo "$yasmin_cpp_out" | grep "^RUNTIME_MS=" | cut -d= -f2)
    echo "yasmin_cpp,$NX,$yasmin_cpp_build,$yasmin_cpp_gen,$yasmin_cpp_compile,$yasmin_cpp_runtime" >> "$CSV_PATH"
    
    # NumPy
    echo "Running NumPy reference..."
    numpy_out=$(cd "$PROBLEM_DIR" && python3 run_numpy.py "$NX" "$NRUNS")
    echo "$numpy_out"
    echo ""
    numpy_runtime=$(echo "$numpy_out" | grep "^RUNTIME_MS=" | cut -d= -f2)
    echo "numpy,$NX,0,0,0,$numpy_runtime" >> "$CSV_PATH"
    
    # C++ reference (if available)
    if [[ -n "$CPP_BINARY" && -f "$CPP_BINARY" ]]; then
        echo "Running C++ reference..."
        cpp_out=$("$CPP_BINARY" "$NX" "$NRUNS")
        echo "$cpp_out"
        echo ""
        cpp_runtime=$(echo "$cpp_out" | grep "^RUNTIME_MS=" | cut -d= -f2)
        echo "cpp,$NX,0,0,0,$cpp_runtime" >> "$CSV_PATH"
    fi
    
    echo "Results saved to: $CSV_PATH"
    echo ""
done

echo "=== Benchmark Complete ==="
