#!/usr/bin/env bash

set -euo pipefail

out="benchmarks/outputs/report"

export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

rm -rf "$out"
mkdir -p "$out"

echo "Writing results to: $out"
echo "Commit: $(git rev-parse HEAD)"
echo "Python: $(python --version 2>&1)"
echo "Compiler: $(g++ --version | head -n 1)"
echo "OMP_DYNAMIC=$OMP_DYNAMIC"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"
echo "OMP_PLACES=$OMP_PLACES"
echo

echo "=== Runtime size sweeps ==="
for workload in laplacian advection_diffusion; do
  python -m benchmarks.benchmark \
    --workload "$workload" \
    --size 127 \
    --size 255 \
    --size 511 \
    --size 1023 \
    --size 2047 \
    --size 4095 \
    --size 8191 \
    --include-openmp \
    --include-gt4py \
    --warmups 3 \
    --repeats 15 \
    --cxx g++ \
    --output-dir "$out"
done

echo
echo "=== Strong scaling ==="
for workload in laplacian advection_diffusion; do
  python -m benchmarks.benchmark \
    --workload "$workload" \
    --strong-scaling \
    --include-gt4py \
    --size 8191 \
    --thread-counts 1 2 4 8 16 32 64 \
    --warmups 3 \
    --repeats 15 \
    --cxx g++ \
    --output-dir "$out"
done

echo
echo "=== Weak scaling ==="
for workload in laplacian advection_diffusion; do
  python -m benchmarks.benchmark \
    --workload "$workload" \
    --weak-scaling \
    --include-gt4py \
    --size 1023 \
    --thread-counts 1 2 4 8 16 32 64 \
    --warmups 3 \
    --repeats 15 \
    --cxx g++ \
    --output-dir "$out"
done

echo
echo "=== Generate plots ==="

for workload in laplacian advection_diffusion; do
  python -m benchmarks.plot_runtime "$workload" \
    --data-dir "$out" \
    --output-dir "$out/plots"

  python -m benchmarks.plot_scaling "$workload" \
    --mode strong \
    --data-dir "$out" \
    --output-dir "$out/plots"

  python -m benchmarks.plot_scaling "$workload" \
    --mode weak \
    --data-dir "$out" \
    --output-dir "$out/plots"
done

echo
echo "Done."
echo "Results: $out"
echo "Plots:   $out/plots"
