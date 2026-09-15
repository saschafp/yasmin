from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

from benchmarks.common import (
    BenchmarkResult,
    WorkloadResult,
    arrays_close,
    compile_cpp_executable,
    load_workload,
    numpy_reference,
    print_csv,
    run_key_value_executable,
    validate_arguments,
    workload_directory,
    write_csv,
)

Implementation = Literal[
    "yasmin_numpy",
    "yasmin_cpp",
    "yasmin_openmp",
    "numpy",
    "cpp",
    "cpp_openmp",
]

DEFAULT_SIZES = [128, 256, 512, 1024, 2048]
DEFAULT_IMPLEMENTATIONS: list[Implementation] = [
    "yasmin_numpy",
    "numpy",
    "yasmin_cpp",
    "cpp",
]
OPENMP_IMPLEMENTATIONS: tuple[Implementation, Implementation] = (
    "yasmin_openmp",
    "cpp_openmp",
)
DEFAULT_OUTPUT_DIR = Path("benchmarks/outputs/raw/csv")


def run(
    *,
    workload: str,
    sizes: list[int],
    implementations: list[Implementation],
    warmups: int,
    repeats: int,
    cxx: str,
    threads: int | None,
) -> dict[int, list[BenchmarkResult]]:
    results_by_size: dict[int, list[BenchmarkResult]] = {}

    for nx in sizes:
        results_by_size[nx] = [
            _run_implementation(
                workload=workload,
                implementation=implementation,
                nx=nx,
                warmups=warmups,
                repeats=repeats,
                cxx=cxx,
                threads=threads,
            )
            for implementation in implementations
        ]

    return results_by_size


def _run_implementation(
    *,
    workload: str,
    implementation: Implementation,
    nx: int,
    warmups: int,
    repeats: int,
    cxx: str,
    threads: int | None,
) -> BenchmarkResult:
    validate_arguments(nx, warmups, repeats)
    if threads is not None and threads < 1:
        raise ValueError("Expected threads >= 1")
    if implementation in ("cpp", "cpp_openmp"):
        result = run_cpp(
            workload=workload,
            backend="openmp" if implementation == "cpp_openmp" else "cpp",
            nx=nx,
            warmups=warmups,
            repeats=repeats,
            cxx=cxx,
            threads=threads,
        )
    else:
        result = run_python(
            workload=workload,
            implementation=implementation,
            nx=nx,
            warmups=warmups,
            repeats=repeats,
            cxx=cxx,
            threads=threads,
        )

    expected = numpy_reference(workload, nx)
    correct = result.output.size == expected.size and arrays_close(
        result.output.reshape(expected.shape),
        expected,
    )
    return BenchmarkResult(
        implementation=implementation,
        nx=nx,
        threads=threads if implementation in OPENMP_IMPLEMENTATIONS else None,
        runtime_ms=result.runtime_ms,
        correct=correct,
    )


def run_cpp(
    *,
    workload: str,
    backend: Literal["cpp", "openmp"],
    nx: int,
    warmups: int,
    repeats: int,
    cxx: str,
    threads: int | None,
) -> WorkloadResult:
    source = workload_directory(workload) / f"{workload}_{backend}.cpp"
    executable = compile_cpp_executable(source, cxx=cxx, openmp=backend == "openmp")
    env = os.environ.copy()
    if backend == "openmp":
        env["OMP_DYNAMIC"] = "FALSE"
        if threads is not None:
            env["OMP_NUM_THREADS"] = str(threads)

    try:
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / f"{workload}.bin"
            output = run_key_value_executable(
                executable.path,
                [str(nx), str(warmups), str(repeats), str(output_path)],
                env=env,
            )
            actual = np.fromfile(output_path, dtype=np.float64)
    finally:
        executable.directory.cleanup()
    return WorkloadResult(
        output=actual,
        runtime_ms=float(output["RUNTIME_MS"]),
    )


def run_python(
    *,
    workload: str,
    implementation: Implementation,
    nx: int,
    warmups: int,
    repeats: int,
    cxx: str,
    threads: int | None,
) -> WorkloadResult:
    result: WorkloadResult
    if implementation == "numpy":
        module = load_workload(workload, "numpy")
        result = module.main(nx=nx, warmups=warmups, repeats=repeats)
    else:
        module = load_workload(workload, "yasmin")
        result = module.main(
            backend=implementation.removeprefix("yasmin_"),
            nx=nx,
            warmups=warmups,
            repeats=repeats,
            cxx=cxx,
            threads=threads,
        )
    return result


def _selected_implementations(
    implementations: list[Implementation] | None,
    *,
    include_openmp: bool,
) -> list[Implementation]:
    selected = list(implementations or DEFAULT_IMPLEMENTATIONS)

    if include_openmp:
        for implementation in OPENMP_IMPLEMENTATIONS:
            if implementation not in selected:
                selected.append(implementation)

    return selected


def _write_outputs(
    *,
    workload: str,
    output_dir: Path,
    results_by_size: dict[int, list[BenchmarkResult]],
) -> None:
    for nx, results in results_by_size.items():
        write_csv(output_dir / f"{workload}_{nx}.csv", results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Yasmin benchmarks.")
    parser.add_argument("--workload", default="laplacian")
    parser.add_argument(
        "--size",
        type=int,
        action="append",
        dest="sizes",
        help="Grid size to benchmark. Can be supplied multiple times.",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--implementation",
        choices=(
            "yasmin_numpy",
            "yasmin_cpp",
            "yasmin_openmp",
            "numpy",
            "cpp",
            "cpp_openmp",
        ),
        action="append",
        dest="implementations",
        help="Implementation to benchmark. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--include-openmp",
        action="store_true",
        help="Include Yasmin OpenMP and standalone OpenMP implementations.",
    )
    parser.add_argument(
        "--cxx",
        default=os.environ.get("CXX") or "c++",
        help="C++ compiler used for native and standalone C++ benchmarks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-size CSV files are written.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Print CSV rows without writing benchmark output files.",
    )

    args = parser.parse_args()
    implementations = _selected_implementations(
        args.implementations,
        include_openmp=args.include_openmp,
    )

    results_by_size = run(
        workload=args.workload,
        sizes=args.sizes or DEFAULT_SIZES,
        implementations=implementations,
        warmups=args.warmups,
        repeats=args.repeats,
        cxx=args.cxx,
        threads=args.threads,
    )

    all_results = [result for results in results_by_size.values() for result in results]
    print_csv(all_results)

    if not args.no_output:
        _write_outputs(
            workload=args.workload,
            output_dir=args.output_dir,
            results_by_size=results_by_size,
        )


if __name__ == "__main__":
    main()
