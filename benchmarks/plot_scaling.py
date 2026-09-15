from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal

from benchmarks.color_list import COLOR_BY_IMPLEMENTATION, MARKER_BY_IMPLEMENTATION
from benchmarks.plot_runtime import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, _load_pyplot


def load_scaling_data(
    data_dir: Path,
    problem: str,
    mode: Literal["strong", "weak"] = "strong",
) -> tuple[int, int, dict[str, list[tuple[int, float]]]]:
    data: dict[str, list[tuple[int, float]]] = {}
    sizes: dict[int, tuple[int, int]] = {}
    seen: set[tuple[str, int]] = set()
    path = data_dir / f"{problem}_{mode}_scaling.csv"

    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            implementation = row["implementation"].strip()
            nx, ny, threads = int(row["nx"]), int(row["ny"]), int(row["threads"])
            runtime = float(row["runtime_ms"])

            if implementation not in ("yasmin_openmp", "cpp_openmp", "gt4py_cpu"):
                raise ValueError(f"Unexpected scaling implementation: {implementation}")

            if row["correct"] != "true":
                raise ValueError(
                    f"Incorrect result for {implementation}, {threads} threads"
                )

            if nx < 2 or ny < 2 or threads < 1 or runtime <= 0:
                raise ValueError(
                    "Scaling requires positive sizes, thread counts and runtimes"
                )

            if (implementation, threads) in seen:
                raise ValueError(
                    f"Duplicate result for {implementation}, {threads} threads"
                )

            seen.add((implementation, threads))

            size = (nx, ny)
            if threads in sizes and sizes[threads] != size:
                raise ValueError(
                    "Implementations must use the same size at each thread count"
                )

            sizes[threads] = size
            data.setdefault(implementation, []).append((threads, runtime))

    if 1 not in sizes:
        raise ValueError("Missing one-thread grid size")

    base_nx, base_ny = sizes[1]

    for threads, size in sizes.items():
        expected = (
            (base_nx, base_ny * threads) if mode == "weak" else (base_nx, base_ny)
        )

        if size != expected:
            raise ValueError(
                f"Unexpected grid size for {mode} scaling at {threads} threads"
            )

    for implementation, points in data.items():
        points.sort()
        if points[0][0] != 1:
            raise ValueError(f"Missing one-thread baseline for {implementation}")

    return base_nx, base_ny, data


def plot_scaling(
    *,
    problem: str,
    data_dir: Path,
    output_dir: Path,
    mode: Literal["strong", "weak"] = "strong",
) -> Path:
    _, _, data = load_scaling_data(data_dir, problem, mode)
    plt = _load_pyplot()
    fig, axis = plt.subplots(figsize=(6, 5))

    all_threads = sorted({threads for points in data.values() for threads, _ in points})

    for implementation, points in data.items():
        threads = [count for count, _ in points]
        runtime = [value for _, value in points]
        scaling = [
            runtime[0] / value * (100 if mode == "weak" else 1) for value in runtime
        ]

        axis.plot(
            threads,
            scaling,
            marker=MARKER_BY_IMPLEMENTATION.get(implementation, "o"),
            color=COLOR_BY_IMPLEMENTATION.get(implementation),
            linewidth=2,
            label=implementation,
        )

    ideal = [100] * len(all_threads) if mode == "weak" else all_threads
    axis.plot(all_threads, ideal, "--", color="gray", label="Ideal")

    axis.set_ylabel(
        "Efficiency (%)" if mode == "weak" else "Speedup vs. own one-thread runtime"
    )
    axis.set_xlabel("Threads")
    axis.set_title("Weak Scaling" if mode == "weak" else "Strong Scaling")

    axis.set_xscale("log", base=2)
    axis.set_xticks(all_threads)
    axis.set_xticklabels([str(thread) for thread in all_threads])
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{problem}_{mode}_scaling.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CPU scaling.")
    parser.add_argument("--mode", choices=("strong", "weak"), default="strong")
    parser.add_argument("problem", nargs="?", default="laplacian")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()

    output = plot_scaling(
        problem=args.problem,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mode=args.mode,
    )

    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    main()
