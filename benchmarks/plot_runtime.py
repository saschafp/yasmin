#!/usr/bin/env python3
"""Plot benchmark runtime over grid size for a given problem.

Usage:
    python3 plot_runtime.py --problem laplacian
    python3 plot_runtime.py advection_diffusion
"""

from __future__ import annotations
from color_list import color_list
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_problem_csvs(data_dir: Path, problem: str):
    csv_files = []
    for path in sorted(data_dir.glob(f"{problem}_*.csv")):
        name = path.name
        if "scaling" in name:
            continue
        stem = path.stem
        size_part = stem[len(problem) + 1 :]
        if size_part.isdigit():
            csv_files.append(path)
    return csv_files


def load_runtime_data(data_dir: Path, problem: str):
    runtime_by_impl: dict[str, list[tuple[int, float]]] = {}

    for csv_path in find_problem_csvs(data_dir, problem):
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                implementation = row["implementation"].strip()
                nx = int(row["nx"])
                runtime_ms = float(row["runtime_ms"])
                runtime_by_impl.setdefault(implementation, []).append((nx, runtime_ms))

    for values in runtime_by_impl.values():
        values.sort(key=lambda item: item[0])

    return runtime_by_impl


def plot_runtime(problem: str):
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "outputs" / "raw" / "csv"
    out_dir = repo_root / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime_by_impl = load_runtime_data(data_dir, problem)
    if not runtime_by_impl:
        raise FileNotFoundError(
            f"No runtime CSV data found for problem '{problem}' in {data_dir}"
        )

    plt.figure(figsize=(10, 6))
    for implementation, points in runtime_by_impl.items():
        nx_values = [nx for nx, _ in points]
        runtime_values = [runtime for _, runtime in points]
        c = color_list[implementation]
        m = "o" if "yasmin" in implementation else "x"
        plt.plot(nx_values, runtime_values, marker=m, linewidth=2, label=implementation, color=c)

    plt.title(f"{problem.replace('_', ' ').title()} runtime vs. grid size")
    plt.xlabel("Grid size (NxN)")
    plt.ylabel("Runtime (ms)")
    plt.xscale("log", base=2)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(title="Implementation")
    plt.tight_layout()

    output_path = out_dir / f"{problem}_runtime.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark runtime over grid sizes.")
    parser.add_argument("problem", nargs="?", help="Problem name, e.g. laplacian or advection_diffusion")
    parser.add_argument("--problem", dest="problem_flag", help="Problem name, e.g. laplacian or advection_diffusion")
    args = parser.parse_args()

    problem = args.problem_flag or args.problem
    if not problem:
        parser.error("A problem name is required: e.g. laplacian or advection_diffusion")

    plot_runtime(problem)
