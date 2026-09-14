#!/usr/bin/env python3
"""Plot strong and weak scaling benchmark data for a given problem.

Usage:
    python3 plot_scaling.py laplacian
    python3 plot_scaling.py --problem advection_diffusion
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from color_list import color_list


def load_scaling_data(data_dir: Path, problem: str):
    strong_data: dict[str, list[tuple[int, float]]] = {}
    weak_data: dict[str, list[tuple[int, float]]] = {}

    for mode, data in (("strong", strong_data), ("weak", weak_data)):
        csv_path = data_dir / f"{problem}_{mode}_scaling.csv"
        if not csv_path.exists():
            continue

        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                implementation = row["implementation"].strip()
                threads = int(row["threads"])
                runtime_ms = float(row["runtime_ms"])
                data.setdefault(implementation, []).append((threads, runtime_ms))

        for values in data.values():
            values.sort(key=lambda item: item[0])

    return strong_data, weak_data


def plot_scaling(problem: str):
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "outputs" / "raw" / "csv"
    out_dir = repo_root / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    strong_data, weak_data = load_scaling_data(data_dir, problem)
    if not strong_data and not weak_data:
        raise FileNotFoundError(
            f"No scaling CSV files found for problem '{problem}' in {data_dir}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    if strong_data:
        for implementation, points in strong_data.items():
            threads = [threads for threads, _ in points]
            runtime = [runtime for _, runtime in points]
            c = color_list[implementation]
            m = "o" if "yasmin" in implementation else "x"
            axes[0].plot(
                threads,
                runtime,
                marker=m,
                linewidth=2,
                label=implementation,
                color=c,
            )
        axes[0].set_title(f"{problem.replace('_', ' ').title()} strong scaling")
        axes[0].set_xlabel("Threads")
        axes[0].set_ylabel("Runtime (ms)")
        axes[0].grid(True, which="both", ls="--", alpha=0.4)
        axes[0].legend(title="Implementation")

    if weak_data:
        for implementation, points in weak_data.items():
            threads = [threads for threads, _ in points]
            runtime = [runtime for _, runtime in points]
            c = color_list[implementation]
            m = "o" if "yasmin" in implementation else "x"
            axes[1].plot(
                threads,
                runtime,
                marker=m,
                linewidth=2,
                label=implementation,
                color=c,
            )
        axes[1].set_title(f"{problem.replace('_', ' ').title()} weak scaling")
        axes[1].set_xlabel("Threads")
        axes[1].set_ylabel("Runtime (ms)")
        axes[1].grid(True, which="both", ls="--", alpha=0.4)
        axes[1].legend(title="Implementation")

    fig.tight_layout()
    output_path = out_dir / f"{problem}_scaling.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot strong and weak scaling benchmark results.")
    parser.add_argument("problem", nargs="?", help="Problem name, e.g. laplacian or advection_diffusion")
    parser.add_argument("--problem", dest="problem_flag", help="Problem name, e.g. laplacian or advection_diffusion")
    args = parser.parse_args()

    problem = args.problem_flag or args.problem
    if not problem:
        parser.error("A problem name is required: e.g. laplacian or advection_diffusion")

    plot_scaling(problem)
