from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
from typing import Any

from benchmarks.color_list import COLOR_BY_IMPLEMENTATION, MARKER_BY_IMPLEMENTATION

DEFAULT_DATA_DIR = Path("benchmarks/outputs/raw/csv")
DEFAULT_OUTPUT_DIR = Path("benchmarks/outputs/plots")


def find_problem_csvs(data_dir: Path, problem: str) -> list[Path]:
    csv_files: list[Path] = []

    for path in sorted(data_dir.glob(f"{problem}_*.csv")):
        if "scaling" in path.name:
            continue

        size_part = path.stem[len(problem) + 1 :]
        if size_part.isdigit():
            csv_files.append(path)

    return csv_files


def load_runtime_data(
    *,
    data_dir: Path,
    problem: str,
) -> dict[str, list[tuple[int, float]]]:
    runtime_by_implementation: dict[str, list[tuple[int, float]]] = {}

    for csv_path in find_problem_csvs(data_dir, problem):
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["correct"] != "true":
                    continue

                implementation = row["implementation"].strip()
                nx = int(row["nx"])
                runtime_ms = float(row["runtime_ms"])
                runtime_by_implementation.setdefault(implementation, []).append(
                    (nx, runtime_ms)
                )

    for values in runtime_by_implementation.values():
        values.sort(key=lambda item: item[0])

    return runtime_by_implementation


def plot_runtime(
    *,
    problem: str,
    data_dir: Path,
    output_dir: Path,
) -> Path:
    runtime_by_implementation = load_runtime_data(
        data_dir=data_dir,
        problem=problem,
    )

    if not runtime_by_implementation:
        raise FileNotFoundError(
            f"No correct runtime CSV data found for {problem!r} in {data_dir}"
        )

    plt = _load_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, axis = plt.subplots(figsize=(10, 6))

    for implementation, points in runtime_by_implementation.items():
        nx_values = [nx for nx, _runtime_ms in points]
        runtime_values = [runtime_ms for _nx, runtime_ms in points]

        axis.plot(
            nx_values,
            runtime_values,
            marker=MARKER_BY_IMPLEMENTATION.get(implementation, "o"),
            linewidth=2,
            label=implementation,
            color=COLOR_BY_IMPLEMENTATION.get(implementation),
        )

    axis.set_title(f"{problem.replace('_', ' ').title()} runtime vs. grid size")
    axis.set_xlabel("Grid size (NxN)")
    axis.set_ylabel("Runtime (ms)")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.grid(True, which="both", linestyle="--", alpha=0.4)
    axis.legend(title="Implementation")

    output_path = output_dir / f"{problem}_runtime.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def _load_pyplot() -> Any:
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        return importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Plotting requires matplotlib. Install the benchmark extras first."
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark runtime results.")
    parser.add_argument("problem", nargs="?", default="laplacian")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()
    output_path = plot_runtime(
        problem=args.problem,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
