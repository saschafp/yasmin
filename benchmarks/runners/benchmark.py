import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS_DIR = ROOT / "kernels"


def list_kernels():
    return [p.stem for p in KERNELS_DIR.glob("*.py") if p.is_file() and p.stem != "__init__"]


def run_smoke(kernel_name: str):
    kernel_path = KERNELS_DIR / f"{kernel_name}.py"
    if not kernel_path.exists():
        print("Kernel not found:", kernel_name)
        return 2
    ns = runpy.run_path(str(kernel_path))
    if "run_smoke" in ns:
        ns["run_smoke"]()
    else:
        print("No smoke runner in kernel", kernel_name)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["list", "smoke", "time"], default="list")
    parser.add_argument("--kernel", default="laplacian")
    args = parser.parse_args()

    if args.action == "list":
        print("Available kernels:")
        for k in list_kernels():
            print(" -", k)
    elif args.action == "smoke":
        return run_smoke(args.kernel)
    elif args.action == "time":
        print("Timing not implemented yet")
        return 0

if __name__ == "__main__":
    sys.exit(main())
