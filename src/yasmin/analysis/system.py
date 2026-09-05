import os


def available_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS / Windows
        return os.cpu_count() or 1
