import os


def available_cores() -> int:
    try:
        affinity_cores = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS / Windows
        affinity_cores = os.cpu_count() or 1

    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_num_threads is not None:
        try:
            omp_value = int(omp_num_threads)
        except ValueError:
            return affinity_cores
        if omp_value > 0:
            return min(omp_value, affinity_cores)

    return affinity_cores
