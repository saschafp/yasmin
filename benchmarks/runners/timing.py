import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end = time.perf_counter()

    @property
    def elapsed(self):
        if self.start is None:
            return None
        return (self.end or time.perf_counter()) - self.start


def time_function(func, *args, **kwargs):
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    t1 = time.perf_counter()
    return (t1 - t0), res
