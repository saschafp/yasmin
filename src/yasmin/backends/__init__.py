from typing import Any

from yasmin.backends.base import ExecutionBackend
from yasmin.backends.numpy import NumPyBackend

__all__ = [
    "ExecutionBackend",
    "NumPyBackend",
    "get_backend",
]


_BACKENDS: dict[str, ExecutionBackend[Any]] = {
    "numpy": NumPyBackend(),
}


def get_backend(name: str) -> ExecutionBackend[Any]:
    try:
        return _BACKENDS[name]
    except KeyError as e:
        raise ValueError(f"Unknown backend: {name!r}") from e
