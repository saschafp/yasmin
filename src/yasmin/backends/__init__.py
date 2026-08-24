from typing import Any

from yasmin.backends.base import ExecutionBackend
from yasmin.backends.cpp import CppBackend
from yasmin.backends.numpy import NumPyBackend
from yasmin.backends.openmp import OpenMPBackend

__all__ = [
    "CppBackend",
    "ExecutionBackend",
    "NumPyBackend",
    "OpenMPBackend",
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
