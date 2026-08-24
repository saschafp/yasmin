import ctypes
import platform
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from yasmin.core import Field, Scalar
from yasmin.ir import loop

Array = npt.NDArray[Any]
ScalarValue = int | float


@dataclass(frozen=True, slots=True)
class SharedLibrary:
    library: ctypes.CDLL
    directory: tempfile.TemporaryDirectory[str]


def compile_cpp(
    source: str,
    *,
    compiler: str = "c++",
    extra_flags: tuple[str, ...] = (),
) -> SharedLibrary:
    directory = tempfile.TemporaryDirectory()
    root = Path(directory.name)

    source_path = root / "kernel.cpp"

    system = platform.system()

    shared_flags: tuple[str, ...]

    if system == "Darwin":
        library_path = root / "libkernel.dylib"
        shared_flags = ("-dynamiclib",)
    elif system == "Linux":
        library_path = root / "libkernel.so"
        shared_flags = ("-shared", "-fPIC")
    else:
        directory.cleanup()
        raise RuntimeError(f"Unsupported platform: {system}")

    source_path.write_text(source)

    command = [
        compiler,
        "-O3",
        "-std=c++17",
        *shared_flags,
        *extra_flags,
        str(source_path),
        "-o",
        str(library_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        directory.cleanup()
        raise RuntimeError(f"Compilation failed:\n{e.stderr}") from e
    except OSError as e:
        directory.cleanup()
        raise RuntimeError(f"Failed to run compiler: {e!r}") from e
    return SharedLibrary(
        library=ctypes.CDLL(str(library_path)),
        directory=directory,
    )


class CompiledFunction:
    def __init__(self, function: loop.Function, shared_library: SharedLibrary) -> None:
        self.function = function
        self.shared_library = shared_library

        native_function = getattr(shared_library.library, function.name)

        native_function.restype = None

        argtypes: list[Any] = []

        for _field in function.fields:
            argtypes.append(ctypes.POINTER(ctypes.c_double))

        for field in function.fields:
            for _dim in field.dims:
                argtypes.append(ctypes.c_int)

        for _scalar in function.scalars:
            argtypes.append(ctypes.c_double)

        native_function.argtypes = argtypes

        self._function = native_function

    def __call__(
        self,
        *,
        fields: Mapping[Field, Array],
        scalars: Mapping[Scalar, ScalarValue] | None = None,
    ) -> None:
        scalar_bindings = scalars or {}

        args: list[Any] = []

        for field in self.function.fields:
            array = fields[field]

            if array.dtype != np.float64:
                raise TypeError(f"Field {field.name} must be of type float64")

            if not array.flags.c_contiguous:
                raise ValueError(f"Field {field.name} must be C-contiguous")

            if array.ndim != len(field.dims):
                raise ValueError(
                    f"Field {field.name!r} expects {len(field.dims)} dimensions, "
                    f"got {array.ndim}"
                )

            args.append(array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        for field in self.function.fields:
            array = fields[field]

            for extent in array.shape:
                args.append(int(extent))

        for scalar in self.function.scalars:
            args.append(float(scalar_bindings[scalar]))

        self._function(*args)
