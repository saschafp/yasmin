from collections.abc import Mapping
from typing import Any

import numpy.typing as npt

from yasmin.backends import CppBackend, NumPyBackend, OpenMPBackend
from yasmin.compiler.config import OpenMPOptions
from yasmin.compiler.openmp import resolve_openmp_config
from yasmin.core import Field as CoreField
from yasmin.frontend import Field, Operator, Scalar
from yasmin.ir import loop
from yasmin.lowering import lower
from yasmin.runtime.native import CompiledFunction

Array = npt.NDArray[Any]

ShapeKey = tuple[tuple[CoreField, tuple[int, ...]], ...]

_compilation_cache: dict[
    tuple[str, loop.Function, ShapeKey],
    CompiledFunction,
] = {}


def _shape_key(
    shapes: Mapping[CoreField, tuple[int, ...]],
) -> ShapeKey:
    return tuple((field, tuple(shape)) for field, shape in shapes.items())


def _get_compiled_function(
    backend: str,
    function: loop.Function,
    shapes: Mapping[CoreField, tuple[int, ...]],
) -> CompiledFunction:
    key = (backend, function, _shape_key(shapes))

    cached = _compilation_cache.get(key)
    if cached is not None:
        return cached

    if backend == "cpp":
        compiled_function = CppBackend().compile(function)

    elif backend == "openmp":
        options = OpenMPOptions()

        config = resolve_openmp_config(
            function,
            options=options,
            shapes=shapes,
        )

        compiled_function = OpenMPBackend(
            config=config,
        ).compile(function)

    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    _compilation_cache[key] = compiled_function
    return compiled_function


def _clear_compilation_cache() -> None:
    _compilation_cache.clear()


def execute(
    operator: Operator,
    *,
    backend: str,
    fields: Mapping[Field, Array],
    scalars: Mapping[Scalar, int | float] | None = None,
) -> None:
    field_bindings = {field._core: value for field, value in fields.items()}
    scalar_bindings = {scalar._core: value for scalar, value in (scalars or {}).items()}

    operator_ir = operator._as_ir()

    if backend == "numpy":
        NumPyBackend().execute(
            operator=operator_ir,
            fields=field_bindings,
            scalars=scalar_bindings,
        )
        return

    function = lower(
        operator=operator_ir,
        name="kernel",
    )

    shapes = {field: array.shape for field, array in field_bindings.items()}

    compiled_function = _get_compiled_function(
        backend,
        function,
        shapes,
    )

    compiled_function(
        fields=field_bindings,
        scalars=scalar_bindings,
    )
