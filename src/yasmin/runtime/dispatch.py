from collections.abc import Mapping
from typing import Any

import numpy.typing as npt

from yasmin.backends import CppBackend, NumPyBackend, OpenMPBackend
from yasmin.frontend import Field, Operator, Scalar
from yasmin.lowering import lower

Array = npt.NDArray[Any]


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

    function = lower(operator=operator_ir, name="kernel")
    shapes = {field: array.shape for field, array in field_bindings.items()}

    if backend == "cpp":
        print(CppBackend().source(function, shapes=shapes))  # TODO Saskia: Remove
        compiled_function = CppBackend().compile(function, shapes=shapes)
    elif backend == "openmp":
        print(OpenMPBackend().source(function, shapes=shapes))  # TODO Saskia: Remove
        compiled_function = OpenMPBackend().compile(function, shapes=shapes)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    compiled_function(
        fields=field_bindings,
        scalars=scalar_bindings,
    )
