from collections.abc import Mapping

from yasmin.backends import get_backend
from yasmin.frontend import Field, Operator, Scalar


def execute(
    operator: Operator,
    *,
    backend: str,
    fields: Mapping[Field, object],
    scalars: Mapping[Scalar, int | float] | None = None,
) -> None:
    field_bindings = {field.core: value for field, value in fields.items()}
    scalar_bindings = {scalar.core: value for scalar, value in (scalars or {}).items()}

    get_backend(backend).execute(
        operator.ir,
        fields=field_bindings,
        scalars=scalar_bindings,
    )
