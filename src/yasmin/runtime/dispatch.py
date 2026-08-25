from collections.abc import Mapping

from yasmin.backends import get_backend
from yasmin.core import Scalar
from yasmin.frontend import Field, Operator


def execute(
    operator: Operator,
    *,
    backend: str,
    fields: Mapping[Field, object],
    scalars: Mapping[Scalar, int | float] | None = None,
) -> None:
    field_bindings = {field.core: value for field, value in fields.items()}

    get_backend(backend).execute(
        operator.ir,
        fields=field_bindings,
        scalars=scalars,
    )
