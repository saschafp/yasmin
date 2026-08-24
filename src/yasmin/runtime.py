from collections.abc import Mapping

from yasmin.backends import get_backend
from yasmin.core import Field, Scalar
from yasmin.ir.stencil import Operator


def execute(
    operator: Operator,
    *,
    backend: str,
    fields: Mapping[Field, object],
    scalars: Mapping[Scalar, int | float] | None = None,
) -> None:
    get_backend(backend).execute(
        operator,
        fields=fields,
        scalars=scalars,
    )
