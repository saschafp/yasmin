from dataclasses import dataclass

from yasmin.core import Dimension, DType
from yasmin.core import Field as CoreField
from yasmin.frontend.expr import Expr
from yasmin.ir import stencil as stencil_ir


@dataclass(frozen=True, slots=True)
class Field:
    core: CoreField

    def __init__(
        self,
        name: str,
        *,
        dims: tuple[Dimension, ...],
        dtype: DType,
    ) -> None:
        object.__setattr__(self, "core", CoreField(name=name, dims=dims, dtype=dtype))

    @property
    def name(self) -> str:
        return self.core.name

    @property
    def dims(self) -> tuple[Dimension, ...]:
        return self.core.dims

    @property
    def dtype(self) -> DType:
        return self.core.dtype

    def __getitem__(
        self,
        offsets: int | tuple[int, ...],
    ) -> Expr:
        normalized = (offsets,) if isinstance(offsets, int) else offsets

        if len(normalized) != len(self.core.dims):
            raise ValueError(
                f"Field {self.core.name!r} expects {len(self.core.dims)} "
                f"offsets, got {len(normalized)}"
            )

        return Expr(
            stencil_ir.FieldAccess(
                field=self.core,
                offsets=normalized,
            )
        )
