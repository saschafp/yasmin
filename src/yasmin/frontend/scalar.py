from dataclasses import dataclass

from yasmin.core import DType
from yasmin.core import Scalar as CoreScalar
from yasmin.frontend.expr import Expr
from yasmin.ir import stencil as stencil_ir


@dataclass(frozen=True, slots=True)
class Scalar:
    core: CoreScalar

    def __init__(
        self,
        name: str,
        *,
        dtype: DType,
    ) -> None:
        object.__setattr__(self, "core", CoreScalar(name=name, dtype=dtype))

    @property
    def name(self) -> str:
        return self.core.name

    @property
    def dtype(self) -> DType:
        return self.core.dtype

    def _expr(self) -> Expr:
        return Expr(stencil_ir.ScalarRef(scalar=self.core))

    def __add__(self, other: Expr | int | float) -> Expr:
        return self._expr() + other

    def __radd__(self, other: Expr | int | float) -> Expr:
        return other + self._expr()

    def __sub__(self, other: Expr | int | float) -> Expr:
        return self._expr() - other

    def __rsub__(self, other: Expr | int | float) -> Expr:
        return other - self._expr()

    def __mul__(self, other: Expr | int | float) -> Expr:
        return self._expr() * other

    def __rmul__(self, other: Expr | int | float) -> Expr:
        return other * self._expr()

    def __truediv__(self, other: Expr | int | float) -> Expr:
        return self._expr() / other

    def __rtruediv__(self, other: Expr | int | float) -> Expr:
        return other / self._expr()
