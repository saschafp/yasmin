from yasmin.core import Dimension, Field, float64
from yasmin.ir import loop, stencil
from yasmin.lowering import lower


def test_lower_1d_stencil() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)

    expression = stencil.BinaryExpr(
        stencil.BinaryOp.ADD,
        stencil.FieldAccess(u, (-1,)),
        stencil.FieldAccess(u, (1,)),
    )

    operator = stencil.Operator(
        statements=(
            stencil.Assign(
                target=stencil.FieldAccess(out, (0,)),
                value=expression,
            ),
        )
    )

    function = lower(operator, name="stencil_1d")

    assert function.name == "stencil_1d"
    assert function.fields == (out, u)
    assert function.scalars == ()
    assert len(function.body) == 1

    outer = function.body[0]

    assert isinstance(outer, loop.For)
    assert outer.index == loop.Index("x")
    assert outer.lower == loop.Literal(1)
    assert outer.upper == loop.BinaryExpr(
        loop.BinaryOp.SUB,
        loop.Extent(out, 0),
        loop.Literal(1),
    )
    assert len(outer.body) == 1

    store = outer.body[0]

    assert isinstance(store, loop.Store)
    assert store.field == out
    assert store.indices == (loop.Index("x"),)
    assert store.value == loop.BinaryExpr(
        loop.BinaryOp.ADD,
        loop.Load(
            field=u,
            indices=(
                loop.BinaryExpr(
                    loop.BinaryOp.SUB,
                    loop.Index("x"),
                    loop.Literal(1),
                ),
            ),
        ),
        loop.Load(
            field=u,
            indices=(
                loop.BinaryExpr(
                    loop.BinaryOp.ADD,
                    loop.Index("x"),
                    loop.Literal(1),
                ),
            ),
        ),
    )
