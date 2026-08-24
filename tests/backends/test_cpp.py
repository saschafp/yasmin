from yasmin.backends.cpp import CppBackend
from yasmin.core import Dimension, Field, float64
from yasmin.ir import loop


def test_emit_one_dimensional_stencil() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)

    i = loop.Index("x")

    function = loop.Function(
        name="stencil_1d",
        fields=(out, u),
        scalars=(),
        body=(
            loop.For(
                index=i,
                lower=loop.Literal(1),
                upper=loop.BinaryExpr(
                    loop.BinaryOp.SUB,
                    loop.Extent(out, 0),
                    loop.Literal(1),
                ),
                body=(
                    loop.Store(
                        field=out,
                        indices=(i,),
                        value=loop.BinaryExpr(
                            loop.BinaryOp.ADD,
                            loop.Load(
                                u,
                                (
                                    loop.BinaryExpr(
                                        loop.BinaryOp.SUB,
                                        i,
                                        loop.Literal(1),
                                    ),
                                ),
                            ),
                            loop.Load(
                                u,
                                (
                                    loop.BinaryExpr(
                                        loop.BinaryOp.ADD,
                                        i,
                                        loop.Literal(1),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    source = CppBackend().source(function)

    assert "void stencil_1d" in source
    assert "for (int x = 1;" in source
    assert "out[x]" in source
    assert "u[(x - 1)]" in source
    assert "u[(x + 1)]" in source
