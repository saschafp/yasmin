from yasmin.core import Dimension, DType, Field, Scalar


def test_core_declarations() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field(
        name="u",
        dims=(x, y),
        dtype=DType.FLOAT64,
    )

    alpha = Scalar(
        name="alpha",
        dtype=DType.FLOAT64,
    )

    assert u.name == "u"
    assert u.dims == (x, y)
    assert u.dtype is DType.FLOAT64

    assert alpha.name == "alpha"
    assert alpha.dtype is DType.FLOAT64
