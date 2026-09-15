from yasmin.analysis.dependencies import (
    Dependency,
    DependencyKind,
    collect_dependencies,
    dependencies_allow_fusion,
    statement_accesses,
)
from yasmin.core import Dimension, Field, float64
from yasmin.ir.stencil import Assign, BinaryExpr, BinaryOp, FieldAccess, Literal


def test_statement_accesses() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)

    left = FieldAccess(u, (-1,))
    right = FieldAccess(u, (1,))
    target = FieldAccess(out, (0,))

    statement = Assign(
        target=target,
        value=BinaryExpr(
            BinaryOp.ADD,
            left,
            right,
        ),
    )

    accesses = statement_accesses(statement)

    assert accesses.reads == frozenset({left, right})
    assert accesses.writes == frozenset({target})


def test_collect_raw_dependency_preserves_offsets() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    tmp = Field("tmp", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)

    tmp_write = FieldAccess(tmp, (0,))
    tmp_read = FieldAccess(tmp, (1,))

    source = Assign(
        target=tmp_write,
        value=FieldAccess(u, (0,)),
    )

    sink = Assign(
        target=FieldAccess(out, (0,)),
        value=tmp_read,
    )

    dependencies = collect_dependencies(source, sink)

    assert dependencies == frozenset(
        {
            Dependency(
                kind=DependencyKind.RAW,
                source=tmp_write,
                sink=tmp_read,
            )
        }
    )


def test_collect_war_dependency() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)

    u_read = FieldAccess(u, (0,))
    u_write = FieldAccess(u, (0,))

    source = Assign(
        target=FieldAccess(out, (0,)),
        value=u_read,
    )

    sink = Assign(
        target=u_write,
        value=Literal(1.0),
    )

    dependencies = collect_dependencies(source, sink)

    assert dependencies == frozenset(
        {
            Dependency(
                kind=DependencyKind.WAR,
                source=u_read,
                sink=u_write,
            )
        }
    )


def test_collect_waw_dependency() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)

    first_write = FieldAccess(u, (0,))
    second_write = FieldAccess(u, (0,))

    source = Assign(
        target=first_write,
        value=Literal(1.0),
    )

    sink = Assign(
        target=second_write,
        value=Literal(2.0),
    )

    dependencies = collect_dependencies(source, sink)

    assert dependencies == frozenset(
        {
            Dependency(
                kind=DependencyKind.WAW,
                source=first_write,
                sink=second_write,
            )
        }
    )


def test_collect_dependencies_ignores_unrelated_fields() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    v = Field("v", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)

    source = Assign(
        target=FieldAccess(out, (0,)),
        value=FieldAccess(u, (0,)),
    )

    sink = Assign(
        target=FieldAccess(v, (0,)),
        value=Literal(1.0),
    )

    assert collect_dependencies(source, sink) == frozenset()


def test_pointwise_dependency_has_zero_distance() -> None:
    x = Dimension("x")

    tmp = Field("tmp", dims=(x,), dtype=float64)

    dependency = Dependency(
        kind=DependencyKind.RAW,
        source=FieldAccess(tmp, (0,)),
        sink=FieldAccess(tmp, (0,)),
    )

    assert dependency.distance == (0,)
    assert not dependency.is_loop_carried


def test_forward_loop_carried_dependency() -> None:
    x = Dimension("x")

    tmp = Field("tmp", dims=(x,), dtype=float64)

    dependency = Dependency(
        kind=DependencyKind.RAW,
        source=FieldAccess(tmp, (0,)),
        sink=FieldAccess(tmp, (-1,)),
    )

    assert dependency.distance == (1,)
    assert dependency.is_loop_carried


def test_backward_loop_carried_dependency() -> None:
    x = Dimension("x")

    tmp = Field("tmp", dims=(x,), dtype=float64)

    dependency = Dependency(
        kind=DependencyKind.RAW,
        source=FieldAccess(tmp, (0,)),
        sink=FieldAccess(tmp, (1,)),
    )

    assert dependency.distance == (-1,)
    assert dependency.is_loop_carried


def test_multidimensional_dependency_distance() -> None:
    x = Dimension("x")
    y = Dimension("y")

    tmp = Field("tmp", dims=(x, y), dtype=float64)

    dependency = Dependency(
        kind=DependencyKind.RAW,
        source=FieldAccess(tmp, (0, 0)),
        sink=FieldAccess(tmp, (-1, 1)),
    )

    assert dependency.distance == (1, -1)
    assert dependency.is_loop_carried


def test_independent_statements_allow_fusion() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    v = Field("v", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)
    tmp = Field("tmp", dims=(x,), dtype=float64)

    source = Assign(
        target=FieldAccess(tmp, (0,)),
        value=FieldAccess(u, (0,)),
    )

    sink = Assign(
        target=FieldAccess(out, (0,)),
        value=FieldAccess(v, (0,)),
    )

    assert dependencies_allow_fusion(source, sink)


def test_pointwise_raw_dependency_allows_fusion() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    tmp = Field("tmp", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)

    source = Assign(
        target=FieldAccess(tmp, (0,)),
        value=FieldAccess(u, (0,)),
    )

    sink = Assign(
        target=FieldAccess(out, (0,)),
        value=FieldAccess(tmp, (0,)),
    )

    assert dependencies_allow_fusion(source, sink)


def test_loop_carried_raw_dependency_prevents_fusion() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    tmp = Field("tmp", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)

    source = Assign(
        target=FieldAccess(tmp, (0,)),
        value=FieldAccess(u, (0,)),
    )

    sink = Assign(
        target=FieldAccess(out, (0,)),
        value=FieldAccess(tmp, (1,)),
    )

    assert not dependencies_allow_fusion(source, sink)


def test_loop_carried_war_dependency_prevents_fusion() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)

    source = Assign(
        target=FieldAccess(out, (0,)),
        value=FieldAccess(u, (1,)),
    )

    sink = Assign(
        target=FieldAccess(u, (0,)),
        value=Literal(1.0),
    )

    assert not dependencies_allow_fusion(source, sink)
