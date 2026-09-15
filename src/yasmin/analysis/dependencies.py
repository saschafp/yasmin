from dataclasses import dataclass
from enum import Enum

from yasmin.analysis.accesses import collect_accesses
from yasmin.ir.stencil import Assign, FieldAccess


@dataclass(frozen=True, slots=True)
class StatementAccesses:
    reads: frozenset[FieldAccess]
    writes: frozenset[FieldAccess]


class DependencyKind(Enum):
    RAW = "raw"
    WAR = "war"
    WAW = "waw"


@dataclass(frozen=True, slots=True)
class Dependency:
    kind: DependencyKind
    source: FieldAccess
    sink: FieldAccess

    @property
    def distance(self) -> tuple[int, ...]:
        return tuple(
            source_offset - sink_offset
            for source_offset, sink_offset in zip(
                self.source.offsets,
                self.sink.offsets,
                strict=True,
            )
        )

    @property
    def is_loop_carried(self) -> bool:
        return any(distance != 0 for distance in self.distance)


def statement_accesses(statement: Assign) -> StatementAccesses:
    return StatementAccesses(
        reads=frozenset(collect_accesses(statement.value)),
        writes=frozenset((statement.target,)),
    )


def collect_dependencies(
    source: Assign,
    sink: Assign,
) -> frozenset[Dependency]:
    source_accesses = statement_accesses(source)
    sink_accesses = statement_accesses(sink)

    dependencies: set[Dependency] = set()

    for source_write in source_accesses.writes:
        for sink_read in sink_accesses.reads:
            if source_write.field == sink_read.field:
                dependencies.add(
                    Dependency(
                        kind=DependencyKind.RAW,
                        source=source_write,
                        sink=sink_read,
                    )
                )

    for source_read in source_accesses.reads:
        for sink_write in sink_accesses.writes:
            if source_read.field == sink_write.field:
                dependencies.add(
                    Dependency(
                        kind=DependencyKind.WAR,
                        source=source_read,
                        sink=sink_write,
                    )
                )

    for source_write in source_accesses.writes:
        for sink_write in sink_accesses.writes:
            if source_write.field == sink_write.field:
                dependencies.add(
                    Dependency(
                        kind=DependencyKind.WAW,
                        source=source_write,
                        sink=sink_write,
                    )
                )

    return frozenset(dependencies)


def dependencies_allow_fusion(
    source: Assign,
    sink: Assign,
) -> bool:
    return all(
        not dependency.is_loop_carried
        for dependency in collect_dependencies(source, sink)
    )
