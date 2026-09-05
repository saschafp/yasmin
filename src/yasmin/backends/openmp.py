from dataclasses import dataclass

from yasmin.analysis.omp_heuristic import parallel_config
from yasmin.analysis.system import available_cores
from yasmin.backends.cpp import CppBackend, Shapes
from yasmin.core import Field
from yasmin.ir import loop
from yasmin.runtime.native import CompiledFunction, compile_cpp


@dataclass(frozen=True)
class OpenMPOptions:
    use_restrict: bool = True
    use_collapse: bool = True
    schedule: str | None = "static"  # None | "static" | "dynamic" | "guided"
    schedule_chunk: int | None = (
        None  # e.g for schedule(static, 2), the schedule_chunk is 2.
    )
    num_threads: int | None = None  # None = auto-detect
    adaptive: bool = (
        False  # use shape based heuristic to deduce schedule/collapse settings
    )
    min_iters_per_thread: int = 1_000  # TODO: extract dynamically based on hardware
    extra_compile_flags: tuple[str, ...] = ("-march=native", "-fno-math-errno")

    @classmethod
    def baseline(cls) -> "OpenMPOptions":
        """OpenMP without any optimizations"""
        return cls(
            use_restrict=False,
            use_collapse=False,
            schedule=None,
            num_threads=None,
            adaptive=False,
            extra_compile_flags=(),
        )


class OpenMPBackend(CppBackend):
    name = "openmp"

    def __init__(self, options: OpenMPOptions | None = None) -> None:
        self.options = options or OpenMPOptions()

    def compile(
        self, function: loop.Function, shapes: Shapes | None = None
    ) -> CompiledFunction:
        shared_library = compile_cpp(
            self.source(function, shapes=shapes),
            extra_flags=("-fopenmp", *self.options.extra_compile_flags),
        )
        return CompiledFunction(function=function, shared_library=shared_library)

    def _emit_field_param(self, field: Field) -> str:
        if not self.options.use_restrict:
            return super()._emit_field_param(field)
        return f"double* __restrict__ {field.name}"

    @staticmethod
    def _extent_of(expr, *, shapes: Shapes) -> int | None:
        """Constant-folds a loop bound expression against concrete shapes."""
        if isinstance(expr, loop.Literal):
            return int(expr.value)
        if isinstance(expr, loop.Extent):
            shape = shapes.get(expr.field)
            return None if shape is None else shape[expr.dim]
        if isinstance(expr, loop.BinaryExpr):
            lhs = OpenMPBackend._extent_of(expr.lhs, shapes=shapes)
            rhs = OpenMPBackend._extent_of(expr.rhs, shapes=shapes)
            if lhs is None or rhs is None:
                return None
            if expr.op is loop.BinaryOp.SUB:
                return lhs - rhs
            if expr.op is loop.BinaryOp.ADD:
                return lhs + rhs
            if expr.op is loop.BinaryOp.MUL:
                return lhs * rhs
        return None

    @staticmethod
    def _trip_count(statement: loop.For, *, shapes: Shapes | None) -> int | None:
        """Count loop iterations"""
        if shapes is None:
            return None
        upper = OpenMPBackend._extent_of(statement.upper, shapes=shapes)
        if upper is None:
            return None
        lower_expr = getattr(statement, "lower", None)
        lower = (
            OpenMPBackend._extent_of(lower_expr, shapes=shapes)
            if lower_expr is not None
            else 0
        )
        return upper - (lower or 0)

    def _emit_loop_prefix(
        self,
        statement: loop.For,
        *,
        indent: int,
        loop_depth: int,
        shapes: Shapes | None = None,
    ) -> list[str]:
        if loop_depth != 0:
            return []

        prefix = "    " * indent
        is_perfectly_nested = len(statement.body) == 1 and isinstance(
            statement.body[0], loop.For
        )

        collapse = self.options.use_collapse and is_perfectly_nested
        num_threads = self.options.num_threads
        schedule_chunk = self.options.schedule_chunk

        if self.options.adaptive and shapes is not None:
            outer_extent = self._trip_count(statement, shapes=shapes)
            inner_extent = (
                self._trip_count(statement.body[0], shapes=shapes)
                if is_perfectly_nested
                else None
            )
            if outer_extent is not None:
                config = parallel_config(
                    outer_extent,
                    inner_extent,
                    max_threads=self.options.num_threads or available_cores(),
                    min_iters_per_thread=self.options.min_iters_per_thread,
                    collapse_enabled=collapse,
                    schedule_kind=self.options.schedule,
                )
                if not config.parallelize:
                    return []  # zu wenig Arbeit: kein Pragma, Schleife bleibt seriell
                collapse = config.collapse
                num_threads = config.num_threads
                schedule_chunk = schedule_chunk or config.schedule_chunk

        clauses = ["parallel", "for"]
        if collapse:
            clauses.append("collapse(2)")
        if self.options.schedule:
            if schedule_chunk:
                clauses.append(f"schedule({self.options.schedule}, {schedule_chunk})")
            else:
                clauses.append(f"schedule({self.options.schedule})")
        if num_threads:
            clauses.append(f"num_threads({num_threads})")

        return [f"{prefix}#pragma omp {' '.join(clauses)}"]
