from yasmin.backends.cpp import CppBackend
from yasmin.ir import loop
from yasmin.runtime.native import CompiledFunction, compile_cpp


class OpenMPBackend(CppBackend):
    name = "openmp"

    def compile(self, function: loop.Function) -> CompiledFunction:
        shared_library = compile_cpp(
            self.source(function),
            extra_flags=("-fopenmp",),
        )

        return CompiledFunction(function=function, shared_library=shared_library)

    def _emit_loop_prefix(
        self,
        statement: loop.For,
        *,
        indent: int,
        loop_depth: int,
    ) -> list[str]:
        if loop_depth != 0:
            return []

        prefix = "    " * indent
        return [f"{prefix}#pragma omp parallel for"]
