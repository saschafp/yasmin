from yasmin.backends.cpp import CppBackend
from yasmin.ir import loop


class OpenMPBackend(CppBackend):
    name = "openmp"

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
