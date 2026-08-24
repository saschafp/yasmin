from yasmin.core import Field
from yasmin.ir import loop
from yasmin.runtime.native import CompiledFunction, compile_cpp


class CppBackend:
    name = "cpp"

    def source(self, function: loop.Function) -> str:
        lines: list[str] = []

        params = self._emit_parameters(function)

        lines.append(f'extern "C" void {function.name}({params}) {{')

        for statement in function.body:
            lines.extend(self._emit_stmt(statement, indent=1))

        lines.append("}")

        return "\n".join(lines)

    def compile(self, function: loop.Function) -> CompiledFunction:
        shared_library = compile_cpp(self.source(function))

        return CompiledFunction(function=function, shared_library=shared_library)

    def _emit_loop_prefix(
        self,
        statement: loop.For,
        *,
        indent: int,
        loop_depth: int,
    ) -> list[str]:
        return []

    def _emit_parameters(self, function: loop.Function) -> str:
        params: list[str] = []

        for field in function.fields:
            params.append(f"double* {field.name}")

        for field in function.fields:
            for dim in range(len(field.dims)):
                params.append(f"int {field.name}_shape_{dim}")

        for scalar in function.scalars:
            params.append(f"double {scalar.name}")

        return ", ".join(params)

    def _emit_stmt(
        self,
        statement: loop.Stmt,
        *,
        indent: int,
        loop_depth: int = 0,
    ) -> list[str]:
        prefix = "    " * indent

        match statement:
            case loop.Store(field=field, indices=indices, value=value):
                index = self._emit_flat_index(field, indices)
                return [f"{prefix}{field.name}[{index}] = {self._emit_expr(value)};"]

            case loop.For(index=index, lower=lower, upper=upper, body=body):
                lines = self._emit_loop_prefix(
                    statement, indent=indent, loop_depth=loop_depth
                )

                lines.append(
                    f"{prefix}for (int {index.name} = "
                    f"{self._emit_expr(lower)}; "
                    f"{index.name} < {self._emit_expr(upper)}; "
                    f"++{index.name}) {{"
                )

                for child in body:
                    lines.extend(
                        self._emit_stmt(
                            child, indent=indent + 1, loop_depth=loop_depth + 1
                        )
                    )

                lines.append(f"{prefix}}}")
                return lines

            case _:
                raise TypeError(
                    f"Unsupported loop statement type: {type(statement).__name__}"
                )

    def _emit_expr(self, expr: loop.Expr) -> str:
        match expr:
            case loop.Literal(value=value):
                return repr(value)

            case loop.ScalarRef(scalar=scalar):
                return scalar.name

            case loop.Index(name=name):
                return name

            case loop.Extent(field=field, dim=dim):
                return f"{field.name}_shape_{dim}"

            case loop.Load(field=field, indices=indices):
                index = self._emit_flat_index(field, indices)
                return f"{field.name}[{index}]"

            case loop.BinaryExpr(op=op, lhs=lhs, rhs=rhs):
                return f"({self._emit_expr(lhs)} {op.value} {self._emit_expr(rhs)})"

            case _:
                raise TypeError(
                    f"Unsupported loop expression type: {type(expr).__name__}"
                )

    def _emit_flat_index(
        self,
        field: Field,
        indices: tuple[loop.Expr, ...],
    ) -> str:
        if len(indices) == 1:
            return self._emit_expr(indices[0])

        if len(indices) == 2:
            i = self._emit_expr(indices[0])
            j = self._emit_expr(indices[1])

            return f"({i} * {field.name}_shape_1 + {j})"

        raise NotImplementedError(
            "C++ backend currently supports only 1D and 2D fields"
        )
