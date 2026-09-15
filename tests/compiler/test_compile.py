from unittest.mock import MagicMock

import numpy as np
import pytest

import yasmin as yasi
from yasmin.backends.cpp import CppBackend
from yasmin.backends.openmp import OpenMPBackend
from yasmin.compiler.compile import _clear_compilation_cache


def test_compile_cpp_returns_executable_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_compilation_cache()

    x = yasi.Dimension("x")

    u = yasi.Field("u", dims=(x,), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x,), dtype=yasi.float64)

    @yasi.operator
    def copy(u: yasi.Field, out: yasi.Field) -> None:
        out[0] = u[0]

    operator = copy(u, out)

    compiled = MagicMock()
    compile_mock = MagicMock(return_value=compiled)

    monkeypatch.setattr(
        CppBackend,
        "compile",
        compile_mock,
    )

    kernel = yasi.compile(
        operator,
        backend="cpp",
    )

    assert isinstance(kernel, yasi.Kernel)
    assert compile_mock.call_count == 1

    u_data = np.ones(16, dtype=np.float64)
    out_data = np.zeros_like(u_data)

    kernel(
        fields={
            u: u_data,
            out: out_data,
        },
    )

    assert compiled.call_count == 1

    call = compiled.call_args

    assert call.kwargs["fields"] == {
        u._core: u_data,
        out._core: out_data,
    }
    assert call.kwargs["scalars"] == {}


def test_compile_openmp_does_not_require_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_compilation_cache()

    x = yasi.Dimension("x")

    u = yasi.Field("u", dims=(x,), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x,), dtype=yasi.float64)

    @yasi.operator
    def copy(u: yasi.Field, out: yasi.Field) -> None:
        out[0] = u[0]

    operator = copy(u, out)

    compiled = MagicMock()
    compile_mock = MagicMock(return_value=compiled)

    monkeypatch.setattr(
        OpenMPBackend,
        "compile",
        compile_mock,
    )

    kernel = yasi.compile(
        operator,
        backend="openmp",
    )

    assert isinstance(kernel, yasi.Kernel)
    assert compile_mock.call_count == 1
