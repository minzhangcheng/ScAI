"""Arboreto's exact empty metadata call remains usable on current Dask."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import uuid

import pytest

from backend.utils.sandbox import LocalJupyterSandbox


def _fake_arboreto(monkeypatch, core):
    package = ModuleType("arboreto")
    package.__path__ = []
    package.core = core
    dask = ModuleType("dask")
    dask.__path__ = []
    dataframe = ModuleType("dask.dataframe")
    dask.dataframe = dataframe
    monkeypatch.setitem(sys.modules, "arboreto", package)
    monkeypatch.setitem(sys.modules, "arboreto.core", core)
    monkeypatch.setitem(sys.modules, "dask", dask)
    monkeypatch.setitem(sys.modules, "dask.dataframe", dataframe)
    return dataframe


def test_wrapper_intercepts_only_empty_list_with_identical_metadata_schema():
    import pandas as pd
    from backend.utils.kernel_startup import scai_arboreto_compat

    schema = pd.DataFrame(
        {"target": pd.Series([], dtype="object"), "n_estimators": pd.Series([], dtype="int64")}
    )
    equal_schema = schema.copy()
    original_calls = []
    empty_result = object()

    def original(delayed_values, *args, **kwargs):
        original_calls.append((delayed_values, args, kwargs))
        return "native"

    def from_pandas(frame, *, npartitions):
        assert frame.equals(schema)
        assert frame is not schema
        assert npartitions == 1
        return empty_result

    wrapped = scai_arboreto_compat._wrapper(original, schema, from_pandas)

    assert wrapped([], meta=schema) is empty_result
    assert wrapped([object()], meta=schema) == "native"
    assert wrapped([], meta=equal_schema) == "native"
    assert wrapped((), meta=schema) == "native"
    assert wrapped([], schema) == "native"
    assert len(original_calls) == 4


def test_wrapper_preserves_nonmatching_values_meta_and_native_errors():
    import pandas as pd
    from backend.utils.kernel_startup import scai_arboreto_compat

    schema = pd.DataFrame({"target": pd.Series([], dtype="object")})
    equal_schema = schema.copy()

    def original(delayed_values, *args, **kwargs):
        if delayed_values == ["error"]:
            raise RuntimeError("native arboreto failure")
        return delayed_values, args, kwargs

    wrapped = scai_arboreto_compat._wrapper(original, schema, lambda *_a, **_k: None)
    assert wrapped(["value"], meta=schema) == (["value"], (), {"meta": schema})
    assert wrapped([], meta=equal_schema) == ([], (), {"meta": equal_schema})
    assert wrapped([], schema) == ([], (schema,), {})
    with pytest.raises(RuntimeError, match="native arboreto failure"):
        wrapped(["error"], meta=schema)


def test_installer_probes_backend_and_is_noop_when_empty_call_is_supported(monkeypatch):
    from backend.utils.kernel_startup import scai_arboreto_compat

    schema = object()
    calls = []

    def compatible(delayed_values, *args, **kwargs):
        calls.append((delayed_values, args, kwargs))
        return "supported"

    core = ModuleType("arboreto.core")
    core._META_SCHEMA = schema
    core.from_delayed = compatible
    dataframe = _fake_arboreto(monkeypatch, core)
    dataframe.from_pandas = lambda *_a, **_k: object()

    assert scai_arboreto_compat.install() is True
    assert core.from_delayed is compatible
    assert calls == [([], (), {"meta": schema})]


def test_installer_activates_only_for_exact_empty_delayed_error_and_is_idempotent(
    monkeypatch,
):
    from backend.utils.kernel_startup import scai_arboreto_compat

    schema = SimpleNamespace(copy=lambda: "schema-copy")
    marker = object()

    def rejecting(delayed_values, *args, **kwargs):
        if not delayed_values and kwargs.get("meta") is schema:
            raise TypeError("Must supply at least one delayed object")
        return marker

    core = ModuleType("arboreto.core")
    core._META_SCHEMA = schema
    core.from_delayed = rejecting
    dataframe = _fake_arboreto(monkeypatch, core)
    dataframe.from_pandas = lambda frame, *, npartitions: (frame, npartitions)

    assert scai_arboreto_compat.install() is True
    installed = core.from_delayed
    assert getattr(installed, "_scai_empty_meta_compat", False) is True
    assert scai_arboreto_compat.install() is True
    assert core.from_delayed is installed
    assert installed([], meta=schema) == ("schema-copy", 1)
    assert installed([object()], meta=schema) is marker

    def unexpected(_values, *args, **kwargs):
        raise TypeError("unexpected backend failure")

    core.from_delayed = unexpected
    with pytest.raises(TypeError, match="unexpected backend failure"):
        scai_arboreto_compat.install()


@pytest.mark.parametrize(
    "module_name",
    ["scai_pydeseq2_compat", "scai_scanpy_compat", "scai_arboreto_compat"],
)
@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
@pytest.mark.asyncio
async def test_live_trusted_adapter_modules_cannot_be_shadowed(tmp_path, module_name):
    sandbox = LocalJupyterSandbox(
        session_id=f"trusted-adapter-{uuid.uuid4().hex[:8]}",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    shadow = sandbox.working_dir / f"{module_name}.py"
    sentinel = sandbox.working_dir / f"{module_name}.sentinel"
    shadow.write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('shadow executed')\n",
        encoding="utf-8",
    )
    await sandbox.start()
    try:
        result = await sandbox.execute(
            f"import {module_name}\n"
            f"print('__SCAI_ADAPTER_ORIGIN__' + str({module_name}.__file__))\n",
            save_to_history=False,
            skip_safety_check=True,
        )
        assert result.success, result.error
        assert not sentinel.exists()
        origin = next(
            line.removeprefix("__SCAI_ADAPTER_ORIGIN__")
            for line in result.output.splitlines()
            if line.startswith("__SCAI_ADAPTER_ORIGIN__")
        )
        expected_startup = (
            Path(__file__).resolve().parents[1] / "utils" / "kernel_startup"
        ).resolve()
        assert Path(origin).resolve().parent == expected_startup
        assert Path(origin).resolve() != shadow.resolve()
    finally:
        await sandbox.stop(save=False)


def test_installer_distinguishes_missing_top_level_from_internal_module_error(monkeypatch):
    import builtins
    from backend.utils.kernel_startup import scai_arboreto_compat

    native_import = builtins.__import__

    def missing_arboreto(name, *args, **kwargs):
        if name.startswith("arboreto"):
            raise ModuleNotFoundError(
                "No module named 'arboreto'", name="arboreto"
            )
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_arboreto)
    monkeypatch.delitem(sys.modules, "arboreto.core", raising=False)
    assert scai_arboreto_compat.install() is False

    monkeypatch.setattr(builtins, "__import__", native_import)
    core = ModuleType("arboreto.core")
    core._META_SCHEMA = object()
    dataframe = _fake_arboreto(monkeypatch, core)
    dataframe.from_pandas = lambda *_a, **_k: object()
    package = sys.modules["arboreto"]
    monkeypatch.delitem(sys.modules, "arboreto.core", raising=False)
    delattr(package, "core")
    with pytest.raises(ModuleNotFoundError) as missing_submodule:
        scai_arboreto_compat.install()
    assert missing_submodule.value.name == "arboreto.core"

    core = ModuleType("arboreto.core")
    core._META_SCHEMA = object()
    dataframe = _fake_arboreto(monkeypatch, core)
    dataframe.from_pandas = lambda *_a, **_k: object()
    with pytest.raises(AttributeError):
        scai_arboreto_compat.install()

    def missing_dask(name, *args, **kwargs):
        if name.startswith("dask"):
            raise ModuleNotFoundError("No module named 'dask'", name="dask")
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_dask)
    assert scai_arboreto_compat.install() is False


@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
@pytest.mark.asyncio
async def test_live_arboreto_grnboost2_uses_locked_backend(tmp_path):
    sandbox = LocalJupyterSandbox(
        session_id=f"arboreto-live-{uuid.uuid4().hex[:8]}",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    await sandbox.start()
    try:
        result = await sandbox.execute(
            "import numpy as np\n"
            "from arboreto.algo import grnboost2\n"
            "expression = np.arange(80, dtype=float).reshape(16, 5)\n"
            "links = grnboost2(expression, ['G1', 'G2', 'G3', 'G4', 'G5'], "
            "['G1', 'G2'], seed=0)\n"
            "assert len(links) >= 0\n",
            save_to_history=False,
            skip_safety_check=True,
        )
        assert result.success, result.error
    finally:
        await sandbox.stop(save=False)
