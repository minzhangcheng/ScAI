"""Arboreto's exact empty metadata call remains usable on current Dask."""

from __future__ import annotations

import os
from pathlib import Path
import uuid

import pytest

from backend.utils.sandbox import LocalJupyterSandbox


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
            records_history=False,
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
            records_history=False,
            skip_safety_check=True,
        )
        assert result.success, result.error
    finally:
        await sandbox.stop(save=False)
