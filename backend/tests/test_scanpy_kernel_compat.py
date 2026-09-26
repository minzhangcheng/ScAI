"""Scanpy Scrublet layer compatibility must preserve AnnData identity."""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest


class _Matrix:
    def __init__(self, name: str, values=(1, 2, 3)):
        self.name = name
        self.values = tuple(values)

    def copy(self):
        return _Matrix(self.name, self.values)


class _Adata:
    def __init__(self):
        self.X = _Matrix("normalized", (9, 8, 7))
        self.layers = {"counts": _Matrix("counts", (1, 2, 3))}
        self.obs = {}

    def copy(self):
        copied = _Adata()
        copied.X = self.X.copy()
        copied._pre_swap_x = copied.X
        copied.layers = {key: value.copy() for key, value in self.layers.items()}
        copied.obs = dict(self.obs)
        return copied


def test_copy_false_restores_exact_x_identity_on_success_and_failure(monkeypatch):
    seen = []
    state = {"raises": False}

    def scrublet(adata, adata_sim=None, *, copy=False, random_state=0):
        seen.append((adata.X, copy, random_state))
        adata.obs["doublet_score"] = 0.2
        if state["raises"]:
            raise RuntimeError("scrublet failed")
        return "native-result"

    module = ModuleType("scanpy")
    module.pp = SimpleNamespace(scrublet=scrublet)
    monkeypatch.setitem(sys.modules, "scanpy", module)
    from backend.utils.kernel_startup import scai_scanpy_compat

    scai_scanpy_compat.install()
    adata = _Adata()
    original_x = adata.X
    original_counts = adata.layers["counts"]
    original_count_values = original_counts.values
    module.pp.scrublet(adata, layer="counts", random_state=3)
    assert seen[0][1:] == (False, 3)
    assert seen[0][0] is not original_x
    assert seen[0][0] is not original_counts
    assert seen[0][0].name == "counts"
    assert seen[0][0].values == original_count_values
    assert adata.X is original_x
    assert adata.layers["counts"] is original_counts
    assert adata.layers["counts"].values == original_count_values
    assert adata.obs["doublet_score"] == 0.2

    state["raises"] = True
    with pytest.raises(RuntimeError, match="scrublet failed"):
        module.pp.scrublet(adata, layer="counts")
    assert seen[1][0] is not original_counts
    assert seen[1][0].values == original_count_values
    assert adata.X is original_x
    assert adata.layers["counts"] is original_counts
    assert adata.layers["counts"].values == original_count_values


@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
@pytest.mark.asyncio
async def test_live_scanpy_scrublet_layer_preserves_x_and_copy_contract(
    tmp_path, monkeypatch
):
    from backend.utils.sandbox import LocalJupyterSandbox

    monkeypatch.setenv("NUMBA_DISABLE_JIT", "0")
    sandbox = LocalJupyterSandbox(
        session_id="s2-scanpy-live",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    await sandbox.start()
    try:
        result = await sandbox.execute(
            "import anndata as ad\n"
            "import inspect\n"
            "import numpy as np\n"
            "import scanpy as sc\n"
            "assert getattr(sc.pp.scrublet, '_scai_layer_compat', False)\n"
            "layer_parameter = inspect.signature(sc.pp.scrublet).parameters['layer']\n"
            "assert layer_parameter.kind is inspect.Parameter.KEYWORD_ONLY\n"
            "counts = (np.arange(200, dtype=float).reshape(20, 10) % 17) + 1\n"
            "adata = ad.AnnData(counts / 10)\n"
            "adata.layers['counts'] = counts.copy()\n"
            "original_x = adata.X\n"
            "original_layer = adata.layers['counts']\n"
            "original_layer_values = original_layer.copy()\n"
            "sc.pp.scrublet(adata, layer='counts', copy=False, random_state=0, "
            "n_prin_comps=5)\n"
            "assert adata.X is original_x\n"
            "assert adata.layers['counts'] is original_layer\n"
            "assert np.array_equal(adata.layers['counts'], original_layer_values)\n"
            "assert 'doublet_score' in adata.obs\n"
            "input_obs = adata.obs.copy()\n"
            "input_x = adata.X\n"
            "input_layer = adata.layers['counts']\n"
            "input_layer_values = input_layer.copy()\n"
            "returned = sc.pp.scrublet(adata, layer='counts', copy=True, "
            "random_state=0, n_prin_comps=5)\n"
            "assert returned is not adata\n"
            "assert returned.X is not adata.X\n"
            "assert adata.X is input_x\n"
            "assert adata.obs.equals(input_obs)\n"
            "assert adata.layers['counts'] is input_layer\n"
            "assert np.array_equal(adata.layers['counts'], input_layer_values)\n"
            "assert 'doublet_score' in returned.obs\n",
            records_history=False,
            skip_safety_check=True,
        )
        assert result.success, result.error
    finally:
        await sandbox.stop(save=False)


def test_installer_distinguishes_missing_top_level_from_internal_module_error(monkeypatch):
    import builtins
    from backend.utils.kernel_startup import scai_scanpy_compat

    monkeypatch.setitem(sys.modules, "scanpy", None)
    assert scai_scanpy_compat.install() is False

    native_import = builtins.__import__

    def broken_scanpy_import(name, *args, **kwargs):
        if name == "scanpy":
            raise ModuleNotFoundError(
                "No module named 'numpy'", name="numpy"
            )
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_scanpy_import)
    with pytest.raises(ModuleNotFoundError) as missing_submodule:
        scai_scanpy_compat.install()
    assert missing_submodule.value.name == "numpy"

    monkeypatch.setattr(builtins, "__import__", native_import)
    monkeypatch.setitem(sys.modules, "scanpy", SimpleNamespace(pp=SimpleNamespace()))
    with pytest.raises(AttributeError):
        scai_scanpy_compat.install()
