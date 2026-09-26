"""PyDESeq2 must not launch local processes inside a ScAI kernel."""

from __future__ import annotations

import os
import sys
from types import ModuleType

import pytest


@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
@pytest.mark.asyncio
async def test_live_pydeseq2_minimal_inference_uses_threads(tmp_path):
    from backend.utils.sandbox import LocalJupyterSandbox

    sandbox = LocalJupyterSandbox(
        session_id="s2-pydeseq2-live",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    await sandbox.start()
    try:
        result = await sandbox.execute(
            "import joblib\n"
            "import pydeseq2.default_inference as inference_module\n"
            "assert getattr(inference_module.parallel_backend, "
            "'_scai_thread_backend', False)\n"
            "import pandas as pd\n"
            "from pydeseq2.dds import DeseqDataSet\n"
            "from pydeseq2.default_inference import DefaultInference\n"
            "global_parallel_backend = joblib.parallel_backend\n"
            "counts = pd.DataFrame(\n"
            "  [[100, 50, 20, 80], [110, 48, 22, 75], [95, 52, 19, 82],\n"
            "   [200, 25, 40, 40], [190, 27, 38, 42], [210, 24, 41, 39]],\n"
            "  index=['sample_a1', 'sample_a2', 'sample_a3', 'sample_b1', "
            "'sample_b2', 'sample_b3'],\n"
            "  columns=['gene_a', 'gene_b', 'gene_c', 'gene_d'], dtype=int)\n"
            "metadata = pd.DataFrame(\n"
            "  {'condition': ['a', 'a', 'a', 'b', 'b', 'b']},\n"
            "  index=counts.index)\n"
            "inference = DefaultInference(n_cpus=2)\n"
            "dataset = DeseqDataSet(\n"
            "  counts=counts, metadata=metadata, design='~condition',\n"
            "  refit_cooks=False, inference=inference, quiet=True)\n"
            "dataset.deseq2()\n"
            "assert joblib.parallel_backend is global_parallel_backend\n",
            records_history=False,
            skip_safety_check=True,
        )
        assert result.success, result.error
    finally:
        await sandbox.stop(save=False)


def test_installer_distinguishes_missing_top_level_from_internal_module_error(monkeypatch):
    import builtins
    from backend.utils.kernel_startup import scai_pydeseq2_compat

    native_import = builtins.__import__

    def missing_top_level(name, *args, **kwargs):
        if name == "pydeseq2":
            raise ModuleNotFoundError(
                "No module named 'pydeseq2'", name="pydeseq2"
            )
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_top_level)
    monkeypatch.delitem(sys.modules, "pydeseq2.default_inference", raising=False)
    assert scai_pydeseq2_compat.install() is False

    monkeypatch.setattr(builtins, "__import__", native_import)
    package = ModuleType("pydeseq2")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "pydeseq2", package)
    monkeypatch.delitem(sys.modules, "pydeseq2.default_inference", raising=False)
    with pytest.raises(ModuleNotFoundError) as missing_submodule:
        scai_pydeseq2_compat.install()
    assert missing_submodule.value.name == "pydeseq2.default_inference"

    inference = ModuleType("pydeseq2.default_inference")
    package.default_inference = inference
    monkeypatch.setitem(sys.modules, "pydeseq2.default_inference", inference)
    with pytest.raises(AttributeError):
        scai_pydeseq2_compat.install()
