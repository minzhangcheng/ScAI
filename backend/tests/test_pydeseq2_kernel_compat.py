"""PyDESeq2 must not launch local processes inside a ScAI kernel."""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from types import ModuleType

import pytest


def _fake_inference(monkeypatch, original):
    package = ModuleType("pydeseq2")
    package.__path__ = []
    inference = ModuleType("pydeseq2.default_inference")
    inference.parallel_backend = original
    package.default_inference = inference
    monkeypatch.setitem(sys.modules, "pydeseq2", package)
    monkeypatch.setitem(sys.modules, "pydeseq2.default_inference", inference)
    return inference


def test_installer_patches_only_pydeseq2_parallel_backend(monkeypatch):
    from joblib import parallel_backend as global_parallel_backend

    @contextmanager
    def original(backend, *args, **kwargs):
        yield (backend, args, kwargs)

    inference = _fake_inference(monkeypatch, original)
    from backend.utils.kernel_startup import scai_pydeseq2_compat

    assert scai_pydeseq2_compat.install() is True
    assert inference.parallel_backend is not original
    assert global_parallel_backend is not inference.parallel_backend


def test_process_backends_route_to_threading_without_inner_thread_option(monkeypatch):
    calls = []

    @contextmanager
    def original(backend, *args, **kwargs):
        calls.append((backend, args, kwargs))
        yield "active"

    inference = _fake_inference(monkeypatch, original)
    from backend.utils.kernel_startup import scai_pydeseq2_compat

    assert scai_pydeseq2_compat.install() is True
    for backend in ("LOKY", "multiprocessing"):
        with inference.parallel_backend(
            backend, "positional", n_jobs=2, inner_max_num_threads=1,
            unrelated=True,
        ) as active:
            assert active == "active"

    assert calls == [
        ("threading", ("positional",), {"n_jobs": 2, "unrelated": True}),
        ("threading", ("positional",), {"n_jobs": 2, "unrelated": True}),
    ]


def test_nonprocess_backend_preserves_arguments_options_and_context_result(monkeypatch):
    calls = []

    @contextmanager
    def original(backend, *args, **kwargs):
        calls.append((backend, args, kwargs))
        yield {"context": "preserved"}

    inference = _fake_inference(monkeypatch, original)
    from backend.utils.kernel_startup import scai_pydeseq2_compat

    assert scai_pydeseq2_compat.install() is True
    with inference.parallel_backend(
        "custom", "positional", n_jobs=3, inner_max_num_threads=7,
        feature=True,
    ) as active:
        assert active == {"context": "preserved"}
    assert calls == [
        (
            "custom", ("positional",),
            {"n_jobs": 3, "inner_max_num_threads": 7, "feature": True},
        )
    ]


def test_installer_is_idempotent_and_missing_dependency_is_nonfatal(monkeypatch):
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
    monkeypatch.delitem(sys.modules, "pydeseq2", raising=False)
    monkeypatch.delitem(sys.modules, "pydeseq2.default_inference", raising=False)
    assert scai_pydeseq2_compat.install() is False

    monkeypatch.setattr(builtins, "__import__", native_import)

    @contextmanager
    def original(backend, *args, **kwargs):
        yield None

    inference = _fake_inference(monkeypatch, original)
    assert scai_pydeseq2_compat.install() is True
    wrapper = inference.parallel_backend
    assert scai_pydeseq2_compat.install() is True
    assert inference.parallel_backend is wrapper


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
            save_to_history=False,
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
