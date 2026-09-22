"""Runtime import contracts executed only inside the immutable test image."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version


def test_locked_runtime_imports_and_zarr_range():
    for module_name in (
        "fastapi",
        "langgraph",
        "scanpy",
        "anndata",
        "zarr",
        "dask_image",
        "h5py",
        "jupyter_client",
        "pytest",
        "pytest_asyncio",
        "pytest_benchmark",
    ):
        assert import_module(module_name)
    parts = tuple(int(piece) for piece in version("zarr").split(".")[:2])
    assert (2, 18) <= parts < (3, 0)


def test_nonhuman_network_translation_dependency_chain(monkeypatch):
    """Exercise the failed dependency path without downloading orthology data."""
    import pandas as pd
    from decoupler.omnip import _check_if_pypath, translate_net
    from pypath.utils import orthology, taxonomy

    _check_if_pypath()
    assert version("pypath-omnipath") == "0.16.17"
    network = pd.DataFrame({"source": ["pathway"], "target": ["GENE"], "weight": [0.75]})
    calls = []

    def translate_df(*, df, target, cols, source):
        calls.append((target, cols, source))
        result = df.copy()
        result["target"] = ["Gene"]
        return result

    monkeypatch.setattr(taxonomy, "ensure_ncbi_tax_id", lambda organism: organism)
    monkeypatch.setattr(orthology, "translate_df", translate_df)
    translated = translate_net(
        network, columns="target", source_organism=9606, target_organism=10090,
    )
    assert calls == [(10090, {"target": "genesymbol"}, 9606)]
    assert translated["target"].tolist() == ["Gene"]
    assert translated["weight"].tolist() == [0.75]
    assert network["target"].tolist() == ["GENE"]
