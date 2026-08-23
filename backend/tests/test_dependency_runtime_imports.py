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
