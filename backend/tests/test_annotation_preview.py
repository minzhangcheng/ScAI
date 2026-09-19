"""Candidate review plots read exact sources without changing accepted data."""
import base64
import contextlib
import io
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from unittest.mock import AsyncMock

from backend.agents.bioinformatics.annotation_preview import render_candidate_annotation_preview
from backend.models.schemas import ClusterAnnotation


class PreviewSandbox:
    def __init__(self, namespace, monkeypatch):
        self.namespace = namespace
        self.images = []
        monkeypatch.setattr("IPython.display.display", lambda image: self.images.append(image.data))

    async def execute(self, code, **kwargs):
        assert kwargs == {"skip_safety_check": True, "mutates_namespace": False}
        self.images.clear()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(code, "<annotation-preview>", "exec"), self.namespace)
        return SimpleNamespace(error=None, output=output.getvalue(),
                               figures=[base64.b64encode(image).decode() for image in self.images])


def _adata():
    data = ad.AnnData(np.ones((6, 2)), obs=pd.DataFrame({
        "clusters": ["0", "0", "1", "1", "2", "2"],
        "cell_type": ["Previously accepted"] * 6,
    }, index=[str(i) for i in range(6)]))
    data.obsm["X_umap"] = np.arange(12).reshape(6, 2).astype(float)
    return data


@pytest.mark.asyncio
async def test_preview_reads_exact_object_preserves_data_and_refreshes_candidates(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    data = _adata()
    before_obs = data.obs.copy(deep=True)
    before_coordinates = data.obsm["X_umap"].copy()
    before_x = data.X.copy()
    namespace = {"chosen": data, "unrelated": _adata(), "__builtins__": __builtins__}
    sandbox = PreviewSandbox(namespace, monkeypatch)
    source = {"adata_var": "chosen", "cluster_key": "clusters"}
    candidates = [ClusterAnnotation(cluster_id=str(i), cell_type=f"Candidate {i}") for i in range(3)]
    first = await render_candidate_annotation_preview(sandbox, source, candidates)
    assert first["image"].startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(sandbox.images[0])) as image:
        assert image.width > 500 and image.height > 300
    candidates[0].cell_type = "Revised candidate"
    second = await render_candidate_annotation_preview(sandbox, source, candidates)
    assert first["image"] != second["image"]
    pd.testing.assert_frame_equal(data.obs, before_obs)
    np.testing.assert_array_equal(data.X, before_x)
    np.testing.assert_array_equal(data.obsm["X_umap"], before_coordinates)
    assert set(namespace) == {"chosen", "unrelated", "__builtins__"}


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["object", "coordinates", "clusters"])
async def test_missing_source_does_not_borrow_another_dataset_or_recompute(monkeypatch, missing):
    data = _adata()
    namespace = {"chosen": data, "unrelated": _adata()}
    if missing == "object":
        del namespace["chosen"]
    elif missing == "coordinates":
        del data.obsm["X_umap"]
    else:
        del data.obs["clusters"]
    result = await render_candidate_annotation_preview(
        PreviewSandbox(namespace, monkeypatch), {"adata_var": "chosen", "cluster_key": "clusters"}, [],
    )
    assert result["image"] is None
    assert result["note"]
    if missing == "coordinates":
        assert "X_umap" not in data.obsm


@pytest.mark.asyncio
async def test_unavailable_kernel_leaves_review_nonblocking():
    sandbox = SimpleNamespace(execute=AsyncMock(side_effect=RuntimeError("kernel unavailable")))
    result = await render_candidate_annotation_preview(
        sandbox, {"adata_var": "chosen", "cluster_key": "clusters"}, [],
    )
    assert result["image"] is None
