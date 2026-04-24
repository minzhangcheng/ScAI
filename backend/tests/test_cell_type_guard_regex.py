"""Ensure the manual cell-type assignment guard catches common alias columns."""
import pytest

from backend.agents.bioinformatics_agent import _CELL_TYPE_ASSIGN_RE


@pytest.mark.parametrize("code", [
    "adata.obs['cell_type'] = x",
    'adata.obs["cell_type"]=x',
    "adata.obs['celltype'] = x",
    "adata.obs['cluster_identity'] = x",
    "adata.obs['cell_identity'] = x",
    "adata.obs['cell_annotation'] = x",
    "adata.obs['annotation'] = x",
    "adata_hep.obs['cluster_identity'] = adata_hep.obs['leiden_broad'].map(m)",
])
def test_guard_matches(code):
    assert _CELL_TYPE_ASSIGN_RE.search(code) is not None, code


@pytest.mark.parametrize("code", [
    "adata.obs['leiden'] = x",
    "adata.obs['leiden_broad'] = x",
    "adata.obs['condition'] = x",
    "adata.obs['sample'] = x",
    "y = adata.obs['cell_type']",
    "adata.obs['cell_type'] == 'Hepatocyte'",
])
def test_guard_does_not_match(code):
    assert _CELL_TYPE_ASSIGN_RE.search(code) is None, code