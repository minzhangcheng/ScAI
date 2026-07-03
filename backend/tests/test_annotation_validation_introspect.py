import json
import pytest
from types import SimpleNamespace
from backend.agents.bioinformatics_agent import BioinformaticsAgent


class _FakeSandbox:
    def __init__(self, output):
        self._output = output
        self.last_code = None

    async def execute(self, code, save_to_history=True, skip_safety_check=False):
        self.last_code = code
        return SimpleNamespace(success=True, output=self._output, error=None)


def _agent(sandbox):
    a = BioinformaticsAgent.__new__(BioinformaticsAgent)
    a.sandbox = sandbox
    a.session_id = "s"
    return a


@pytest.mark.asyncio
async def test_introspect_parses_enriched_metadata():
    payload = {"var": "adata", "cluster_key": "leiden",
               "candidate_cluster_keys": ["leiden", "leiden_r1"],
               "obs_columns": ["leiden", "condition", "doublet_score"],
               "qc_like": ["doublet_score"], "condition_like": ["condition"],
               "sample_like": [], "batch_like": [],
               "obs_summaries": {"condition": {"n_categories": 2, "top": ["Con", "DKO"]}},
               "layers": ["counts"], "n_obs": 18233, "n_vars": 32000, "has_raw": True}
    a = _agent(_FakeSandbox(json.dumps(payload)))
    ctx = await a.introspect_validation_metadata("leiden")
    assert ctx["candidate_cluster_keys"] == ["leiden", "leiden_r1"]
    assert ctx["obs_summaries"]["condition"]["n_categories"] == 2
    assert ctx["has_raw"] is True


class _ExecSandbox:
    """Actually executes the introspection code against a real AnnData in a
    private namespace, so the embedded obs-column keyword lists (_cond/_samp/...)
    are exercised for real rather than mocked."""

    def __init__(self, adata):
        self._adata = adata

    async def execute(self, code, save_to_history=True, skip_safety_check=False):
        import io
        import contextlib
        ns = {"adata": self._adata}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
        return SimpleNamespace(success=True, output=buf.getvalue(), error=None)


@pytest.mark.asyncio
async def test_introspect_detects_group_named_condition_column():
    """A column literally named 'group' must be recognized as a condition-like
    (experimental grouping) column, not missed. Regression for the proportion
    feature degrading to overall when the group column is named 'group'."""
    anndata = pytest.importorskip("anndata")
    import numpy as np
    import pandas as pd

    obs = pd.DataFrame({
        "leiden": pd.Categorical(["0", "1", "0", "1"]),
        "group": pd.Categorical(["Control", "NASH", "Control", "NASH"]),
    })
    adata = anndata.AnnData(X=np.zeros((4, 3), dtype="float32"), obs=obs)

    ctx = await _agent(_ExecSandbox(adata)).introspect_validation_metadata("leiden")
    assert "group" in ctx["condition_like"]
    assert ctx["obs_summaries"]["group"]["n_categories"] == 2


@pytest.mark.asyncio
async def test_introspect_empty_on_no_anndata():
    assert await _agent(_FakeSandbox("{}")).introspect_validation_metadata(None) == {}


@pytest.mark.asyncio
async def test_introspect_empty_on_garbage():
    assert await _agent(_FakeSandbox("not json")).introspect_validation_metadata(None) == {}
