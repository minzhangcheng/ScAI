import pytest
from types import SimpleNamespace
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.agents.bioinformatics.executor_execute import record_active_adata_var
from backend.models.schemas import AnalysisState


def _agent(sandbox):
    a = BioinformaticsAgent.__new__(BioinformaticsAgent)
    a.sandbox = sandbox
    a.session_id = "s"
    return a


class _ExecSandbox:
    """Actually executes the introspection code against a real AnnData in a
    private namespace, so the embedded obs-column keyword lists (_cond/_samp/...)
    are exercised for real rather than mocked."""

    def __init__(self, adata):
        self._objects = adata if isinstance(adata, dict) else {"adata": adata}

    async def execute(
        self, code, records_history=True, skip_safety_check=False, **_kwargs
    ):
        import io
        import contextlib
        ns = dict(self._objects)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
        return SimpleNamespace(success=True, output=buf.getvalue(), error=None)


def _partition_objects():
    anndata = pytest.importorskip("anndata")
    import numpy as np
    import pandas as pd
    old = anndata.AnnData(
        np.zeros((2, 2)), obs=pd.DataFrame({"leiden_prelim": pd.Categorical(["0", "1"])}, index=["a", "b"]),
    )
    final = anndata.AnnData(
        np.zeros((4, 2)), obs=pd.DataFrame({"leiden": pd.Categorical(["0", "0", "1", "1"])}, index=["a", "b", "c", "d"]),
    )
    return old, final


@pytest.mark.asyncio
async def test_explicit_partition_ignores_earlier_subset_and_counts_same_source():
    old, final = _partition_objects()
    agent = _agent(_ExecSandbox({"ad": old, "adata_merged": final, "alias": final}))
    assert await agent._discover_adata_in_sandbox("leiden") == ("adata_merged", "leiden")
    source = await agent.resolve_analysis_source_metadata(cluster_key="leiden")
    assert source["var"] == "adata_merged"
    assert source["n_obs"] == 4
    proportions = await agent.compute_cluster_group_proportions(
        SimpleNamespace(treatment_groups=[]), annotation_source=source,
    )
    assert proportions["counts"] == {"0": 2, "1": 2}
    assert "cell_type" not in old.obs


@pytest.mark.asyncio
async def test_current_working_object_resolves_among_multiple_live_adata():
    old, final = _partition_objects()
    agent = _agent(_ExecSandbox({"adata": old, "adata_hep": final}))

    ambiguous = await agent.resolve_analysis_source_metadata(
        {"mode": "current"}, require_cluster_key=False,
    )
    assert ambiguous["candidate_names"] == ["adata", "adata_hep"]
    assert "multiple AnnData" in ambiguous["selection_error"]

    selected = await agent.resolve_analysis_source_metadata(
        {"mode": "current", "active_adata_var": "adata_hep"},
        require_cluster_key=False,
    )
    assert selected["var"] == "adata_hep"
    assert selected["cluster_key"] == "leiden"
    assert selected["candidate_names"] == ["adata_hep"]

    agent.sandbox.host_managed_global_names = lambda: ["adata_hep"]
    managed = await agent.resolve_analysis_source_metadata(
        {"mode": "current", "active_adata_var": "adata_hep"},
        require_cluster_key=False,
    )
    assert managed["var"] == "adata_hep"


@pytest.mark.asyncio
async def test_named_unclustered_adata_is_available_for_nonclustering_followup():
    old, _ = _partition_objects()
    old.obs.drop(columns="leiden_prelim", inplace=True)
    agent = _agent(_ExecSandbox({"adata": old}))

    selected = await agent.resolve_analysis_source_metadata(
        {"mode": "current", "active_adata_var": "adata"},
        require_cluster_key=False,
    )
    assert selected["var"] == "adata"
    assert selected["cluster_key"] is None
    assert (await agent.resolve_analysis_source_metadata(
        {"mode": "current", "active_adata_var": "adata"},
    ))["selection_error"] == "preferred AnnData is unavailable or has no clustering key"


@pytest.mark.asyncio
async def test_declared_working_object_updates_persisted_source_context():
    old, final = _partition_objects()
    agent = _agent(_ExecSandbox({"adata": old, "adata_hep": final}))
    state = AnalysisState(id="s", user_id="u")
    context = SimpleNamespace(resolved_adata_source_context={
        "mode": "current", "active_adata_var": "adata",
        "cluster_key": "leiden_prelim", "partition_digest": "old",
    })

    assert await record_active_adata_var(agent, state, context, "adata_hep")
    assert context.resolved_adata_source_context["active_adata_var"] == "adata_hep"
    assert "cluster_key" not in context.resolved_adata_source_context
    assert state.sandbox_state["resolved_adata_source_context_v1"] == context.resolved_adata_source_context
    assert not await record_active_adata_var(agent, state, context, "missing")
    assert state.sandbox_state["resolved_adata_source_context_v1"]["active_adata_var"] == "adata_hep"


@pytest.mark.asyncio
async def test_exact_preparation_resolves_managed_binding_and_rejects_rebound_object():
    old, final = _partition_objects()
    sandbox = _ExecSandbox({"ad": old, "adata_merged": final.copy(), "scai_work_1": final, "scai_anchor_1": final})
    sandbox.host_managed_global_names = lambda: ["scai_work_1", "scai_anchor_1"]
    agent = _agent(sandbox)
    agent._current_analysis_state = SimpleNamespace(adata_source_preparations=[SimpleNamespace(
        preparation_id="prep_1", binding_identity_status="exact", binding_name="scai_work_1", anchor_name="scai_anchor_1",
        binding_id="binding_1", adata_lineage_id="lineage_1", loaded_checkpoint_id="checkpoint_1",
    )])
    metadata = await agent.resolve_analysis_source_metadata({"preparation_id": "prep_1"}, cluster_key="leiden")
    assert metadata["var"] == "scai_work_1"
    assert metadata["preparation_id"] == "prep_1"
    sandbox._objects["scai_work_1"] = final.copy()
    metadata = await agent.resolve_analysis_source_metadata({"preparation_id": "prep_1"}, cluster_key="leiden")
    assert not metadata.get("var")
    assert "anchor" in metadata["selection_error"]
    metadata = await agent.resolve_analysis_source_metadata(
        {"preparation_id": "prep_1"}, cluster_key="leiden", permit_evolved_binding=True,
    )
    assert metadata["var"] == "scai_work_1"
    assert metadata["binding_identity_status"] == "evolved"
    metadata = await agent.resolve_analysis_source_metadata({"preparation_id": "missing"}, cluster_key="leiden")
    assert "missing" in metadata["selection_error"]
