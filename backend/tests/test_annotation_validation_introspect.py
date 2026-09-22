import json
import pytest
from types import SimpleNamespace
from backend.agents.bioinformatics_agent import BioinformaticsAgent


class _FakeSandbox:
    def __init__(self, output):
        self._output = output
        self.last_code = None

    async def execute(
        self, code, records_history=True, skip_safety_check=False, **_kwargs
    ):
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
async def test_introspect_prefers_canonical_name_over_globals_order():
    anndata = pytest.importorskip("anndata")
    import numpy as np
    import pandas as pd

    def _adata(groups):
        return anndata.AnnData(
            X=np.zeros((len(groups), 2), dtype="float32"),
            obs=pd.DataFrame({
                "leiden": pd.Categorical([str(i % 2) for i in range(len(groups))]),
                "condition": pd.Categorical(groups),
            }),
        )

    sandbox = _ExecSandbox({
        "dko": _adata(["DKO", "DKO"]),
        "adata": _adata(["Con", "DKO", "Con", "DKO"]),
    })
    ctx = await _agent(sandbox).introspect_validation_metadata(
        None, preferred_var_name="adata", require_preferred=True,
    )

    assert ctx["var"] == "adata"
    assert ctx["selection_source"] == "preferred"
    assert set(ctx["candidate_names"]) == {"dko", "adata"}
    assert ctx["obs_summaries"]["condition"]["n_categories"] == 2


@pytest.mark.asyncio
async def test_introspect_does_not_guess_among_legacy_candidates():
    anndata = pytest.importorskip("anndata")
    import numpy as np
    import pandas as pd

    obj = anndata.AnnData(
        X=np.zeros((2, 2), dtype="float32"),
        obs=pd.DataFrame({"leiden": pd.Categorical(["0", "1"])}),
    )
    ctx = await _agent(_ExecSandbox({"left": obj, "right": obj.copy()})).introspect_validation_metadata(
        None, preferred_var_name="adata", require_preferred=False,
    )

    assert not ctx.get("var")
    assert ctx["candidate_names"] == ["left", "right"]
    assert "multiple AnnData" in ctx["selection_error"]


@pytest.mark.asyncio
async def test_introspect_empty_on_no_anndata():
    assert await _agent(_FakeSandbox("{}")).introspect_validation_metadata(None) == {}


@pytest.mark.asyncio
async def test_introspect_empty_on_garbage():
    assert await _agent(_FakeSandbox("not json")).introspect_validation_metadata(None) == {}


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
async def test_explicit_cluster_missing_never_falls_back_and_ambiguity_never_guesses():
    old, final = _partition_objects()
    assert await _agent(_ExecSandbox({"ad": old}))._discover_adata_in_sandbox("leiden") == (None, None)
    agent = _agent(_ExecSandbox({"left": final, "right": final.copy()}))
    metadata = await agent.introspect_validation_metadata("leiden")
    assert "multiple AnnData" in metadata["selection_error"]
    metadata = await agent.introspect_validation_metadata("leiden", preferred_var_name="right", require_preferred=True)
    assert metadata["var"] == "right"


@pytest.mark.asyncio
async def test_preferred_alias_is_resolved_and_partition_changes_invalidate_identity():
    _, final = _partition_objects()
    agent = _agent(_ExecSandbox({"first": final, "alias": final}))
    before = await agent.resolve_analysis_source_metadata(cluster_key="leiden", preferred_var_name="alias")
    assert before["var"] == "alias"
    final.obs["leiden"] = ["0", "1", "0", "1"]
    after = await agent.resolve_analysis_source_metadata(cluster_key="leiden", preferred_var_name="alias")
    assert before["partition_digest"] != after["partition_digest"]


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


@pytest.mark.asyncio
async def test_annotation_tool_threads_final_source_through_markers_review_and_apply(monkeypatch):
    from unittest.mock import AsyncMock
    from backend.agents.bioinformatics import executor_collaboration
    from backend.constants import LlmFailureAction
    import pandas as pd
    import scanpy as sc

    old, final = _partition_objects()
    final.uns["rank_genes_groups"] = {"params": {"groupby": "leiden"}}
    seen_marker_objects = []

    def markers(obj, **kwargs):
        seen_marker_objects.append(obj)
        return pd.DataFrame({"names": ["GeneA"], "pvals_adj": [0.01]})

    from backend.agents import bioinformatics_agent
    executor_collaboration.bind_facade(bioinformatics_agent)
    monkeypatch.setattr(sc.get, "rank_genes_groups_df", markers)
    monkeypatch.setattr(executor_collaboration, "_get_marker_db", lambda: SimpleNamespace(usable=lambda: False))
    agent = _agent(_ExecSandbox({"ad": old, "adata_merged": final}))
    state = SimpleNamespace(species="human", deg_thresholds={})
    agent._current_analysis_state = state
    agent._notify = AsyncMock()
    agent.marker_db_intervention_callback = AsyncMock(return_value=LlmFailureAction.SKIP)
    agent.annotate_clusters_callback = AsyncMock(return_value={"annotations": {
        "0": {"cell_type": "TypeA"}, "1": {"cell_type": "TypeB"},
    }})
    context = SimpleNamespace(
        agent=agent, state=state, messages=[], analysis_log=[], step_count=1,
        agent_reasoning="Annotate final partition", resolved_adata_source_context={},
        save_checkpoint=AsyncMock(), planning_stage="full",
    )
    call = SimpleNamespace(id="annotation-call", arguments={"cluster_key": "leiden", "tissue": "tissue"})
    await executor_collaboration.handle_annotate_clusters(context, call)
    source = agent.annotate_clusters_callback.await_args.kwargs["annotation_source"]
    assert source["adata_var"] == "adata_merged" and source["cluster_key"] == "leiden"
    assert all(obj is final for obj in seen_marker_objects) and len(seen_marker_objects) == 2
    assert final.obs["cell_type"].tolist() == ["TypeA", "TypeA", "TypeB", "TypeB"]
    assert "cell_type" not in old.obs
    assert context.resolved_adata_source_context["cluster_key"] == "leiden"


@pytest.mark.asyncio
async def test_branch_source_resolves_canonical_and_selection_error_cannot_fall_back():
    from backend.models.schema_parts.adata_checkpoints import AdataSourcePreparation, AdataSourceSelection
    _, final = _partition_objects()
    sandbox = _ExecSandbox({"adata": final, "scai_anchor": final})
    sandbox.host_managed_global_names = lambda: ["scai_anchor"]
    agent = _agent(sandbox)
    preparation = AdataSourcePreparation(
        owner_id="plan", preparation_id="prep_branch", binding_identity_status="exact",
        anchor_name="scai_anchor", selection=AdataSourceSelection(mode="checkpoint_branch", checkpoint_id="cp"),
    )
    agent._current_analysis_state = SimpleNamespace(
        current_plan=SimpleNamespace(adata_source_preparation_id="prep_branch"),
        adata_source_preparations=[preparation],
    )
    metadata = await agent.resolve_analysis_source_metadata({"mode": "checkpoint_branch"}, cluster_key="leiden")
    assert metadata["var"] == "adata" and metadata["preparation_id"] == "prep_branch"
    metadata = await agent.resolve_analysis_source_metadata({"selection_error": "stale preparation"})
    assert not metadata.get("var") and metadata["selection_error"] == "stale preparation"
    preparation.binding_identity_status = "missing"
    metadata = await agent.resolve_analysis_source_metadata({"mode": "checkpoint_branch"})
    assert not metadata.get("var") and "no longer exact" in metadata["selection_error"]
