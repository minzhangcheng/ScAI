"""Path-free, bounded catalog projection used by both agents."""
from __future__ import annotations

from backend.models.schemas import (
    AdataCheckpointRecord,
    AdataLiveBindingRecord,
    AdataSourceSelection,
    AnalysisPlan,
    AnalysisState,
)
from backend.services.adata_checkpoints import build_checkpoint_projection


def _state() -> AnalysisState:
    digest = "a" * 64
    state = AnalysisState(user_id="u")
    state.adata_checkpoints = [
        AdataCheckpointRecord(
            checkpoint_id="cp-stage1",
            status="available",
            boundary="stage1_complete",
            boundary_event_id="ev-stage1",
            round_number=1,
            planning_stage="stage1",
            session_lineage_id="main",
            adata_lineage_id="lin-1",
            blob_sha256=digest,
            blob_relative_path=f"adata-checkpoints/blobs/{digest}.h5ad",
            size_bytes=1,
            n_obs=10,
            n_vars=20,
            research_label="Round 1 Stage 1 accepted atlas",
        ),
        AdataCheckpointRecord(
            checkpoint_id="cp-round2",
            status="available",
            boundary="round_complete",
            boundary_event_id="ev-round2",
            round_number=2,
            planning_stage="full",
            boundary_revision=1,
            session_lineage_id="main",
            adata_lineage_id="lin-1",
            parent_checkpoint_id="cp-stage1",
            blob_sha256=digest,
            blob_relative_path=f"adata-checkpoints/blobs/{digest}.h5ad",
            size_bytes=1,
            n_obs=11,
            n_vars=20,
            research_label="Round 2 result",
        ),
    ]
    state.active_adata_checkpoint_id = "cp-round2"
    state.adata_live_bindings = [
        AdataLiveBindingRecord(
            binding_id="lb-1",
            name="scai_work_7",
            anchor_name="scai_adata_anchor_7",
            mode_created="checkpoint_copy",
            adata_lineage_id="lin-work",
            preparation_id="prep-1",
            session_lineage_id="main",
        )
    ]
    return state


def test_projection_is_bounded_path_free_and_filterable():
    state = _state()
    projection = build_checkpoint_projection(state, page=1, page_size=10)
    assert projection["entries"]
    assert projection["entries"][0]["entry_kind"] == "current"
    assert any(
        entry.get("checkpoint_id") == "cp-stage1"
        for entry in projection["entries"]
    )
    assert any(
        entry.get("live_binding_id") == "lb-1"
        for entry in projection["entries"]
    )
    assert "blob_relative_path" not in str(projection)

    exact = build_checkpoint_projection(state, checkpoint_id="cp-stage1")
    assert [entry["checkpoint_id"] for entry in exact["entries"]] == ["cp-stage1"]
    binding = build_checkpoint_projection(state, live_binding_id="lb-1")
    assert [entry["live_binding_id"] for entry in binding["entries"]] == ["lb-1"]


def test_supplementary_revision_is_newest_round_head():
    state = _state()
    base = state.adata_checkpoints[1]
    supplement = base.model_copy(update={
        "checkpoint_id": "cp-supp",
        "boundary": "supplementary_complete",
        "boundary_event_id": "ev-supp",
        "boundary_revision": 2,
    })
    state.adata_checkpoints.append(supplement)
    state.active_adata_checkpoint_id = "cp-supp"

    projection = build_checkpoint_projection(state, page_size=20)
    round_heads = [
        entry for entry in projection["entries"]
        if entry.get("entry_kind") == "checkpoint"
        and entry.get("round") == 2
    ]
    assert round_heads and round_heads[0]["checkpoint_id"] == "cp-supp"


def test_stale_binding_not_advertised_pending_is_labeled():
    from backend.models.schemas import AdataLiveBindingRecord
    state = _state()
    state.adata_live_bindings.append(
        AdataLiveBindingRecord(
            binding_id="lb-pending",
            name="scai_work_8",
            anchor_name="scai_adata_anchor_8",
            mode_created="checkpoint_copy",
            adata_lineage_id="lin-pending",
            preparation_id="prep-pending",
            session_lineage_id="main",
            status="pending_reconcile",
        )
    )
    state.adata_live_bindings.append(
        AdataLiveBindingRecord(
            binding_id="lb-stale",
            name="scai_work_9",
            anchor_name="scai_adata_anchor_9",
            mode_created="checkpoint_copy",
            adata_lineage_id="lin-stale",
            preparation_id="prep-stale",
            session_lineage_id="main",
            status="stale",
        )
    )
    projection = build_checkpoint_projection(state, page_size=20)
    bindings = {
        entry.get("live_binding_id"): entry
        for entry in projection["entries"]
        if entry.get("entry_kind") == "live_binding"
    }
    assert bindings["lb-pending"]["availability"] == "requires_runtime_verification"
    assert "lb-stale" not in bindings


def test_explicit_old_checkpoint_is_kept_ahead_of_round_heads():
    state = _state()
    digest = "b" * 64
    state.adata_checkpoints.append(AdataCheckpointRecord(
        checkpoint_id="cp-old-selected",
        status="available",
        boundary="round_complete",
        boundary_event_id="ev-old-selected",
        round_number=1,
        planning_stage="full",
        session_lineage_id="main",
        adata_lineage_id="lin-old",
        blob_sha256=digest,
        blob_relative_path=f"adata-checkpoints/blobs/{digest}.h5ad",
        size_bytes=1,
        n_obs=9,
        n_vars=20,
        research_label="Explicit older branch",
    ))
    state.current_plan = AnalysisPlan(
        phase="exploration",
        overview="Use an earlier checkpoint",
        adata_source=AdataSourceSelection(
            mode="checkpoint_branch",
            checkpoint_id="cp-old-selected",
        ),
    )

    projection = build_checkpoint_projection(state, page_size=3)
    selected = [
        entry for entry in projection["entries"]
        if entry.get("checkpoint_id") == "cp-old-selected"
    ]
    assert selected and selected[0]["selected_by_current_plan"] is True
