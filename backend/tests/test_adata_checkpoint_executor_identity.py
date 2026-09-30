"""B2: scoped executor transcript identity and legacy compatibility."""
from __future__ import annotations

from backend.agents.bioinformatics.executor_checkpoint import (
    validate_checkpoint,
)
from backend.agents.bioinformatics.executor_execute import (
    _apply_registry_reconciliation_to_state,
)
from backend.models.schemas import AnalysisState, AdataLiveBindingRecord


def _payload(**overrides):
    payload = {
        "_scope_digest": "a" * 24,
        "run_kind": "ordinary",
        "run_scope_id": "run-1",
        "settlement_scope_id": "settle-1",
        "source_scope_id": "prep_scope_1",
        "preparation_id": "prep-1",
        "selection_signature": "b" * 64,
        "application_token_digest": "c" * 64,
        "supplementary_request_id": None,
        "figure_followup": False,
        "round_number": 2,
        "planning_stage": "full",
        "step_number": 1,
        "messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        "findings": [],
        "figures": [],
    }
    payload.update(overrides)
    return payload


def test_identity_mismatch_is_not_resumable():
    errors = validate_checkpoint(
        _payload(),
        current_round=2,
        planning_stage="full",
        expected_identity={"_scope_digest": "e" * 24, "run_kind": "ordinary"},
    )
    assert any("_scope_digest mismatch" in item for item in errors)


def test_stale_binding_reconciliation_preserves_loadable_session_identity():
    binding = AdataLiveBindingRecord(
        name="scai_work_1",
        anchor_name="scai_adata_anchor_1",
        mode_created="checkpoint_copy",
        adata_lineage_id="lineage-1",
        preparation_id="prep-1",
    )
    state = AnalysisState(user_id="owner", adata_live_bindings=[binding])

    assert _apply_registry_reconciliation_to_state(
        state, {"ok": True, "binding_statuses": {binding.binding_id: "stale"}},
    ) is True

    loaded = AnalysisState.model_validate_json(state.model_dump_json())
    assert loaded.adata_live_bindings[0].status == "stale"
    assert loaded.adata_live_bindings[0].anchor_name == "scai_adata_anchor_1"
