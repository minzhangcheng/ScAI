"""B2: scoped executor transcript identity and legacy compatibility."""
from __future__ import annotations

from backend.agents.bioinformatics.executor_checkpoint import (
    ExecutorCheckpointIdentity,
    executor_checkpoint_prefix,
    is_executor_checkpoint_name,
    validate_checkpoint,
)


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


def test_scoped_checkpoint_filename_contract_and_classifier():
    prefix = executor_checkpoint_prefix(
        run_kind="supplementary", scope_digest="d" * 24,
    )
    name = f"sess-1_{prefix}_checkpoint.json"
    assert name == "sess-1_bioinfo_supplementary_" + "d" * 24 + "_checkpoint.json"
    assert is_executor_checkpoint_name(name)
    assert is_executor_checkpoint_name("sess-1_bioinfo_checkpoint.json")
    assert is_executor_checkpoint_name(
        "sess_bioinfo_owner_bioinfo_ordinary_0123456789abcdef01234567_checkpoint.json"
    )
    assert is_executor_checkpoint_name(name, session_id="sess-1")
    assert not is_executor_checkpoint_name(name, session_id="sess-2")
    assert not is_executor_checkpoint_name(
        "sess-1_bioinfo_arbitrary_checkpoint.json"
    )
    assert not is_executor_checkpoint_name("sess-1_report_checkpoint.json")


def test_identity_mismatch_is_not_resumable():
    errors = validate_checkpoint(
        _payload(),
        current_round=2,
        planning_stage="full",
        expected_identity={"_scope_digest": "e" * 24, "run_kind": "ordinary"},
    )
    assert any("_scope_digest mismatch" in item for item in errors)


def test_matching_identity_validates_cleanly():
    errors = validate_checkpoint(
        _payload(),
        current_round=2,
        planning_stage="full",
        expected_identity={"_scope_digest": "a" * 24, "run_kind": "ordinary"},
    )
    assert errors == []


def test_direct_same_source_revision_keeps_filename_and_can_resume_parent_plan():
    common = dict(
        run_kind="ordinary",
        round_number=2,
        planning_stage="full",
        current_lineage_id="main",
        origin_lineage_id="main",
        reconcile_from_lineage_id="",
        revision_of_plan_id="",
        run_scope_id="run-1",
        settlement_scope_id="settle-1",
        source_scope_id="prep-scope",
        preparation_id="prep-1",
        selection_signature="b" * 64,
        application_token_digest="c" * 64,
        supplementary_request_id="",
    )
    parent = ExecutorCheckpointIdentity(active_plan_id="plan-1", **common)
    revision = ExecutorCheckpointIdentity(
        active_plan_id="plan-2",
        **{**common, "revision_of_plan_id": "plan-1"},
    )
    assert parent.scope_digest == revision.scope_digest
    payload = _payload(**parent.checkpoint_fields())
    assert validate_checkpoint(
        payload,
        current_round=2,
        planning_stage="full",
        expected_identity=revision.checkpoint_fields(),
    ) == []
