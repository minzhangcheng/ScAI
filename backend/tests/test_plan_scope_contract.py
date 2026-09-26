"""Technical Innovation scope and accepted-foundation identity contracts."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backend.models.schemas import (
    AcceptedFoundationInventory,
    AcceptedFoundationStep,
    AnalysisPhase,
    AnalysisPlan,
    AnalysisState,
    CodeExecution,
    FoundationReference,
    PlanAnalysisScope,
    PlanStep,
)
from backend.models.schema_parts.adata_checkpoints import (
    AdataCheckpointRecord,
    canonical_id_set_digest,
)
from backend.services.plan_scope import (
    PlanScopeValidationError,
    resolve_foundation_inventory,
    validate_innovation_plan_scope,
)


def _production_candidate_receipt(
    state: AnalysisState,
    *,
    execution_id: str,
    step_id: int,
) -> CodeExecution:
    from backend.agents import bioinformatics_agent as facade
    from backend.agents.bioinformatics import executor_execute

    executor_execute.bind_facade(facade)
    purpose, step_ids, binding_status, binding_reason = (
        executor_execute.execution_plan_binding(state, {
            "execution_purpose": "planned",
            "plan_step_ids": [step_id],
        })
    )
    receipt = executor_execute.build_code_execution_receipt(
        state,
        "print('completed plan step')",
        SimpleNamespace(output="completed", error=None, figures=[]),
        0.1,
        purpose,
        step_ids,
        {"cohort_generation": 1, "cohort_digest": "a" * 64},
        [],
        binding_status,
        binding_reason,
        [],
        executor_context=SimpleNamespace(
            executor_run_kind="ordinary",
            executor_run_scope_id="round1-run",
            executor_settlement_scope_id="round1-settlement",
            supplementary_request_id=None,
            resolved_adata_source_context={},
            adata_source=state.current_plan.adata_source,
        ),
        input_cohort={},
        execution_id=execution_id,
    )
    assert receipt.plan_binding_status == "candidate"
    return receipt


def _inventory(*, lineage_id: str = "main") -> AcceptedFoundationInventory:
    return AcceptedFoundationInventory(
        inventory_id="foundation-v1-r1",
        round_number=1,
        plan_id="round1-plan",
        planning_stage="stage1",
        lineage_id=lineage_id,
        adata_checkpoint_id="checkpoint-r1",
        steps=[AcceptedFoundationStep(
            plan_step_id=4,
            title="Accepted clustering baseline",
            method="scanpy.tl.leiden",
            evidence_execution_ids=["receipt-r1-step4"],
            figure_requirement_keys=["plan.4.umap"],
        )],
    )


def _state(*, lineage_id: str = "main") -> AnalysisState:
    state = AnalysisState(
        user_id="scope-test",
        current_round=2,
        analysis_phase=AnalysisPhase.INNOVATION,
        accepted_foundation_inventories=[_inventory(lineage_id=lineage_id)],
    )
    state.restore_branch_id = None if lineage_id == "main" else lineage_id
    return state


def _reference(**updates) -> FoundationReference:
    payload = {
        "inventory_id": "foundation-v1-r1",
        "round_number": 1,
        "plan_id": "round1-plan",
        "plan_step_ids": [4],
        "execution_ids": ["receipt-r1-step4"],
        "checkpoint_id": "checkpoint-r1",
    }
    payload.update(updates)
    return FoundationReference(**payload)


_DEFAULT_SCOPE = object()


def _plan(scope: PlanAnalysisScope | None = _DEFAULT_SCOPE, **updates) -> AnalysisPlan:
    if scope is _DEFAULT_SCOPE:
        scope = PlanAnalysisScope(
            relationship="reuse",
            cohort_scope="subset",
            foundation_refs=[_reference()],
            reason="Recluster the defined Cluster 18 subset using the accepted measurements.",
        )
    payload = {
        "phase": "innovation",
        "stage": "full",
        "scope_contract_version": 1,
        "foundation_inventory_id": "foundation-v1-r1",
        "steps": [PlanStep(
            id=1,
            title="Cluster 18 sensitivity",
            method="scanpy.tl.leiden",
            analysis_scope=scope,
        )],
    }
    payload.update(updates)
    return AnalysisPlan(**payload)


@pytest.mark.parametrize(
    "scope, message",
    [
        (None, "missing analysis_scope"),
        (
            PlanAnalysisScope(
                relationship="reuse",
                cohort_scope="subset",
                foundation_refs=[_reference(plan_step_ids=[99])],
                reason="Reuse the selected subset.",
            ),
            "unknown plan_step_ids",
        ),
        (
            PlanAnalysisScope(
                relationship="reuse",
                cohort_scope="subset",
                foundation_refs=[_reference(execution_ids=["missing-receipt"])],
                reason="Reuse the selected subset.",
            ),
            "execution_ids not retained",
        ),
        (
            PlanAnalysisScope(
                relationship="revise",
                cohort_scope="full",
                foundation_refs=[_reference()],
                reason="",
                requested_by=None,
            ),
            "without requested_by",
        ),
        (
            PlanAnalysisScope(
                relationship="revise",
                cohort_scope="full",
                foundation_refs=[],
                reason="Reevaluate a retained result outside the foundation inventory.",
            ),
            "without requested_by",
        ),
        (
            PlanAnalysisScope(
                relationship="revise",
                cohort_scope="full",
                foundation_refs=[],
                requested_by="pi",
                reason="   ",
            ),
            "without a reason",
        ),
        (
            PlanAnalysisScope(
                relationship="revise",
                cohort_scope="full",
                foundation_refs=[_reference(checkpoint_id="non-foundation-checkpoint")],
                requested_by="pi",
                reason="Reevaluate a prior result while claiming a foundation reference.",
            ),
            "foundation checkpoint_id does not match inventory",
        ),
    ],
)
def test_invalid_declarations_and_references_fail_as_technical_errors(scope, message):
    with pytest.raises(PlanScopeValidationError, match=message):
        validate_innovation_plan_scope(_state(), _plan(scope))


def test_legacy_approved_plan_can_resume_without_invented_scope_fields():
    state = _state()
    legacy = AnalysisPlan(
        phase="innovation",
        stage="full",
        status="approved",
        steps=[PlanStep(id=1, title="Legacy", method="scanpy")],
    )

    validate_innovation_plan_scope(state, legacy, allow_legacy_approved=True)
    assert legacy.scope_contract_version is None
    assert legacy.foundation_inventory_id is None


def test_reconstruction_uses_approved_plan_checkpoint_and_successful_receipts():
    state = _state()
    state.accepted_foundation_inventories = []
    plan = AnalysisPlan(
        id="round1-plan",
        phase="exploration",
        stage="stage1",
        round_number=1,
        steps=[PlanStep(
            id=4,
            title="Accepted clustering baseline",
            method="scanpy.tl.leiden",
            figure_requirements=[{"figure_type": "umap", "description": "Cluster view"}],
        )],
    )
    checkpoint = AdataCheckpointRecord(
        checkpoint_id="checkpoint-r1",
        status="available",
        boundary="stage1_complete",
        boundary_event_id="stage1-event-r1",
        round_number=1,
        planning_stage="stage1",
        session_lineage_id="main",
        adata_lineage_id="adata-lineage-r1",
        source_plan_id="round1-plan",
        blob_sha256="b" * 64,
        blob_relative_path=f"adata-checkpoints/blobs/{'b' * 64}.h5ad",
        size_bytes=1,
        n_obs=1,
        n_vars=1,
        source_preparation_ids_at_capture=["prep-r1"],
        source_preparation_id_count_at_capture=1,
        source_preparation_id_set_digest_at_capture=canonical_id_set_digest(
            ["prep-r1"]
        ),
    )
    state.adata_checkpoints = [checkpoint]
    state.plan_dialogue_history = [{
        "round": 1,
        "pi_decision": "approved",
        "plan": plan.model_dump(mode="json"),
    }]
    state.current_round = 1
    state.current_plan = plan
    state.code_history = [_production_candidate_receipt(
        state,
        execution_id="receipt-r1-step4",
        step_id=4,
    )]
    state.current_plan = None
    state.sandbox_state = {"cohort_identity_v1": {
        "cohort_digest": "a" * 64,
        "cohort_generation": 1,
        "lineage_id": "main",
    }}

    inventory = resolve_foundation_inventory(state)

    assert inventory is not None
    assert inventory.provenance == "reconstructed"
    assert inventory.adata_checkpoint_id == "checkpoint-r1"
    assert inventory.adata_source_preparation_ids == ["prep-r1"]
    assert inventory.cohort_digest == "a" * 64
    assert inventory.steps[0].plan_step_id == 4
    assert inventory.steps[0].evidence_execution_ids == ["receipt-r1-step4"]
    assert inventory.steps[0].figure_requirement_keys
