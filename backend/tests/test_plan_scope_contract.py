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
    clone_foundation_inventories_for_branch,
    render_innovation_scope_context,
    resolve_foundation_inventory,
    validate_innovation_plan_scope,
)
from backend.services.figure_followup import create_needs_new_analysis_request


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


def test_valid_subset_reuse_may_overlap_a_foundation_method_family():
    state = _state()
    plan = _plan()

    validate_innovation_plan_scope(state, plan)


def test_reuse_of_other_prior_results_does_not_invent_a_foundation_link():
    plan = _plan(PlanAnalysisScope(
        relationship="reuse",
        cohort_scope="full",
        foundation_refs=[],
        reason="Reuse the stored result from prior-plan-other, outside the foundation inventory.",
    ))
    plan.steps[0].input = "Stored result table from prior-plan-other; execution provenance unavailable."
    before = plan.model_dump(mode="json")

    validate_innovation_plan_scope(_state(), plan)
    restored = AnalysisPlan.model_validate(before)
    validate_innovation_plan_scope(_state(), restored)

    assert restored.model_dump(mode="json") == before
    assert plan.model_dump(mode="json") == before
    assert plan.steps[0].analysis_scope.relationship == "reuse"


def test_scope_validation_reports_independent_errors_without_rewriting_references():
    plan = _plan()
    plan.steps[0].analysis_scope.foundation_refs[0].execution_ids = ["spliced-receipt"]
    plan.steps.append(PlanStep(
        id=8, title="Other reuse", method="Read retained measurements",
        analysis_scope=PlanAnalysisScope(
            relationship="reuse", cohort_scope="subset",
            foundation_refs=[_reference(plan_step_ids=[99], checkpoint_id="wrong-checkpoint")],
        ),
    ))
    before = plan.model_dump(mode="json")

    with pytest.raises(PlanScopeValidationError) as error:
        validate_innovation_plan_scope(_state(), plan)

    message = str(error.value)
    assert "Innovation step 1 analysis_scope.foundation_refs[0]" in message
    assert "spliced-receipt" in message
    assert "receipt-r1-step4" in message
    assert "Innovation step 8 analysis_scope.foundation_refs[0]" in message
    assert "unknown plan_step_ids: [99]" in message
    assert "expected 'checkpoint-r1'" in message
    assert plan.model_dump(mode="json") == before


@pytest.mark.parametrize("step_count", [11, 128])
def test_large_inventory_context_preserves_parseable_identity_and_last_step(step_count):
    state = _state()
    inventory = state.accepted_foundation_inventories[0]
    inventory.steps = [AcceptedFoundationStep(
        plan_step_id=index,
        title="Accepted baseline " + "x" * 220,
        method="m" * 600,
        evidence_execution_ids=[f"receipt-{index}-{receipt}" for receipt in range(12)],
        figure_requirement_keys=[f"plan.{index}.figure"],
    ) for index in range(1, step_count + 1)]
    before = inventory.model_dump(mode="json")

    context = render_innovation_scope_context(state)
    inventory_text = context.split(
        "Accepted foundation inventory (Host-observed technical identity only):\n", 1
    )[1].split("\n\nEach analysis_scope", 1)[0]
    payload = json.loads(inventory_text)

    assert len(inventory_text) <= 9_000
    assert payload["inventory_id"] == inventory.inventory_id
    assert payload["plan_id"] == inventory.plan_id
    ids = payload.get("plan_step_ids") or [step["plan_step_id"] for step in payload["steps"]]
    assert ids == list(range(1, step_count + 1))
    assert inventory.model_dump(mode="json") == before
    if "steps" in payload:
        assert payload["steps"][-1]["evidence_execution_ids"] == inventory.steps[-1].evidence_execution_ids
    else:
        assert payload["execution_ids_omitted"] is True


def test_valid_foundation_revision_requires_and_accepts_authority_and_reason():
    state = _state()
    plan = _plan(PlanAnalysisScope(
        relationship="revise",
        cohort_scope="full",
        foundation_refs=[_reference()],
        reason="The PI requested a full-cohort reevaluation after reviewing the accepted baseline.",
        requested_by="pi",
    ))

    validate_innovation_plan_scope(state, plan)


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
    ],
)
def test_invalid_declarations_and_references_fail_as_technical_errors(scope, message):
    with pytest.raises(PlanScopeValidationError, match=message):
        validate_innovation_plan_scope(_state(), _plan(scope))


def test_cross_lineage_inventory_is_not_resolved_for_new_innovation_plan():
    state = _state()
    state.restore_branch_id = "branch-child"
    plan = _plan()

    with pytest.raises(PlanScopeValidationError, match="identity is unavailable"):
        validate_innovation_plan_scope(state, plan)


def test_legacy_plan_loads_but_cannot_be_approved_as_a_new_innovation_plan():
    state = _state()
    legacy = AnalysisPlan(
        phase="innovation",
        stage="full",
        steps=[PlanStep(id=1, title="Legacy", method="scanpy")],
    )

    with pytest.raises(PlanScopeValidationError, match="scope_contract_version=1"):
        validate_innovation_plan_scope(state, legacy)


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


def test_reconstruction_uses_stage1_boundary_when_only_round2_plan_is_retained():
    state = AnalysisState(
        user_id="legacy-staged-scope",
        current_round=1,
        analysis_phase=AnalysisPhase.EXPLORATION,
    )
    round1_plan = AnalysisPlan(
        id="round1-stage1-plan",
        phase="exploration",
        stage="stage1",
        round_number=1,
        status="approved",
        steps=[PlanStep(id=4, title="Accepted baseline", method="approved method")],
    )
    state.current_plan = round1_plan
    receipt = _production_candidate_receipt(
        state,
        execution_id="legacy-stage1-candidate",
        step_id=4,
    )
    checkpoint = AdataCheckpointRecord(
        checkpoint_id="legacy-stage1-checkpoint",
        status="available",
        boundary="stage1_complete",
        boundary_event_id="legacy-stage1-event",
        round_number=1,
        planning_stage="stage1",
        session_lineage_id="main",
        adata_lineage_id="legacy-stage1-adata-lineage",
        source_plan_id=round1_plan.id,
        blob_sha256="c" * 64,
        blob_relative_path=f"adata-checkpoints/blobs/{'c' * 64}.h5ad",
        size_bytes=1,
        n_obs=1,
        n_vars=1,
        attempt_execution_ids_at_capture=[receipt.id],
        attempt_execution_id_count_at_capture=1,
        attempt_execution_id_set_digest_at_capture=canonical_id_set_digest(
            [receipt.id]
        ),
    )
    state.code_history = [receipt]
    state.adata_checkpoints = [checkpoint]
    state.current_round = 2
    state.analysis_phase = AnalysisPhase.INNOVATION
    state.current_plan = AnalysisPlan(
        id="round2-plan",
        phase="innovation",
        round_number=2,
        status="approved",
        steps=[PlanStep(id=1, title="New direction", method="new method")],
    )
    state.plan_dialogue_history = [{
        "round": 2,
        "pi_decision": "approved",
        "plan": state.current_plan.model_dump(mode="json"),
    }]

    inventory = resolve_foundation_inventory(state)

    assert inventory is not None
    assert inventory.provenance == "reconstructed"
    assert inventory.plan_id == round1_plan.id
    assert inventory.planning_stage == "stage1"
    assert inventory.steps[0].plan_step_id == 4
    assert inventory.steps[0].title == ""
    assert inventory.steps[0].method == ""
    assert inventory.steps[0].evidence_execution_ids == [receipt.id]


def test_restored_branch_gets_child_local_inventory_with_same_identity():
    state = AnalysisState(
        user_id="scope-test",
        accepted_foundation_inventories=[_inventory()],
    )

    assert clone_foundation_inventories_for_branch(
        state,
        source_lineage_id="main",
        branch_id="branch-child",
    ) == 1
    child = next(
        inventory
        for inventory in state.accepted_foundation_inventories
        if inventory.lineage_id == "branch-child"
    )
    assert child.inventory_id == "foundation-v1-r1"
    assert child.provenance == "inherited"
    assert child.inherited_from_lineage_id == "main"


def test_innovation_prompt_projects_inventory_and_scope_rules():
    state = _state()

    text = render_innovation_scope_context(state)

    assert "Innovation Scope Contract v1" in text
    assert "foundation-v1-r1" in text
    assert "Targeted subset work is legal" in text
    assert "Method overlap alone is not a Host rejection criterion" in " ".join(
        text.split()
    )


def test_pending_figure_analysis_request_enters_later_innovation_context():
    state = _state()
    request = create_needs_new_analysis_request(
        state,
        {
            "id": "figure-task-1",
            "plan_step_id": 4,
            "plan_requirement_key": "plan.4.umap",
            "logical_figure_id": "figure-r2-1",
            "pi_instruction": "Recompute the embedding with another parameter.",
            "pi_rationale": "The requested visual needs a new measurement.",
        },
    )

    context = render_innovation_scope_context(state)

    assert request["status"] == "pending_review"
    assert request["id"] in context
    assert "separately approved analysis" in context
    assert request["request"] in context
