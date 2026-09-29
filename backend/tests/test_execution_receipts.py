"""Focused contracts for host-validated execution receipts."""

from backend.agents.bioinformatics.executor_execute import execution_plan_binding
from backend.agents.bioinformatics.executor_prompt import _executor_prompt_methods
from backend.agents.bioinformatics.shared import current_execution_identity_error
from backend.models.schemas import AnalysisPlan, AnalysisState, CodeExecution, PlanStep


def _state_with_plan() -> AnalysisState:
    return AnalysisState(
        id="receipt-session",
        user_id="user",
        current_plan=AnalysisPlan(
            id="plan-current",
            phase="exploration",
            status="approved",
            steps=[
                PlanStep(id=1, title="Load", method="Load input"),
                PlanStep(id=2, title="QC", method="Assess quality"),
            ],
        ),
    )


def test_planned_execution_records_only_current_step_candidates():
    state = _state_with_plan()

    assert execution_plan_binding(
        state,
        {"execution_purpose": "planned", "plan_step_ids": [1, 2]},
    ) == ("planned", [1, 2], "candidate", None)

    purpose, step_ids, status, warning = execution_plan_binding(
        state,
        {"execution_purpose": "planned", "plan_step_ids": [3]},
    )
    assert purpose == "planned"
    assert step_ids == []
    assert status == "unverified"
    assert "unknown plan_step_ids" in warning


def test_supplementary_execution_is_explicitly_non_plan_evidence():
    state = _state_with_plan()

    assert execution_plan_binding(
        state,
        {"execution_purpose": "supplementary", "plan_step_ids": []},
    ) == ("supplementary", [], "supplementary", None)


def test_executor_prompt_forbids_figure_label_plan_step_inference():
    prompt = _executor_prompt_methods()

    assert "Never derive a plan_step_id from a figure/task label" in prompt
    assert "successful unrelated cell" in prompt


def test_current_derived_record_rejects_foreign_plan_and_source_preparation():
    state = _state_with_plan()
    state.current_round = 2
    state.sandbox_state = {
        "analysis_generation": 4,
        "cohort_identity_v1": {
            "cohort_generation": 2,
            "cohort_digest": "a" * 64,
        },
    }
    receipt = CodeExecution(
        id="current-execution",
        code="compute_result()",
        round_number=2,
        evidence_schema_version=2,
        plan_id=state.current_plan.id,
        plan_step_ids=[1],
        execution_purpose="planned",
        plan_binding_status="candidate",
        execution_status="succeeded",
        analysis_generation=4,
        cohort_generation=2,
        cohort_digest="a" * 64,
    )

    assert current_execution_identity_error(state, receipt) is None

    stale_plan = receipt.model_copy(update={"plan_id": "plan-old"})
    assert "another approved plan" in current_execution_identity_error(
        state, stale_plan
    )

    # Generation/cohort values are provenance observations, not active-value
    # equality gates; a prior generation remains a valid current-plan receipt.
    stale_generation = receipt.model_copy(update={"analysis_generation": 3})
    assert current_execution_identity_error(state, stale_generation) is None

    state.current_plan.adata_source_preparation_id = "prep-current"
    other_source = receipt.model_copy(
        update={"adata_source_preparation_id": "prep-other"}
    )
    assert "another AnnData source preparation" in current_execution_identity_error(
        state, other_source
    )
