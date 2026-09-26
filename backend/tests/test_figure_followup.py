"""PI-owned closure for unfinished publication-figure work."""

from types import SimpleNamespace

import pytest

from backend.models.schemas import (
    AnalysisPlan,
    AnalysisState,
    PlanStep,
)


def _add_eligible_source(state: AnalysisState, *, execution_id: str = "source"):
    """Install one durable source receipt linked to a completed plan step."""
    from backend.agents import bioinformatics_agent as facade
    from backend.agents.bioinformatics import executor_execute

    state.current_plan.steps = [PlanStep(
        id=2, title="Compute the declared contrast", method="approved method",
        status="completed",
    )]
    executor_execute.bind_facade(facade)
    purpose, step_ids, binding_status, binding_reason = (
        executor_execute.execution_plan_binding(state, {
            "execution_purpose": "planned",
            "plan_step_ids": [2],
        })
    )
    receipt = executor_execute.build_code_execution_receipt(
        state,
        "compute_contrast()",
        SimpleNamespace(output="contrast computed", error=None, figures=[]),
        0.1,
        purpose,
        step_ids,
        {"cohort_generation": 0, "cohort_digest": None},
        [],
        binding_status,
        binding_reason,
        [],
        executor_context=SimpleNamespace(
            executor_run_kind="ordinary",
            executor_run_scope_id="ordinary-run",
            executor_settlement_scope_id="ordinary-settlement",
            supplementary_request_id=None,
            resolved_adata_source_context={},
            adata_source=state.current_plan.adata_source,
        ),
        input_cohort={},
        execution_id=execution_id,
    )
    assert receipt.plan_binding_status == "candidate"
    state.code_history = [receipt]
    return receipt


@pytest.mark.parametrize("invalid_source", [
    "missing", "failed", "wrong_plan", "wrong_lineage", "supplementary",
    "wrong_step", "wrong_preparation",
])
def test_pending_step_redraw_preserves_receipt_identity_checks(invalid_source):
    from backend.services.figure_followup import build_figure_followup_scope

    state = AnalysisState(user_id="figure-identity", current_round=3)
    state.current_plan = AnalysisPlan(
        id="plan-1", phase="innovation", status="approved", steps=[],
    )
    receipt = _add_eligible_source(state)
    state.current_plan.steps[0].status = "pending"
    if invalid_source == "missing":
        state.code_history = []
    elif invalid_source == "failed":
        receipt.execution_status = "failed"
    elif invalid_source == "wrong_plan":
        receipt.plan_id = "other-plan"
    elif invalid_source == "wrong_lineage":
        receipt.lineage_id = "other-lineage"
    elif invalid_source == "supplementary":
        receipt.executor_run_kind = "supplementary"
    elif invalid_source == "wrong_step":
        receipt.plan_step_ids = [3]
    else:
        receipt.adata_source_preparation_id = "other-preparation"
    task = {
        "id": "task", "plan_id": "plan-1", "plan_step_id": 2,
        "round_number": 3, "lineage_id": "main",
        "pi_disposition": "targeted_redraw",
    }
    scope = build_figure_followup_scope(state, [task], run_scope_id="redraw")
    assert scope["tasks"][0]["eligible_execution_ids"] == []
    assert state.current_plan.steps[0].status == "pending"
