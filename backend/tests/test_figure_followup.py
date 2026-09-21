"""PI-owned closure for unfinished publication-figure work."""

import base64
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import (
    AnalysisPlan,
    AnalysisState,
    CodeExecution,
    FigureReviewStatus,
    PlanStep,
    PublicationFigure,
)
from backend.services.figure_contract import (
    FigureContractAudit,
    MissingFigure,
    RequiredFigure,
)
from backend.services.report.artifacts import is_report_eligible
from backend.tests.figure_image_helpers import valid_image_bytes
from backend.workflow.callbacks import CallbacksMixin
from backend.workflow.post_execution import PostExecutionReviewMixin


def _candidate(*, logical_id: str, requirement_key: str) -> PublicationFigure:
    payload = valid_image_bytes("JPEG")
    digest = hashlib.sha256(payload).hexdigest()
    return PublicationFigure(
        id="candidate-1",
        name="targeted_candidate",
        description="Best available targeted render",
        figure_type="volcano",
        png_path="/missing/targeted_candidate.jpg",
        pdf_path="/missing/targeted_candidate.pdf",
        png_base64=base64.b64encode(payload).decode(),
        render_sha256=digest,
        preview_sha256=digest,
        pdf_sha256="2" * 64,
        review_status=FigureReviewStatus.REVISION_REQUESTED,
        logical_figure_id=logical_id,
        plan_id="plan-1",
        plan_step_id=2,
        plan_requirement_key=requirement_key,
        round_number=3,
        unresolved_must_fix=["Separate two overlapping labels"],
        layout_violations=[{"kind": "text_over_text", "severity": "block"}],
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


@pytest.mark.asyncio
async def test_targeted_missing_figure_falls_back_to_explicit_pi_selection():
    state = AnalysisState(
        id="figure-followup", user_id="user", current_round=3,
        initial_instruction="Analyze the declared contrast.",
    )
    state.current_plan = AnalysisPlan(
        id="plan-1", phase="innovation", status="approved", steps=[],
    )
    _add_eligible_source(state)
    requirement = RequiredFigure(
        key="plan.2.1",
        figure_type="volcano",
        description="Effect size versus adjusted significance",
        plan_step_id=2,
        plan_id="plan-1",
        require_plan_link=True,
    )
    contract = FigureContractAudit(
        requirements=[requirement],
        missing=[MissingFigure(requirement=requirement)],
    )

    pi = SimpleNamespace()
    pi.decide_figure_followups = AsyncMock(return_value=[{
        "task_id": "placeholder",
        "disposition": "targeted_redraw",
        "instruction": "Render the linked contrast as a volcano plot.",
        "rationale": "The result exists and the missing deliverable is useful.",
        "report_note": "Use the targeted contrast only.",
    }])

    async def decide_with_real_task(_state, tasks, _candidate_counts):
        decision = dict(pi.decide_figure_followups.return_value[0])
        decision["task_id"] = tasks[0].id
        return [decision]

    pi.decide_figure_followups.side_effect = decide_with_real_task
    pi.select_fallback_figure = AsyncMock(side_effect=lambda _state, _task, candidates: {
        "selected_figure_id": candidates[0].id,
        "rationale": "This candidate communicates the contrast most clearly.",
        "report_note": "Two labels still overlap; interpret labelled points cautiously.",
    })

    bioinformatics = SimpleNamespace()

    async def run_targeted(current_state, instruction, **kwargs):
        task = current_state.figure_revision_tasks[0]
        current_state.publication_figures.append(_candidate(
            logical_id=task.logical_figure_id,
            requirement_key=task.plan_requirement_key,
        ))
        return {"status": "completed", "analysis_log": [], "code_history": []}

    bioinformatics.run_autonomous_analysis = AsyncMock(side_effect=run_targeted)
    host = PostExecutionReviewMixin.__new__(PostExecutionReviewMixin)
    host.session_id = state.id
    host.pi_agent = pi
    host.bioinformatics_agent = bioinformatics
    host._figure_review_callback = AsyncMock()
    host._supervisor_callback = AsyncMock()
    host.state_callback = AsyncMock(return_value=True)

    result = await host._run_figure_followup(state, contract, "full")

    task = state.figure_revision_tasks[0]
    selected = state.publication_figures[0]
    assert result["status"] == "completed"
    assert result["targeted_task_ids"] == [task.id]
    assert task.status == "resolved"
    assert task.pi_disposition == "accept_best"
    assert task.selected_figure_id == selected.id
    assert selected.pi_degraded_acceptance is True
    assert "Two labels still overlap" in (selected.pi_degraded_acceptance_reason or "")
    assert selected.unresolved_must_fix == ["Separate two overlapping labels"]
    assert is_report_eligible(selected) is True
    call = bioinformatics.run_autonomous_analysis.await_args
    assert "Render the linked contrast as a volcano plot" in call.args[1]
    import json
    from backend.services.figure_followup import get_figure_followup_scope
    scope = get_figure_followup_scope(state, call.kwargs["run_scope_id"])
    entries = json.loads(call.args[1].split("rather than inventing an ID.\n", 1)[1].split("\n\n", 1)[0])
    assert entries[0]["figure_task_id"] == task.id
    assert entries[0]["eligible_execution_ids"] == scope["tasks"][0]["eligible_execution_ids"] == ["source"]
    assert entries[0]["source_receipts"][0]["plan_step_ids"] == [2]
    assert "Never extract a UUID from logical_figure_id" in call.args[1]
    assert call.kwargs["resume_from_checkpoint"] is True
    assert call.kwargs["run_kind"] == "figure_followup"


@pytest.mark.asyncio
async def test_pi_consultation_is_durable_for_later_figure_review():
    state = AnalysisState(
        id="pi-decision", user_id="user", current_round=4,
        initial_instruction="Analyze the selected direction.",
    )
    state.current_plan = AnalysisPlan(
        id="plan-current", phase="innovation", status="approved", steps=[],
    )
    host = CallbacksMixin.__new__(CallbacksMixin)
    host.session_id = state.id
    host._current_analysis_state = state
    host._current_graph_state = {"planning_stage": "stage2"}
    host._send_message = AsyncMock()
    host._settings_provider = SimpleNamespace(get_temperature=lambda _agent: 0.1)
    host.pi_agent = SimpleNamespace(chat_without_history=AsyncMock(
        return_value="Use the responder-versus-control contrast in the figure."
    ))

    response = await host._pi_callback(
        "Which contrast should be plotted?", "Both contrasts are available."
    )

    assert response.startswith("Use the responder-versus-control")
    assert len(state.pi_decision_records) == 1
    record = state.pi_decision_records[0]
    assert record.plan_id == "plan-current"
    assert record.round_number == 4
    assert record.planning_stage == "stage2"
    restored = AnalysisState.model_validate(state.model_dump(mode="json"))
    assert restored.pi_decision_records[0].response == record.response


@pytest.mark.asyncio
async def test_pi_consultation_keeps_plan_prefix_stable_as_progress_changes():
    state = AnalysisState(id="pi-cache", user_id="user")
    state.current_plan = AnalysisPlan(
        id="plan-current", phase="innovation", status="approved",
        steps=[PlanStep(
            id=1, title="Summarize results", method="Summarize measured effects",
            status="pending",
        )],
    )
    host = CallbacksMixin.__new__(CallbacksMixin)
    host.session_id = state.id
    host._current_analysis_state = state
    host._current_graph_state = {"planning_stage": "stage2"}
    host._send_message = AsyncMock()
    host._settings_provider = SimpleNamespace(get_temperature=lambda _agent: 0.1)
    host.pi_agent = SimpleNamespace(chat_without_history=AsyncMock(return_value="Proceed."))

    await host._pi_callback("First question?", "First evidence")
    state.current_plan.steps[0].status = "completed"
    await host._pi_callback("Second question?", "Updated evidence")
    first, second = [call.args[0] for call in host.pi_agent.chat_without_history.await_args_list]
    prefix, first_tail = first.split("## Current Plan Progress", 1)
    second_prefix, second_tail = second.split("## Current Plan Progress", 1)

    assert prefix == second_prefix
    assert "Please provide clear guidance" in prefix
    assert "CURRENT STAGE: Stage 2" in prefix
    assert "Step 1: Summarize results" in prefix
    assert "For MAJOR: State clearly that a plan revision is needed" in prefix
    assert "[pending] Step 1" in first_tail
    assert "[completed] Step 1" in second_tail
    assert "First evidence" in first_tail and "QUESTION: First question?" in first_tail
    assert "Updated evidence" in second_tail and "QUESTION: Second question?" in second_tail
    assert "First evidence" not in second and "First question?" not in second
    assert [r.question for r in state.pi_decision_records] == ["First question?", "Second question?"]
    first_call, second_call = host.pi_agent.chat_without_history.await_args_list
    cache_prefix = first_call.kwargs["cache_prefix"]
    assert cache_prefix == second_call.kwargs["cache_prefix"]
    assert first.startswith(cache_prefix) and second.startswith(cache_prefix)
    assert "Step 1: Summarize results" in cache_prefix
    assert "Current Plan Progress" not in cache_prefix
    assert "First evidence" not in cache_prefix and "First question?" not in cache_prefix
