"""Accepted HITL actions publish a post-wait state without workflow delay."""

from unittest.mock import MagicMock

import pytest

import backend.routes._helpers as route_helpers
import backend.routes.interaction as interaction
import backend.routes.lifecycle as lifecycle
from backend.constants import WaitInputReason
from backend.models.api_requests import (
    AnnotationConfirmation,
    HumanInputRequest,
    LlmFailureActionRequest,
    PlanReviewRequest,
    SpeciesInputRequest,
)
from backend.models.schemas import AnalysisPhase, AnalysisState, TaskStatus


def _request():
    request = MagicMock()
    request.state.user_id = "user-1"
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "change_type", "route_call", "signal_name"),
    [
        (
            TaskStatus.WAITING_PLAN_REVIEW,
            "plan_review_received",
            lambda request: interaction.submit_plan_review(
                "session-1", request, PlanReviewRequest(action="approve"),
            ),
            "signal_plan_review",
        ),
        (
            TaskStatus.WAITING_SPECIES_INPUT,
            "species_input_received",
            lambda request: interaction.species_input(
                "session-1", request, SpeciesInputRequest(input="Mus musculus"),
            ),
            "signal_species_input",
        ),
    ],
)
async def test_active_review_gates_publish_running(
    monkeypatch, status, change_type, route_call, signal_name,
):
    state = AnalysisState(user_id="user-1", status=status)
    workflow = MagicMock()
    getattr(workflow, signal_name).return_value = True
    sent = []

    async def fake_owner(session_id, user_id):
        return state

    async def fake_send(session_id, message):
        sent.append(message)
        return True

    monkeypatch.setattr(interaction, "require_session_ownership", fake_owner)
    monkeypatch.setattr(interaction, "workflows", {"session-1": workflow})
    monkeypatch.setattr(route_helpers, "send_analysis_json", fake_send)

    response = await route_call(_request())

    assert response.success is True
    assert state.status == TaskStatus.RUNNING
    assert state.waiting_input_started_at is None
    assert state.state_version == 1
    assert sent[-1]["content"]["change_type"] == change_type
    assert sent[-1]["content"]["status"] == "running"
    getattr(workflow, signal_name).assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "phase", "change_type", "route_call", "signal_name"),
    [
        (
            TaskStatus.WAITING_ANNOTATION,
            AnalysisPhase.EXPLORATION,
            "annotation_confirmation_received",
            lambda request: interaction.confirm_annotations(
                "session-1", request, AnnotationConfirmation(),
            ),
            "signal_annotation_confirmed",
        ),
        (
            TaskStatus.WAITING_INPUT,
            AnalysisPhase.INTEGRATION,
            "integration_finalize_received",
            lambda request: interaction.finalize_from_integration(
                "session-1", request,
            ),
            "signal_continue",
        ),
    ],
)
async def test_persisted_gates_sync_running_before_signalling(
    monkeypatch, status, phase, change_type, route_call, signal_name,
):
    state = AnalysisState(user_id="user-1", status=status, analysis_phase=phase)
    workflow = MagicMock()
    events = []

    async def fake_owner(session_id, user_id):
        return state

    async def fake_update(session_id, updated):
        events.append("cache")

    async def fake_save(session_id, updated):
        events.append("save")

    async def fake_send(session_id, message):
        events.append(("sync", message))
        return True

    monkeypatch.setattr(interaction, "require_session_ownership", fake_owner)
    monkeypatch.setattr(interaction, "_update_session_cache", fake_update)
    monkeypatch.setattr(interaction, "async_save_session", fake_save)
    monkeypatch.setattr(interaction, "workflows", {"session-1": workflow})
    monkeypatch.setattr(route_helpers, "send_analysis_json", fake_send)

    response = await route_call(_request())

    assert response.success is True
    assert state.status == TaskStatus.RUNNING
    assert events[0:2] == ["cache", "save"]
    assert events[2][1]["content"]["change_type"] == change_type
    getattr(workflow, signal_name).assert_called_once_with()


@pytest.mark.asyncio
async def test_round_feedback_publishes_running_before_workflow_progress(monkeypatch):
    state = AnalysisState(user_id="user-1", status=TaskStatus.WAITING_INPUT)
    workflow = MagicMock()
    sent = []

    async def fake_owner(session_id, user_id):
        return state

    async def fake_send(session_id, message):
        sent.append(message)
        return True

    monkeypatch.setattr(interaction, "require_session_ownership", fake_owner)
    monkeypatch.setattr(interaction, "workflows", {"session-1": workflow})
    monkeypatch.setattr(interaction, "send_analysis_json", fake_send)
    monkeypatch.setattr(route_helpers, "send_analysis_json", fake_send)

    response = await interaction.send_human_input(
        "session-1", _request(), HumanInputRequest(message="Continue"),
    )

    assert response.success is True
    assert response.data["continuation_started"] is True
    assert state.status == TaskStatus.RUNNING
    sync = next(message for message in sent if message["type"] == "state_sync")
    assert sync["content"]["change_type"] == "human_input_received"
    assert sync["content"]["status"] == "running"
    workflow.signal_continue.assert_called_once_with()


@pytest.mark.asyncio
async def test_llm_intervention_publishes_running_when_accepted(monkeypatch):
    state = AnalysisState(
        user_id="user-1",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason=WaitInputReason.LLM_FAILURE.value,
    )
    workflow = MagicMock()
    workflow.signal_llm_failure_action.return_value = True
    sent = []

    async def fake_owner(session_id, user_id):
        return state

    async def fake_send(session_id, message):
        sent.append(message)
        return True

    monkeypatch.setattr(lifecycle, "require_session_ownership", fake_owner)
    monkeypatch.setattr(lifecycle, "workflows", {"session-1": workflow})
    monkeypatch.setattr(route_helpers, "send_analysis_json", fake_send)

    response = await lifecycle._llm_failure_action_locked(
        "session-1",
        "user-1",
        LlmFailureActionRequest(
            action="retry", node_name="pi_planning", generation=2,
        ),
    )

    assert response.success is True
    assert state.status == TaskStatus.RUNNING
    assert sent[-1]["content"]["change_type"] == "llm_retry"
    assert sent[-1]["content"]["status"] == "running"
    workflow.signal_llm_failure_action.assert_called_once_with(
        "pi_planning", "retry", 2,
    )
