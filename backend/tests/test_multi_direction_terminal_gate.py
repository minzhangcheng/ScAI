"""Test terminal multi-direction synthesis still shows the round-summary HITL."""
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import (
    AgentType,
    AnalysisPhase,
    AnalysisState,
    RoundResult,
    TaskStatus,
)
from backend.workflow.synthesis import SynthesisMixin


class DummyPiAgent:
    async def synthesize_round(self, *args, **kwargs):
        return RoundResult(
            round_number=4,
            pi_summary="Final direction summary",
            pathology_findings=[],
            bioinformatics_findings=["Final direction finding"],
            biochemistry_findings=[],
            literature_findings=[],
            next_steps=[],
            new_findings=False,
        )

    async def analyze_direction_correlations(self, analysis_state, directions_with_results):
        return "Integrated multi-direction synthesis"

    async def identify_data_needs(self, analysis_state, round_result):
        return {"pathology": [], "biochemistry": [], "other": []}


class DummyWorkflow(SynthesisMixin):
    def __init__(self):
        self.session_id = f"test-{uuid.uuid4()}"
        self.pi_agent = DummyPiAgent()
        self.state_callback = None
        self.config = SimpleNamespace(convergence_rounds=2, max_rounds=4)
        self.sent_messages = []
        self.statuses = []
        self._continue_event = SimpleNamespace(clear=lambda: None)
        self._supplementary_queue = []

    async def _require_core_agent(self, state, agent_name, node_name):
        return None

    async def _send_message(self, message_type, content, agent=AgentType.PI):
        self.sent_messages.append((message_type, content, agent))

    async def _send_state_sync(self, analysis_state, change_type):
        self.sent_messages.append(("state_sync", change_type, AgentType.PI))

    async def _create_auto_snapshot(self, analysis_state, phase):
        self.sent_messages.append(("snapshot", phase, AgentType.PI))

    async def _build_and_save_detailed_segment(self, *args, **kwargs):
        return None

    def _set_status(self, analysis_state, status):
        analysis_state.status = status
        self.statuses.append(status)


@pytest.mark.asyncio
async def test_terminal_multi_direction_synthesis_shows_round_wait_gate():
    dir1 = {"direction": "Direction one", "notes": ""}
    dir2 = {"direction": "Direction two", "notes": ""}
    dir3 = {"direction": "Direction three", "notes": ""}
    analysis_state = AnalysisState(
        session_id="terminal-multi-direction",
        user_id="test",
        current_round=4,
        max_rounds=4,
        analysis_phase=AnalysisPhase.INNOVATION,
        current_round_direction="Direction three",
        user_chosen_directions=[dir1, dir2, dir3],
        directions_queue=[dir3],
        active_direction_batch=[dir1, dir2, dir3],
        round_results=[
            RoundResult(round_number=1, phase="exploration"),
            RoundResult(round_number=2, phase="innovation", direction="Direction one", pi_summary="One", bioinformatics_findings=["finding"], evidence_status="substantive"),
            RoundResult(round_number=3, phase="innovation", direction="Direction two", pi_summary="Two", bioinformatics_findings=["finding"], evidence_status="substantive"),
        ],
    )
    workflow = DummyWorkflow()
    workflow.pi_agent.build_cross_modal_synthesis = AsyncMock(return_value=None)

    result = await workflow._pi_synthesis(
        {
            "analysis_state": analysis_state,
            "pathology_result": {"findings": []},
            "bioinformatics_result": {"findings": ["Final direction finding"], "status": "completed"},
            "biochemistry_result": {"findings": []},
            "literature_result": {"findings": []},
        }
    )

    assert analysis_state.status == TaskStatus.WAITING_INPUT

    waiting_round_summaries = [
        content
        for message_type, content, _agent in workflow.sent_messages
        if message_type == "round_summary" and isinstance(content, dict) and content.get("status") == "waiting_input"
    ]
    assert len(waiting_round_summaries) == 1
    assert waiting_round_summaries[0]["round"] == 4
    assert waiting_round_summaries[0]["summary"] == "Final direction summary"
    assert TaskStatus.WAITING_INPUT in workflow.statuses
    assert analysis_state.pending_integration is True


@pytest.mark.asyncio
async def test_no_wait_round_never_advertises_waiting_input():
    analysis_state = AnalysisState(
        session_id="terminal-no-wait",
        user_id="test",
        current_round=4,
        max_rounds=4,
        round_wait_seconds=0,
        analysis_phase=AnalysisPhase.INNOVATION,
        current_round_direction="Direction three",
        user_chosen_directions=[
            {"direction": "Direction one", "notes": ""},
            {"direction": "Direction two", "notes": ""},
            {"direction": "Direction three", "notes": ""},
        ],
        directions_queue=[{"direction": "Direction three", "notes": ""}],
    )
    workflow = DummyWorkflow()
    workflow.pi_agent.build_cross_modal_synthesis = AsyncMock(return_value=None)

    result = await workflow._pi_synthesis(
        {
            "analysis_state": analysis_state,
            "pathology_result": {"findings": []},
            "bioinformatics_result": {
                "findings": ["Final direction finding"],
                "status": "completed",
            },
            "biochemistry_result": {"findings": []},
            "literature_result": {"findings": []},
        }
    )

    assert analysis_state.status == TaskStatus.RUNNING
    assert TaskStatus.WAITING_INPUT not in workflow.statuses
    round_summary = next(
        content
        for message_type, content, _agent in workflow.sent_messages
        if message_type == "round_summary"
    )
    assert round_summary["status"] == "completed"
    assert round_summary["wait_policy"] == "no_wait"
    status_messages = [
        content
        for message_type, content, _agent in workflow.sent_messages
        if message_type == "status"
    ]
    assert any("No-wait mode is enabled" in content for content in status_messages)
    assert all("Waiting for your input" not in content for content in status_messages)

    await workflow._wait_for_user_input(result)

    assert analysis_state.status == TaskStatus.RUNNING
    assert ("state_sync", "auto_continue", AgentType.PI) in workflow.sent_messages
