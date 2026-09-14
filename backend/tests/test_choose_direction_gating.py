"""P1-5: choose-direction must only accept calls at the post-exploration gate."""
import asyncio
import re
from unittest.mock import MagicMock

import backend.routes._helpers as route_helpers
import backend.routes.interaction as ix_mod
from backend.models.api_requests import DirectionChoice
from backend.models.schemas import (
    AnalysisPhase,
    AnalysisState,
    ExplorationReport,
    TaskStatus,
)


def test_choose_direction_checks_for_direction_gate():
    """P1-5: choose-direction must verify session is at post-exploration gate.

    After P1-9 session-lock refactor, the body lives in `_choose_direction_locked`.
    Search both the public handler and the locked helper.
    """
    src = open(ix_mod.__file__).read()
    fn = re.search(r"async def _choose_direction_locked\b.*?(?=\n(?:async )?def |\Z)", src, re.DOTALL)
    if not fn:
        fn = re.search(r"async def choose_direction\b.*?(?=\n(?:async )?def |\Z)", src, re.DOTALL)
    assert fn
    body = fn.group(0)
    # Must check current_round AND analysis_phase, not just status
    assert "current_round" in body, "P1-5: must check current_round"
    assert "INNOVATION" in body or "analysis_phase" in body, \
        "P1-5: must check analysis_phase"
    # Must reject with 400 if not at gate
    assert "400" in body and (
        ("direction" in body.lower() and "gate" in body.lower())
        or "direction selection" in body.lower()
    ), "P1-5: must reject non-gate calls with 400"


def test_choose_direction_repairs_post_exploration_phase_drift(monkeypatch):
    """A completed exploration report is authoritative after snapshot resume.

    A pre-direction snapshot failure can leave a restored checkpoint at
    WAITING_INPUT with phase=EXPLORATION even though Report A and its direction
    candidates already exist. Direction selection must repair that drift rather
    than reject the visible post-exploration gate.
    """
    state = AnalysisState(
        id="sess-direction-drift",
        name="T",
        user_id="u1",
        initial_instruction="analyze",
        status=TaskStatus.WAITING_INPUT,
        current_round=1,
        analysis_phase=AnalysisPhase.EXPLORATION,
        exploration_report=ExplorationReport(
            research_direction_candidates=[{"direction": "Direction A"}],
        ),
    )

    async def fake_owner(session_id, user_id):
        return state

    async def fake_update(session_id, updated):
        return None

    async def fake_save(session_id, updated):
        return None

    monkeypatch.setattr(ix_mod, "require_session_ownership", fake_owner)
    monkeypatch.setattr(ix_mod, "_update_session_cache", fake_update)
    monkeypatch.setattr(ix_mod, "async_save_session", fake_save)
    monkeypatch.setattr(ix_mod, "workflows", {})
    monkeypatch.setattr(ix_mod, "_pause_and_restart_ws", fake_update)

    response = asyncio.run(ix_mod._choose_direction_locked(
        "sess-direction-drift",
        "u1",
        DirectionChoice(directions=[{
            "direction": "Direction A",
            "notes": "",
            "is_custom": False,
        }]),
    ))

    assert response.success is True
    assert state.analysis_phase == AnalysisPhase.INNOVATION
    assert state.user_chosen_directions[0]["direction"] == "Direction A"


def test_choose_direction_publishes_running_when_active_workflow_accepts(monkeypatch):
    """Accepted input must leave WAITING_INPUT before the workflow consumes it.

    Decision-snapshot copying can keep the workflow from reaching its wait loop
    for several seconds.  The route already owns the accepted direction during
    that interval, so the UI should not continue inviting another submission.
    """
    state = AnalysisState(
        id="sess-direction-accepted",
        name="T",
        user_id="u1",
        initial_instruction="analyze",
        status=TaskStatus.WAITING_INPUT,
        current_round=1,
        analysis_phase=AnalysisPhase.INNOVATION,
        exploration_report=ExplorationReport(
            research_direction_candidates=[{"direction": "Direction A"}],
        ),
    )
    state.waiting_input_started_at = state.updated_at
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

    monkeypatch.setattr(ix_mod, "require_session_ownership", fake_owner)
    monkeypatch.setattr(ix_mod, "_update_session_cache", fake_update)
    monkeypatch.setattr(ix_mod, "async_save_session", fake_save)
    monkeypatch.setattr(route_helpers, "send_analysis_json", fake_send)
    monkeypatch.setattr(ix_mod, "workflows", {state.id: workflow})

    response = asyncio.run(ix_mod._choose_direction_locked(
        state.id,
        "u1",
        DirectionChoice(directions=[{
            "direction": "Direction A",
            "notes": "",
            "is_custom": False,
        }]),
    ))

    assert response.success is True
    assert state.status == TaskStatus.RUNNING
    assert state.waiting_input_started_at is None
    assert state.state_version == 1
    sync = next(
        item[1] for item in events
        if isinstance(item, tuple) and item[0] == "sync"
    )
    assert sync["type"] == "state_sync"
    assert sync["content"] == {
        "change_type": "direction_selected",
        "version": 1,
        "status": "running",
        "current_round": 1,
        "analysis_phase": AnalysisPhase.INNOVATION,
        "timestamp": state.last_sync_timestamp.isoformat(),
    }
    assert events[0:2] == ["cache", "save"]
    assert events[2][0] == "sync"
    workflow.signal_continue.assert_called_once_with()
