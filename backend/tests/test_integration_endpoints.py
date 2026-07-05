import re, asyncio, types
import pytest
from unittest.mock import MagicMock
import backend.routes.interaction as ix
from backend.models.schemas import AnalysisState, AnalysisPhase, TaskStatus
from backend.models.api_requests import DirectionChoice
from backend.constants import PI_DECIDE_SIGNAL


def test_choose_direction_accepts_integration_gate_source():
    src = re.search(r"async def _choose_direction_locked\b.*?(?=\n(?:async )?def |\Z)",
                    open(ix.__file__).read(), re.DOTALL).group(0)
    assert "INTEGRATION" in src
    assert "active_direction_batch" in src
    assert "pending_integration" in src


def test_finalize_endpoint_registered():
    src = open(ix.__file__).read()
    assert "finalize-from-integration" in src
    assert "final_report_requested" in src


def test_finalize_only_valid_at_integration_gate_source():
    body = re.search(r"async def finalize_from_integration\b.*?(?=\n(?:async )?def |\Z)",
                     open(ix.__file__).read(), re.DOTALL).group(0)
    assert "INTEGRATION" in body and "400" in body


def test_integration_gate_accepts_let_pi_decide(monkeypatch):
    """let_pi_decide is now accepted at the integration gate — PI auto-selects directions."""
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.status = TaskStatus.WAITING_INPUT
    s.analysis_phase = AnalysisPhase.INTEGRATION
    s.integration_gate_candidates = [{"direction": "X"}]
    s.user_chosen_direction = None
    s.pending_human_input = None

    async def fake_owner(session_id, user_id): return s
    async def fake_update(session_id, state): return None
    async def fake_save(session_id, state): return None
    monkeypatch.setattr(ix, "require_session_ownership", fake_owner)
    monkeypatch.setattr(ix, "_update_session_cache", fake_update)
    monkeypatch.setattr(ix, "async_save_session", fake_save)
    monkeypatch.setattr(ix, "workflows", {})
    monkeypatch.setattr(ix, "resolve_integration_gate", lambda state, event_type: MagicMock())

    result = asyncio.run(ix._choose_direction_locked("sess", "u", DirectionChoice(let_pi_decide=True)))
    # Should succeed and switch to INNOVATION phase
    assert s.user_chosen_direction == PI_DECIDE_SIGNAL
    assert s.analysis_phase == AnalysisPhase.INNOVATION


def test_integration_direction_selection_clears_literature_evidence(monkeypatch):
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.status = TaskStatus.WAITING_INPUT
    s.analysis_phase = AnalysisPhase.INTEGRATION
    s.current_round = 4
    s.integration_cycle = 2
    s.integration_gate_candidates = [{"direction": "X"}]
    s.integration_gate_literature_evidence = [{"direction_id": "x"}]

    async def fake_owner(session_id, user_id): return s
    async def fake_update(session_id, state): return None
    async def fake_save(session_id, state): return None

    monkeypatch.setattr(ix, "require_session_ownership", fake_owner)
    monkeypatch.setattr(ix, "_update_session_cache", fake_update)
    monkeypatch.setattr(ix, "async_save_session", fake_save)
    monkeypatch.setitem(ix.workflows, "sess", None)
    monkeypatch.setattr(ix, "_pause_and_restart_ws", fake_update)

    asyncio.run(ix._choose_direction_locked(
        "sess",
        "u",
        DirectionChoice(directions=[{"direction": "X", "notes": "", "is_custom": False}]),
    ))

    assert s.direction_selection_records[0].literature_evidence == [{"direction_id": "x"}]
    assert s.integration_gate_literature_evidence == []


def test_integration_finalize_clears_literature_evidence(monkeypatch):
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.status = TaskStatus.WAITING_INPUT
    s.analysis_phase = AnalysisPhase.INTEGRATION
    s.current_round = 4
    s.integration_cycle = 2
    s.integration_gate_candidates = [{"direction": "X"}]
    s.integration_gate_literature_evidence = [{"direction_id": "x"}]

    async def fake_owner(session_id, user_id): return s
    async def fake_update(session_id, state): return None
    async def fake_save(session_id, state): return None

    monkeypatch.setattr(ix, "require_session_ownership", fake_owner)
    monkeypatch.setattr(ix, "_update_session_cache", fake_update)
    monkeypatch.setattr(ix, "async_save_session", fake_save)
    monkeypatch.setitem(ix.workflows, "sess", None)

    response = asyncio.run(ix.finalize_from_integration(
        "sess",
        types.SimpleNamespace(state=types.SimpleNamespace(user_id="u")),
    ))

    assert response.success is True
    assert s.direction_selection_records[0].literature_evidence == [{"direction_id": "x"}]
    assert s.integration_gate_literature_evidence == []
