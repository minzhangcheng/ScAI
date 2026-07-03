import re, asyncio, types
import pytest
import backend.routes.interaction as ix
from backend.models.schemas import AnalysisState, AnalysisPhase, TaskStatus
from backend.models.api_requests import DirectionChoice


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


def test_integration_gate_rejects_let_pi_decide(monkeypatch):
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.status = TaskStatus.WAITING_INPUT
    s.analysis_phase = AnalysisPhase.INTEGRATION
    s.integration_gate_candidates = [{"direction": "X"}]

    async def fake_owner(session_id, user_id): return s
    monkeypatch.setattr(ix, "require_session_ownership", fake_owner)

    with pytest.raises(ix.HTTPException) as ei:
        asyncio.run(ix._choose_direction_locked("sess", "u", DirectionChoice(let_pi_decide=True)))
    assert ei.value.status_code == 400
    assert s.active_direction_batch == []


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
