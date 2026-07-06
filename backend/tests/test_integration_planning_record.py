import asyncio, types
from unittest.mock import AsyncMock
from backend.constants import PI_DECIDE_SIGNAL
from backend.models.schemas import AnalysisState, DirectionSelectionRecord, AnalysisPhase
from backend.workflow.planning import PlanningMixin


class _PlanningHarness(PlanningMixin):
    """Minimal harness to exercise the is_integration_decide branch of _pi_planning."""
    def __init__(self, select_integration_return=None, innovation_instruction_return=None):
        self.session_id = "test"
        self.config = types.SimpleNamespace(max_rounds=10)

        pi = types.SimpleNamespace()
        pi.reset_history = lambda: None
        pi.select_integration_directions = AsyncMock(return_value=select_integration_return or {})
        pi.innovation_instruction = AsyncMock(return_value=innovation_instruction_return or {"pathology_tasks": [], "bioinformatics_tasks": [], "biochemistry_tasks": [], "literature_topics": []})
        self.pi_agent = pi

        self.sent = []
        self._continue_event = types.SimpleNamespace()
        self._continue_event.is_set = lambda: True
        self._continue_event.clear = lambda: None

    async def _send_message(self, t, c, agent=None): self.sent.append((t, c))
    async def _send_state_sync(self, state, ct): pass
    async def _require_core_agent(self, state, name, node): return None


def _integration_decide_state(integration_cycle=1, gate_candidates=None):
    s = AnalysisState(id="test", name="T", user_id="u", initial_instruction="o")
    s.current_round = 6
    s.analysis_phase = AnalysisPhase.INNOVATION
    s.integration_cycle = integration_cycle
    s.integration_gate_candidates = gate_candidates or [{"direction": "Integration Dir A"}]
    s.integration_gate_literature_evidence = []
    s.pending_integration = False
    s.user_chosen_direction = PI_DECIDE_SIGNAL
    return s


def test_integration_pi_decide_closes_existing_open_record():
    """When an open integration record exists, _pi_planning closes it as pi_decide
    with integration gate candidates (NOT exploration candidates)."""
    gate_cands = [{"direction": "Integration Dir A"}]
    s = _integration_decide_state(gate_candidates=gate_cands)
    open_rec = DirectionSelectionRecord(
        round_number=5, phase="integration", integration_cycle=1,
        event_type="open", let_pi_decide=True,
    )
    open_rec.candidates = list(gate_cands)
    s.direction_selection_records = [open_rec]

    selection_return = {
        "selected_directions": [{"direction": "Integration Dir A", "notes": ""}],
    }
    h = _PlanningHarness(select_integration_return=selection_return)
    out = asyncio.run(h._pi_planning({"analysis_state": s, "messages": []}))

    assert open_rec.event_type == "pi_decide"
    assert open_rec.candidates == gate_cands
    assert open_rec.pi_selected == [{"direction": "Integration Dir A", "notes": ""}]
    assert s.integration_gate_candidates == []
    assert s.integration_gate_literature_evidence == []


def test_integration_pi_decide_creates_record_when_none():
    """When no open record exists (auto-skip path), _pi_planning creates a new
    pi_decide record with round_number = current_round - 1."""
    gate_cands = [{"direction": "Integration Dir A"}]
    s = _integration_decide_state(gate_candidates=gate_cands)
    assert len(s.direction_selection_records) == 0

    selection_return = {
        "selected_directions": [{"direction": "Integration Dir A", "notes": ""}],
    }
    h = _PlanningHarness(select_integration_return=selection_return)
    out = asyncio.run(h._pi_planning({"analysis_state": s, "messages": []}))

    pi_decide_recs = [r for r in s.direction_selection_records
                      if r.event_type == "pi_decide" and r.phase == "integration"]
    assert len(pi_decide_recs) == 1
    rec = pi_decide_recs[0]
    assert rec.round_number == 5  # current_round - 1
    assert rec.candidates == gate_cands
    assert rec.let_pi_decide is True
    assert s.integration_gate_candidates == []


def test_integration_pi_decide_zero_directions_still_records():
    """When PI selects 0 integration directions, a pi_decide record is still created
    with pi_selected=[], and gate state is cleared."""
    gate_cands = [{"direction": "Integration Dir A"}]
    s = _integration_decide_state(gate_candidates=gate_cands)

    h = _PlanningHarness(select_integration_return={"selected_directions": []})
    out = asyncio.run(h._pi_planning({"analysis_state": s, "messages": []}))

    pi_decide_recs = [r for r in s.direction_selection_records
                      if r.event_type == "pi_decide" and r.phase == "integration"]
    assert len(pi_decide_recs) == 1
    rec = pi_decide_recs[0]
    assert rec.pi_selected == []
    assert s.integration_gate_candidates == []
    assert s.integration_gate_literature_evidence == []
