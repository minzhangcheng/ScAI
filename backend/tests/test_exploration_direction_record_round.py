import asyncio, types
from unittest.mock import AsyncMock
from backend.models.schemas import AnalysisState, DirectionSelectionRecord, AnalysisPhase
from backend.workflow.planning import PlanningMixin


class _ExplorationHarness(PlanningMixin):
    """Minimal harness to exercise the exploration PI_DECIDE fallback branch."""
    def __init__(self, select_best_return=None):
        self.session_id = "test"
        self.config = types.SimpleNamespace(max_rounds=10)

        pi = types.SimpleNamespace()
        pi.reset_history = lambda: None
        pi.select_best_directions = AsyncMock(return_value=select_best_return or {})
        pi.innovation_instruction = AsyncMock(return_value={
            "pathology_tasks": [], "bioinformatics_tasks": [],
            "biochemistry_tasks": [], "literature_topics": [],
        })
        self.pi_agent = pi

        self.sent = []
        self._continue_event = types.SimpleNamespace()
        self._continue_event.is_set = lambda: True
        self._continue_event.clear = lambda: None

    async def _send_message(self, t, c, agent=None): self.sent.append((t, c))
    async def _send_state_sync(self, state, ct): pass
    async def _require_core_agent(self, state, name, node): return None


def _exploration_decide_state():
    """State where current_round=2 (after _initialize_round increment),
    no open pi_decide record exists, so the fallback creates one."""
    s = AnalysisState(id="test", name="T", user_id="u", initial_instruction="o")
    s.current_round = 2
    s.analysis_phase = AnalysisPhase.INNOVATION
    s.integration_cycle = 0
    s.user_chosen_direction = None
    s.pending_integration = False
    return s


def test_exploration_pi_decide_uses_round_minus_one():
    """When no open exploration pi_decide record exists and current_round=2,
    the fallback get_or_open_direction_record should use round_number=1
    (current_round - 1), not round_number=2."""
    s = _exploration_decide_state()
    selection_return = {
        "selected_directions": [{"direction": "Dir A", "notes": ""}],
    }
    h = _ExplorationHarness(select_best_return=selection_return)
    out = asyncio.run(h._pi_planning({"analysis_state": s, "messages": []}))

    exploration_recs = [r for r in s.direction_selection_records
                        if r.phase == "exploration" and r.event_type == "pi_decide"]
    assert len(exploration_recs) == 1
    rec = exploration_recs[0]
    assert rec.round_number == 1
    assert rec.let_pi_decide is True
