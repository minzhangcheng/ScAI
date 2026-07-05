"""After an integration-gate auto-skip sets PI_DECIDE_SIGNAL, routing must reach
pi_planning (continue) rather than stopping on max_rounds."""
from types import SimpleNamespace

from backend.constants import PI_DECIDE_SIGNAL
from backend.models.schemas import (
    AnalysisState, AnalysisPhase, TaskStatus, RoundResult,
)
from backend.workflow.direction_coverage import awaiting_integration_direction
from backend.workflow.routing import RoutingMixin


class _Router(RoutingMixin):
    session_id = "s"
    config = SimpleNamespace(max_rounds=5)


def _completed_dir(round_number: int, direction: str) -> RoundResult:
    return RoundResult(
        round_number=round_number, phase="innovation", direction=direction,
        lineage_id="main", pi_summary=f"done {direction}",
        round_status="completed", evidence_status="substantive",
        bioinformatics_findings=["f1"],
    )


def _state_after_integration_autoskip(max_rounds: int) -> AnalysisState:
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.analysis_phase = AnalysisPhase.INNOVATION
    s.status = TaskStatus.RUNNING
    s.user_chosen_direction = PI_DECIDE_SIGNAL
    s.pending_integration = False
    s.integration_cycle = 1
    s.max_integration_cycles = 3
    s.current_round = 3
    s.max_rounds = max_rounds
    s.active_direction_batch = [{"direction": "A"}, {"direction": "B"}]
    s.user_chosen_directions = [{"direction": "A"}, {"direction": "B"}]
    s.round_results = [_completed_dir(2, "A"), _completed_dir(3, "B")]
    return s


def test_predicate_true_only_for_integration_pi_decide():
    s = _state_after_integration_autoskip(max_rounds=3)
    assert awaiting_integration_direction(s) is True

    s.integration_cycle = 0
    assert awaiting_integration_direction(s) is False

    s.integration_cycle = 1
    s.user_chosen_direction = "some explicit direction"
    assert awaiting_integration_direction(s) is False


def test_routing_reaches_pi_planning_when_budget_exhausted():
    r = _Router()
    s = _state_after_integration_autoskip(max_rounds=3)
    state = {"analysis_state": s, "should_continue": True}
    assert r._should_continue(state) == "continue"
    assert r._check_convergence_condition(state) == "continue"


def test_routing_still_stops_when_integration_cap_reached():
    r = _Router()
    s = _state_after_integration_autoskip(max_rounds=3)
    s.integration_cycle = 3
    state = {"analysis_state": s, "should_continue": True}
    assert r._check_convergence_condition(state) == "stop"
