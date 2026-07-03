import types
from backend.models.schemas import AnalysisState, AnalysisPhase, TaskStatus
from backend.workflow.routing import RoutingMixin


class _Router(RoutingMixin):
    def __init__(self):
        self.session_id = "s"
        self.config = types.SimpleNamespace(max_rounds=3)


def _running(**kw):
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.status = TaskStatus.RUNNING
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_integration_gate_waiting_prevents_maxrounds_stop():
    """At the integration gate (phase=INTEGRATION + candidates present),
    _should_continue must not hard-stop on max_rounds."""
    s = _running(
        current_round=10,
        max_rounds=3,
    )
    s.analysis_phase = AnalysisPhase.INTEGRATION
    s.integration_gate_candidates = [{"direction": "NEW"}]
    s.pending_integration = False
    s.active_direction_batch = []
    assert _Router()._should_continue({"analysis_state": s}) == "continue"


def test_integration_gate_without_candidates_still_stops_on_maxrounds():
    """If phase=INTEGRATION but no candidates (short-circuit path), max_rounds
    stop should still hold — the integration node already set should_continue=False."""
    s = _running(current_round=10, max_rounds=3)
    s.analysis_phase = AnalysisPhase.INTEGRATION
    s.integration_gate_candidates = []
    s.pending_integration = False
    s.active_direction_batch = []
    assert _Router()._should_continue({"analysis_state": s}) == "stop"
