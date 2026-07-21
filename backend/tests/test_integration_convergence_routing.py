import types
from backend.models.schemas import AnalysisState, RoundResult, TaskStatus
from backend.workflow.routing import RoutingMixin


class _Router(RoutingMixin):
    def __init__(self):
        self.session_id = "s"; self.config = types.SimpleNamespace(max_rounds=3)


def _running(**kw):
    s = AnalysisState(name="t", user_id="u", initial_instruction="o"); s.status = TaskStatus.RUNNING
    for k, v in kw.items(): setattr(s, k, v)
    return s


def test_pending_integration_routes_integrate():
    assert _Router()._check_convergence_condition(
        {"analysis_state": _running(current_round=3, pending_integration=True,
                                    integration_cycle=0, max_integration_cycles=3)}
    ) == "integrate"


def test_cap_reached_routes_stop():
    assert _Router()._check_convergence_condition(
        {"analysis_state": _running(current_round=9, pending_integration=True,
                                    integration_cycle=3, max_integration_cycles=3)}
    ) == "stop"


def test_unfinished_batch_routes_continue():
    assert _Router()._check_convergence_condition(
        {"analysis_state": _running(current_round=2, active_direction_batch=[{"direction": "B"}])}
    ) == "continue"


def test_attempted_partial_batch_routes_stop():
    direction = "B"
    state = _running(
        current_round=3,
        active_direction_batch=[{"direction": direction}],
        round_results=[RoundResult(
            round_number=3,
            phase="innovation",
            direction=direction,
            pi_summary="Partial result",
            bioinformatics_findings=["finding"],
            round_status="partial",
            evidence_status="substantive",
            direction_coverage_eligible=False,
        )],
    )
    assert _Router()._check_convergence_condition(
        {"analysis_state": state}
    ) == "stop"
