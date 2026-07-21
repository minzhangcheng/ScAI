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


def test_pending_integration_prevents_maxrounds_stop():
    assert _Router()._should_continue(
        {"analysis_state": _running(current_round=3, max_rounds=3, pending_integration=True)}
    ) == "continue"


def test_unfinished_batch_prevents_maxrounds_stop():
    assert _Router()._should_continue(
        {"analysis_state": _running(current_round=3, max_rounds=3,
                                    active_direction_batch=[{"direction": "C"}])}
    ) == "continue"


def test_attempted_partial_batch_stops_instead_of_looping():
    direction = "C"
    state = _running(
        current_round=3,
        max_rounds=3,
        active_direction_batch=[{"direction": direction}],
        round_results=[RoundResult(
            round_number=3,
            phase="innovation",
            direction=direction,
            pi_summary="Core analysis remained partial",
            bioinformatics_findings=["finding"],
            round_status="partial",
            evidence_status="substantive",
            direction_coverage_eligible=False,
        )],
    )
    assert _Router()._should_continue({"analysis_state": state}) == "stop"


def test_plain_maxrounds_stop_still_holds():
    assert _Router()._should_continue(
        {"analysis_state": _running(current_round=3, max_rounds=3)}
    ) == "stop"
