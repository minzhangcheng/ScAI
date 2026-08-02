"""User-requested supplementary analysis: merging its output back.

`_initialize_round` seeds every agent-result channel with an explicit `None`, and
`_current_graph_state` is the bioinformatics node's INPUT dict, so these keys are
present-but-None for the whole round. That made `.get(key, {})` return `None`, and
the merge then assigned into it — raising for every supplementary target while the
analysis had already been paid for. Nothing covered this path before.
"""
import pytest

from backend.models.schemas import AnalysisState, AnalysisStep, CodeExecution
from backend.workflow.core import SingleCellAnalysisWorkflow


class _Host(SingleCellAnalysisWorkflow):
    """Bypasses __init__: only the two merge helpers are under test."""

    def __init__(self, graph_state=None):
        self.session_id = "sess-merge"
        self._current_graph_state = graph_state


def _seeded_state():
    """What _initialize_round actually puts on the channel."""
    return {
        "bioinformatics_result": None,
        "literature_result": None,
        "pathology_result": None,
    }


@pytest.mark.parametrize("agent_key", ["bioinformatics_result", "literature_result"])
def test_merge_survives_a_present_but_none_channel(agent_key):
    host = _Host(_seeded_state())

    host._merge_supplementary_findings(agent_key, ["a supplementary finding"])

    assert host._current_graph_state[agent_key] == {
        "findings": ["[Supplementary] a supplementary finding"]
    }


def test_merge_appends_to_an_existing_result():
    host = _Host({"bioinformatics_result": {"findings": ["original"], "status": "completed"}})

    host._merge_supplementary_findings("bioinformatics_result", ["extra"])

    result = host._current_graph_state["bioinformatics_result"]
    assert result["findings"] == ["original", "[Supplementary] extra"]
    assert result["status"] == "completed", "unrelated keys must survive"


def test_merge_handles_an_absent_key_and_no_findings():
    # A non-empty state missing the key: an empty dict is falsy and short-circuits,
    # which is the pre-existing "no graph context yet" behaviour.
    host = _Host({"analysis_state": object()})
    host._merge_supplementary_findings("bioinformatics_result", ["only finding"])
    assert host._current_graph_state["bioinformatics_result"]["findings"] == [
        "[Supplementary] only finding"
    ]

    # No findings, no graph state, and an empty graph state are all no-ops.
    host._merge_supplementary_findings("bioinformatics_result", [])
    assert host._current_graph_state["bioinformatics_result"]["findings"] == [
        "[Supplementary] only finding"
    ]
    _Host(None)._merge_supplementary_findings("bioinformatics_result", ["x"])
    empty = _Host({})
    empty._merge_supplementary_findings("bioinformatics_result", ["x"])
    assert empty._current_graph_state == {}


def _state():
    state = AnalysisState(name="t", user_id="u", initial_instruction="t")
    state.current_round = 3
    return state


def test_supplementary_executions_become_citable_evidence():
    """Without this the pass runs, spends kernel time, and leaves nothing a finding
    can be traced to — unlike the round path and the automatic review pass."""
    host, state = _Host({}), _state()
    supp = {
        "code_history": [CodeExecution(code="dc.run_ulm(adata)")],
        "analysis_log": [AnalysisStep(step_number=1, step_type="execute_code",
                                      code="dc.run_ulm(adata)")],
    }

    host._absorb_supplementary_executions(state, supp)

    assert [c.code for c in state.code_history] == ["dc.run_ulm(adata)"]
    assert [s.round_number for s in state.analysis_log] == [3]


def test_absorbing_twice_does_not_duplicate():
    host, state = _Host({}), _state()
    supp = {
        "code_history": [CodeExecution(code="dc.run_ulm(adata)")],
        "analysis_log": [AnalysisStep(step_number=1, step_type="execute_code", code="x")],
    }

    host._absorb_supplementary_executions(state, supp)
    host._absorb_supplementary_executions(state, supp)

    assert len(state.code_history) == 1
    assert len(state.analysis_log) == 1


def test_absorb_accepts_dict_shaped_log_entries_and_never_raises():
    host, state = _Host({}), _state()

    host._absorb_supplementary_executions(state, {
        "analysis_log": [{"step_number": 2, "step_type": "execute_code", "code": "y"}],
    })
    assert len(state.analysis_log) == 1

    # Malformed input must degrade, not fail the round the user is waiting on.
    host._absorb_supplementary_executions(state, {"code_history": ["not an execution"]})
    host._absorb_supplementary_executions(state, {})


def test_round_context_reaches_the_pi_for_the_round_checkpoint():
    """The round checkpoint used to fall through to the direction branch, so the PI
    was told it was choosing research directions and got none of the round's
    results to plan against."""
    import inspect

    from backend.agents.pi.research_planning import ResearchPlanningMixin

    source = inspect.getsource(ResearchPlanningMixin.plan_supplementary_analysis)
    assert 'elif context == "round":' in source, "the round checkpoint needs its own branch"
    assert "completed analysis round" in source
    assert "round_context" in source, "the round's own results must reach the prompt"


def test_user_requested_supplementary_keeps_publication_figure_review_callback():
    """Only the automatic restricted repair pass is figure-free. A user-requested
    HITL supplement uses the main autonomous executor and must retain its blocking
    publication-figure review callback."""
    import inspect

    source = inspect.getsource(SingleCellAnalysisWorkflow._run_supplementary_analysis)
    assert "run_autonomous_analysis" in source
    assert "figure_review_callback=self._figure_review_callback" in source
