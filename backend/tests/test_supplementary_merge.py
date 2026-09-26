"""User-requested supplementary analysis: merging its output back.

`_initialize_round` seeds every agent-result channel with an explicit `None`, and
`_current_graph_state` is the bioinformatics node's INPUT dict, so these keys are
present-but-None for the whole round. That made `.get(key, {})` return `None`, and
the merge then assigned into it — raising for every supplementary target while the
analysis had already been paid for. Nothing covered this path before.
"""

from backend.models.schemas import AnalysisState, AnalysisStep, CodeExecution
from backend.workflow.core import SingleCellAnalysisWorkflow


class _Host(SingleCellAnalysisWorkflow):
    """Bypasses __init__: only the two merge helpers are under test."""

    def __init__(self, graph_state=None):
        self.session_id = "sess-merge"
        self._current_graph_state = graph_state


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


def test_user_requested_supplementary_keeps_publication_figure_review_callback():
    """Only the automatic restricted repair pass is figure-free. A user-requested
    HITL supplement uses the main autonomous executor and must retain its blocking
    publication-figure review callback."""
    import inspect

    from backend.workflow.core_parts import supplementary

    source = inspect.getsource(supplementary._run_supplementary_analysis)
    assert "run_autonomous_analysis" in source
    assert "figure_review_callback=self._figure_review_callback" in source
