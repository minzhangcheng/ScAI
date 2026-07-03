import inspect
from backend.workflow.synthesis import SynthesisMixin
from backend.workflow import synthesis as syn


def test_pi_synthesis_no_inline_correlation():
    src = inspect.getsource(SynthesisMixin._pi_synthesis)
    assert "analyze_direction_correlations" not in src


def test_should_run_direction_correlation_deleted():
    assert not hasattr(syn, "_should_run_direction_correlation")


def test_multi_direction_completed_decoupled():
    src = inspect.getsource(syn._multi_direction_analysis_completed)
    assert "multi_direction_correlation" not in src
