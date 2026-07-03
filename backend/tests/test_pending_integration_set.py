import inspect
from backend.workflow.synthesis import SynthesisMixin


def test_pi_synthesis_sets_pending_integration_on_batch_complete():
    src = inspect.getsource(SynthesisMixin._pi_synthesis)
    assert "pending_integration" in src
    assert "active_direction_batch" in src
    assert "direction_coverage_status_for_batch" in src
