import inspect
from backend.workflow.synthesis import SynthesisMixin


def test_pi_synthesis_sets_pending_integration_on_batch_complete():
    src = inspect.getsource(SynthesisMixin._pi_synthesis)
    assert "pending_integration" in src
    # _pi_synthesis now delegates batch-coverage to the active_batch_complete
    # predicate (extracted in the 0.32.3 refactor) instead of inlining
    # active_direction_batch / direction_coverage_status_for_batch.
    assert "active_batch_complete" in src
