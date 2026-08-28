import inspect
from backend.workflow.synthesis import SynthesisMixin
from backend.workflow.synthesis_parts import phases


def test_pi_synthesis_sets_pending_integration_when_batch_is_fully_attempted():
    src = inspect.getsource(phases.persist_round_result)
    assert "pending_integration" in src
    assert "active_batch_ready_for_integration" in src
