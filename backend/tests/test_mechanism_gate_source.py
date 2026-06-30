import inspect
from backend.agents.bioinformatics_agent import BioinformaticsAgent


def test_gate_rejects_when_not_registered():
    src = inspect.getsource(BioinformaticsAgent.run_autonomous_analysis)
    assert "not mechanism_programs_registered" in src
    # gate scoped to stage-2 downstream exploration
    assert 'planning_stage != "stage1"' in src
    # bounded release
    assert "mechanism_gate_reject_count" in src
    assert "MAX_MECHANISM_REJECTIONS" in src
    assert 'build_gate_skip_note("data-driven mechanism program capture")' in src
