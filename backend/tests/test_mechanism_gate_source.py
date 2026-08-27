import inspect
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.agents.bioinformatics import executor_report_gates


def test_gate_rejects_when_not_registered():
    src = inspect.getsource(executor_report_gates.check_mandatory_report_gates)
    assert "not context.mechanism_programs_registered" in src
    # gate scoped to stage-2 downstream exploration
    assert 'context.planning_stage != "stage1"' in src
    # bounded release
    assert "mechanism_gate_reject_count" in src
    assert "max_mechanism_rejections" in src
    assert 'build_gate_skip_note("data-driven mechanism program capture")' in src
