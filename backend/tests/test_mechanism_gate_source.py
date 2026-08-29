import inspect
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.agents.bioinformatics import executor_report_gates


def test_missing_registration_is_a_non_blocking_pi_observation():
    src = inspect.getsource(executor_report_gates.attach_report_observations)
    assert "not context.mechanism_programs_registered" in src
    assert 'context.planning_stage != "stage1"' in src
    assert "mechanism_gate_reject_count" not in src
    assert 'severity="advise"' in src
    assert "PI review observation; non-blocking" in src
