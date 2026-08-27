import inspect
from backend.agents.supervisor_agent import SupervisorAgent
from backend.agents.supervisor import report_prompt


def test_supervisor_report_prompt_emits_mechanism_block():
    src = inspect.getsource(report_prompt._build_report_prompt)
    assert "data_driven_mechanism_programs" in src
    # must be bounded: most recent 3 rounds
    assert "round_results[-3:]" in src or "[-3:]" in src
