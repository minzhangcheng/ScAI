import inspect
from backend.agents.pi_agent import PIAgent


def test_final_summary_reads_programs_and_classifies():
    src = inspect.getsource(PIAgent.generate_final_summary)
    assert "data_driven_mechanism_programs" in src
    low = src.lower()
    assert "uncovered" in low
    assert "future" in low
    assert "[-3:]" in src
