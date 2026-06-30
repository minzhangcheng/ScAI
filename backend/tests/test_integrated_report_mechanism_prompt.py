import inspect
from backend.agents.pi_agent import PIAgent


def test_integrated_report_includes_mechanism_block():
    src = inspect.getsource(PIAgent.generate_integrated_report)
    assert "data_driven_mechanism_programs" in src
    assert "candidate mechanism" in src.lower() or "candidate mechanisms" in src.lower()
