import inspect
from backend.agents.pi_agent import PIAgent


def test_direction_prompt_includes_mechanism_programs_and_no_hardcode():
    src = inspect.getsource(PIAgent.generate_raw_direction_candidates)
    assert "Data-Driven Mechanism Programs" in src
    assert "data_driven_mechanism_programs" in src
    assert "MECHANISM_EVIDENCE_POLICY" in src
    lowered = src.lower()
    for banned in ("ferroptosis", "cuproptosis"):
        assert banned not in lowered
