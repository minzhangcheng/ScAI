import inspect
from backend.workflow import agents as wf_agents
from backend.workflow import synthesis as wf_synth


def test_final_result_carries_mechanism_programs():
    src = inspect.getsource(wf_agents)
    assert '"mechanism_programs": result.get("mechanism_programs", [])' in src


def test_pi_synthesis_attaches_programs_to_round_result():
    src = inspect.getsource(wf_synth)
    assert "round_result.data_driven_mechanism_programs" in src
    # Defensive: bioinformatics_result is initialized to None, so guard with `or {}`.
    assert 'bioinformatics_result = state.get("bioinformatics_result") or {}' in src
    assert 'bioinformatics_result.get("mechanism_programs"' in src
