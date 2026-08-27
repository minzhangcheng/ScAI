import inspect
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.agents.bioinformatics import executor_auxiliary, executor_loop, executor_registration


def test_run_autonomous_return_lists_mechanism_programs():
    src = inspect.getsource(executor_loop) + inspect.getsource(executor_auxiliary.finalize_run)
    assert '"mechanism_programs": context.mechanism_programs' in src
    assert "mechanism_programs: List" in src or "mechanism_programs = []" in src


def test_handler_branch_exists():
    dispatch_source = inspect.getsource(executor_loop)
    src = inspect.getsource(executor_registration.handle_register_mechanisms)
    assert '"register_mechanism_programs"' in dispatch_source
    assert "validate_mechanism_program_payload" in src
    assert "context.mechanism_programs_registered = True" in src


def test_process_passes_mechanism_programs_through():
    src = inspect.getsource(BioinformaticsAgent.process)
    assert 'result.get("mechanism_programs"' in src


def test_validated_programs_can_be_appended_to_analysis_state_contract():
    """Behavioral test for spec test #10: the handler's write CONTRACT
    (validate -> state.data_driven_mechanism_programs.extend) actually updates
    AnalysisState. This verifies the state-write contract, not the handler branch
    itself (which runs inside the live LLM tool loop)."""
    from backend.models.schemas import AnalysisState
    from backend.agents.bioinformatics_agent import validate_mechanism_program_payload

    state = AnalysisState(user_id="u1")
    payload = {"programs": [{
        "label": "test program",
        "triggering_observations": ["obs"],
        "evidence_execution_ids": ["exec-1"],
        "evidence_strength": "weak", "specificity": "low",
        "interpretation": "candidate, not confirmed",
    }]}
    progs, err = validate_mechanism_program_payload(payload, round_number=1)
    assert err is None
    state.data_driven_mechanism_programs.extend(progs)  # mirrors handler write
    assert len(state.data_driven_mechanism_programs) == 1
    assert state.data_driven_mechanism_programs[0].round_number == 1
