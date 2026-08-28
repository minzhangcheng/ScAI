import inspect
import backend.workflow.core as core_mod
from backend.workflow.core_parts import execution, initialization


def test_run_and_resume_recognize_integration_gate():
    src = inspect.getsource(execution)
    assert "SEMANTIC_WAIT_GATES" in src
    assert src.count("SEMANTIC_WAIT_GATES") >= 2


def test_resume_helper_routes_semantic_action_to_exact_node():
    src = inspect.getsource(initialization.resume_restored_wait_state)
    # A restored gate is rendered from its persisted state before an action.
    # Once the action is accepted, continuation must convert it to the exact
    # post-action graph node instead of re-sending/re-entering the old gate.
    assert "prepare_restored_semantic_action" in src
    assert "_invoke_restore_entry" in src
    assert "_check_convergence(" not in src
