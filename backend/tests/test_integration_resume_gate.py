import inspect
import backend.workflow.core as core_mod


def test_run_and_resume_recognize_integration_gate():
    src = inspect.getsource(core_mod)
    assert "SEMANTIC_WAIT_GATES" in src
    assert src.count("SEMANTIC_WAIT_GATES") >= 2


def test_resume_helper_resends_integration_gate():
    src = inspect.getsource(core_mod.SingleCellAnalysisWorkflow.resume_restored_wait_state)
    assert "integration_gate" in src
