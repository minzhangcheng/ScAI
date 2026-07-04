import asyncio
import inspect
import types
from backend.models.schemas import AnalysisState, AnalysisPhase, TaskStatus
from backend.workflow.core import SingleCellAnalysisWorkflow


def test_resume_helper_skips_stop_guard_at_integration_gate():
    """resume_restored_wait_state must NOT call _should_continue or
    _check_convergence when parked at the integration gate — those guards
    would route to generate_final_report, bypassing the wait_for_user_input
    checkpoint that lets the user choose new directions."""
    import inspect
    from backend.workflow.core import SingleCellAnalysisWorkflow

    src = inspect.getsource(SingleCellAnalysisWorkflow.resume_restored_wait_state)
    assert "at_integration_gate(analysis_state)" in src
    assert "if not is_integration_gate" in src


def test_resume_helper_does_not_generate_report_at_integration_gate():
    """Functional test: at the integration gate with current_round >> max_rounds,
    resume_restored_wait_state must NOT call _generate_final_report."""
    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)

    state = AnalysisState(name="t", user_id="u", initial_instruction="o")
    state.analysis_phase = AnalysisPhase.INTEGRATION
    state.integration_gate_candidates = [{"direction": "NEW"}]
    state.integration_cycle = 1
    state.integration_reports = ["INTEGRATION TEXT"]
    state.current_round = 10
    state.max_rounds = 3
    state.status = TaskStatus.RUNNING
    state.detailed_report_html_path = None

    called = {"report": False, "ainvoke": False, "should_continue": False}

    async def fake_send_message(*a, **kw):
        pass

    async def fake_generate_final_report(s):
        called["report"] = True
        return s

    def fake_should_continue(s):
        called["should_continue"] = True
        return "stop"

    async def fake_check_convergence(s):
        return s

    def fake_check_convergence_condition(s):
        return "stop"

    wf._send_message = fake_send_message
    wf._generate_final_report = fake_generate_final_report
    wf._should_continue = fake_should_continue
    wf._check_convergence = fake_check_convergence
    wf._check_convergence_condition = fake_check_convergence_condition
    wf.session_id = "s"
    wf.compiled_graph = types.SimpleNamespace()

    async def fake_ainvoke(ws, config):
        called["ainvoke"] = True
        return {"analysis_state": state}

    wf.compiled_graph.ainvoke = fake_ainvoke

    async def fake_initialize():
        pass

    wf.initialize = fake_initialize
    wf._set_status = lambda s, st: setattr(s, "status", st)
    wf._consume_restore_resume_metadata = lambda s: None
    wf._make_initial_workflow_state = lambda s, current_round=0: {
        "analysis_state": s, "current_round": current_round
    }
    wf.config = types.SimpleNamespace(max_rounds=3)

    result = asyncio.run(wf.resume_restored_wait_state(state))

    assert called["report"] is False, "_generate_final_report must NOT be called at integration gate"
    assert called["ainvoke"] is True, "compiled_graph.ainvoke must be called to resume the checkpoint"
