import asyncio
import inspect
import types
from backend.models.schemas import AnalysisState, AnalysisPhase, TaskStatus
from backend.workflow.core import SingleCellAnalysisWorkflow
from backend.workflow.core_parts import initialization


def test_resume_helper_skips_stop_guard_at_integration_gate():
    """Restored actions use their exact node, never convergence guards."""
    import inspect
    from backend.workflow.core import SingleCellAnalysisWorkflow

    src = inspect.getsource(initialization.resume_restored_wait_state)
    assert "prepare_restored_semantic_action" in src
    assert "_invoke_restore_entry" in src
    assert "_should_continue(" not in src
    assert "_check_convergence(" not in src


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
    state.restore_resume_kind = "semantic_gate"
    state.restore_resume_gate = "integration_selection"

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
    async def fake_restore_entry(s, thread_id):
        called["ainvoke"] = True
        assert s.restore_resume_kind == "graph_node"
        assert s.restore_resume_node == "check_convergence"
        return s

    wf._invoke_restore_entry = fake_restore_entry
    wf.config = types.SimpleNamespace(max_rounds=3)

    result = asyncio.run(wf.resume_restored_wait_state(state))

    assert called["report"] is False, "_generate_final_report must NOT be called at integration gate"
    assert called["ainvoke"] is True, "compiled_graph.ainvoke must be called to resume the checkpoint"


def test_real_wait_node_uses_semantic_gate_after_status_changes_to_running():
    """A restored gate executes the real wait node with RUNNING lifecycle state."""
    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "s"
    wf._continue_event = asyncio.Event()

    async def fake_state_sync(*args, **kwargs):
        return None

    wf._send_state_sync = fake_state_sync
    state = AnalysisState(
        id="s",
        user_id="u",
        status=TaskStatus.RUNNING,
        analysis_phase=AnalysisPhase.INTEGRATION,
        current_round=4,
        round_wait_seconds=0,
        integration_gate_candidates=[{"direction": "NEW"}],
    )

    result = asyncio.run(wf._wait_for_user_input({
        "analysis_state": state,
        "semantic_gate": "integration_selection",
    }))

    assert result["should_continue"] is True
    assert result["semantic_gate"] is None
    assert state.analysis_phase == AnalysisPhase.INNOVATION
    assert state.status == TaskStatus.RUNNING
    assert state.user_chosen_direction == "[PI_DECIDE]"
