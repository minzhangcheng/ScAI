"""Bounded HITL and durable fallback for final detailed-report content."""
import asyncio
import uuid
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from backend.constants import LlmFailureAction
from backend.config import ensure_session_dirs
from backend.models.schemas import AnalysisState, AnalysisStep, TaskStatus
from backend.services.session_store import ensure_session_loaded, save_session_to_disk
from backend.state import sessions
from backend.utils.llm_client import UpstreamExhaustedError
from backend.workflow import SingleCellAnalysisWorkflow
from backend.workflow.helpers import WorkflowTerminatedError
from backend.workflow.reporting import ReportingMixin
from backend.workflow.reporting_parts import final_report as final_report_module


def _report(status: str = "completed") -> dict:
    return {
        "id": "report-1",
        "title": "Report",
        "sections": [],
        "figures": [],
        "tables": [],
        "status": status,
        "current_step": "Complete",
    }


def _state(
    session_id: str = "report-session", *, round_wait_seconds: int = -1
) -> dict:
    analysis_state = AnalysisState(
        id=session_id,
        user_id="user",
        name="Report session",
        initial_instruction="Analyze",
    )
    analysis_state.current_round = 1
    analysis_state.round_wait_seconds = round_wait_seconds
    analysis_state.analysis_log.append(
        AnalysisStep(step_number=1, step_type="execute_code", code="x = 1")
    )
    return {"analysis_state": analysis_state}


class _Harness(ReportingMixin):
    def __init__(self, supervisor, session_id="report-session"):
        self.session_id = session_id
        self.supervisor_agent = supervisor
        self._settings_provider = MagicMock()
        self._settings_provider.get_llm_client.return_value = MagicMock()
        self._settings_provider.get_model.side_effect = lambda name: f"{name}-model"
        self._settings_provider.clear_client_cache = MagicMock()
        self._vision_settings = {}
        self._send_message = AsyncMock()
        self.persisted = []
        self.transitions = []
        self._llm_failure_lock = asyncio.Lock()
        self._llm_failure_events = {}
        self._llm_failure_actions = {}
        self._llm_failure_generations = {}
        self._unattended_gate_retries = {}

    def ensure_agent_ready(self, _name):
        return True

    @staticmethod
    def _set_status(state, status):
        state.status = status

    async def _send_state_sync(self, state, change_type, **kwargs):
        self.transitions.append(change_type)
        self.persisted.append(
            AnalysisState.model_validate(state.model_dump(mode="json"))
        )


def _patch_report_paths(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    reports = tmp_path / "reports"
    workspace.mkdir()
    monkeypatch.setattr(
        final_report_module, "get_session_workspace", lambda _session_id: workspace
    )
    monkeypatch.setattr(
        final_report_module,
        "resolve_report_output_dir",
        lambda _session_id, _branch_id: reports,
    )


def _bind_real_intervention(harness):
    harness._next_generation = (
        SingleCellAnalysisWorkflow._next_generation.__get__(harness, type(harness))
    )
    harness._wait_for_llm_intervention = (
        SingleCellAnalysisWorkflow._wait_for_llm_intervention.__get__(
            harness, type(harness)
        )
    )
    harness._drive_hitl_gate = (
        SingleCellAnalysisWorkflow._drive_hitl_gate.__get__(harness, type(harness))
    )


def _upstream_error():
    return UpstreamExhaustedError(
        "report upstream exhausted",
        agent_name="report_generator",
        model="pi-model",
        llm_server="https://llm.invalid/v1",
    )


@pytest.mark.asyncio
async def test_report_gate_waits_after_two_retries_until_user_skips(monkeypatch, tmp_path):
    from backend.tests.test_llm_failure_workflow import _no_backoff, _park_probe

    _patch_report_paths(monkeypatch, tmp_path)
    slept = _no_backoff(monkeypatch)
    parked = _park_probe(monkeypatch)
    supervisor = MagicMock()
    supervisor.generate_detailed_report = AsyncMock(
        side_effect=[_upstream_error(), _upstream_error(), _upstream_error(), _report("partial")]
    )
    harness = _Harness(supervisor)
    _bind_real_intervention(harness)
    state = _state(round_wait_seconds=0)
    task = asyncio.create_task(harness._generate_detailed_report(state))
    try:
        await asyncio.wait_for(parked.wait(), 2)
        assert not task.done()
        assert slept == [0, 1]
        assert supervisor.generate_detailed_report.await_count == 3
        assert state["analysis_state"].status == TaskStatus.WAITING_INPUT
        assert state["analysis_state"].detailed_report_fallback_pending is False
        assert harness.persisted[-1].llm_failure_info["wait_seconds"] == -1
        harness._llm_failure_actions["generate_detailed_report"] = "skip"
        harness._llm_failure_events["generate_detailed_report"].set()
        result = await asyncio.wait_for(task, 2)
        calls = supervisor.generate_detailed_report.await_args_list
        assert [call.kwargs["force_fallback"] for call in calls] == [False, False, False, True]
        assert result["analysis_state"].status == TaskStatus.COMPLETED
        assert result["analysis_state"].detailed_report_fallback_pending is False
        assert any(
            saved.status == TaskStatus.GENERATING_REPORT
            and saved.detailed_report_fallback_pending
            for saved in harness.persisted
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_report_retry_disarms_fallback_and_retries_upstream(
    monkeypatch, tmp_path
):
    _patch_report_paths(monkeypatch, tmp_path)
    supervisor = MagicMock()
    supervisor.generate_detailed_report = AsyncMock(
        side_effect=[_upstream_error(), _report()]
    )
    harness = _Harness(supervisor)
    harness._wait_for_llm_intervention = AsyncMock(
        return_value=LlmFailureAction.RETRY
    )

    result = await harness._generate_detailed_report(_state())

    calls = supervisor.generate_detailed_report.await_args_list
    assert [call.kwargs["force_fallback"] for call in calls] == [False, False]
    assert result["analysis_state"].status == TaskStatus.COMPLETED
    assert result["analysis_state"].detailed_report_fallback_pending is False
    harness._wait_for_llm_intervention.assert_awaited_once_with(
        ANY,
        "generate_detailed_report",
        ANY,
    )


@pytest.mark.asyncio
async def test_report_skip_uses_fallback_instead_of_dropping_the_report(
    monkeypatch, tmp_path
):
    _patch_report_paths(monkeypatch, tmp_path)
    supervisor = MagicMock()
    supervisor.generate_detailed_report = AsyncMock(
        side_effect=[_upstream_error(), _report("partial")]
    )
    harness = _Harness(supervisor)
    harness._wait_for_llm_intervention = AsyncMock(
        return_value=LlmFailureAction.SKIP
    )

    result = await harness._generate_detailed_report(_state())

    calls = supervisor.generate_detailed_report.await_args_list
    assert [call.kwargs["force_fallback"] for call in calls] == [False, True]
    assert result["analysis_state"].status == TaskStatus.COMPLETED
    assert "detailed_report_fallback" in harness.transitions


@pytest.mark.asyncio
async def test_report_skip_without_supervisor_uses_generator_fallback_once(
    monkeypatch, tmp_path
):
    _patch_report_paths(monkeypatch, tmp_path)
    generator = MagicMock()
    generator.generate_report = AsyncMock(
        side_effect=[_upstream_error(), _report("partial")]
    )
    generator_factory = MagicMock(return_value=generator)
    monkeypatch.setattr(
        final_report_module, "DetailedReportGenerator", generator_factory
    )
    harness = _Harness(None)
    harness._wait_for_llm_intervention = AsyncMock(
        return_value=LlmFailureAction.SKIP
    )

    result = await harness._generate_detailed_report(_state())

    calls = generator.generate_report.await_args_list
    assert [call.kwargs["force_fallback"] for call in calls] == [False, True]
    harness._wait_for_llm_intervention.assert_awaited_once()
    assert result["analysis_state"].status == TaskStatus.COMPLETED
    assert result["analysis_state"].detailed_report_fallback_pending is False


@pytest.mark.asyncio
async def test_report_terminate_keeps_terminate_contract(monkeypatch, tmp_path):
    _patch_report_paths(monkeypatch, tmp_path)
    supervisor = MagicMock()
    supervisor.generate_detailed_report = AsyncMock(side_effect=_upstream_error())
    harness = _Harness(supervisor)
    harness._wait_for_llm_intervention = AsyncMock(
        return_value=LlmFailureAction.TERMINATE
    )
    state = _state()

    with pytest.raises(WorkflowTerminatedError):
        await harness._generate_detailed_report(state)

    supervisor.generate_detailed_report.assert_awaited_once()
    assert state["analysis_state"].detailed_report_fallback_pending is False
    assert state["analysis_state"].status == TaskStatus.STOPPED
    assert harness.persisted[-1].status == TaskStatus.STOPPED
    assert harness.persisted[-1].detailed_report_fallback_pending is False


@pytest.mark.asyncio
async def test_persisted_fallback_marker_survives_restart_and_bypasses_llm(
    monkeypatch, tmp_path
):
    _patch_report_paths(monkeypatch, tmp_path)
    session_id = f"report-restart-{uuid.uuid4().hex}"
    before_restart = _state(session_id)["analysis_state"]
    before_restart.status = TaskStatus.WAITING_INPUT
    before_restart.wait_input_reason = "llm_failure"
    before_restart.llm_failure_info = {
        "node_name": "generate_detailed_report",
        "generation": 1,
    }
    before_restart.detailed_report_fallback_pending = True
    ensure_session_dirs(session_id)
    save_session_to_disk(session_id, before_restart)
    sessions.pop(session_id, None)

    # Process restart normalization retires the now-unanswerable live gate but
    # must retain the durable report-specific fallback handoff.
    restored = await ensure_session_loaded(session_id)
    assert restored is not None
    assert restored.status == TaskStatus.PAUSED
    assert restored.wait_input_reason is None
    assert restored.llm_failure_info is None
    assert restored.detailed_report_fallback_pending is True

    supervisor = MagicMock()
    supervisor.generate_detailed_report = AsyncMock(return_value=_report("partial"))
    harness = _Harness(supervisor, session_id=session_id)

    result = await harness._generate_detailed_report(
        {"analysis_state": restored}
    )

    call = supervisor.generate_detailed_report.await_args
    assert call.kwargs["force_fallback"] is True
    assert harness.transitions[0] == "detailed_report_fallback_resume"
    assert harness.persisted[0].status == TaskStatus.GENERATING_REPORT
    assert harness.persisted[0].detailed_report_fallback_pending is True
    assert result["analysis_state"].status == TaskStatus.COMPLETED
    assert result["analysis_state"].detailed_report_fallback_pending is False


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["summary", "integrated", "graphic"])
async def test_report_stages_propagate_user_terminate(stage):
    harness = _Harness(MagicMock())
    harness._wait_for_llm_intervention = AsyncMock(return_value=LlmFailureAction.TERMINATE)
    harness.pi_agent = MagicMock()
    harness.pi_agent.generate_final_summary = AsyncMock(return_value="Summary")
    harness.pi_agent.generate_integrated_report = AsyncMock(side_effect=_upstream_error())
    harness._prepare_brief_figures = AsyncMock(return_value=([], [], []))
    state = _state()

    if stage == "summary":
        harness.pi_agent.generate_final_summary.side_effect = _upstream_error()
    if stage == "graphic":
        harness.ensure_agent_ready = lambda _: False
        harness._generate_text_graphic_abstract = AsyncMock(side_effect=_upstream_error())
        call = harness._generate_graphic_abstract
    else:
        call = harness._generate_final_report

    with pytest.raises(WorkflowTerminatedError):
        await call(state)

    assert state["analysis_state"].status == TaskStatus.STOPPED
    assert harness.persisted[-1].status == TaskStatus.STOPPED
    assert not any(call.args[0] == "completed" for call in harness._send_message.await_args_list)
    if stage == "summary":
        harness.pi_agent.generate_integrated_report.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
async def test_local_report_failure_preserves_retry_target(monkeypatch, tmp_path, fallback):
    _patch_report_paths(monkeypatch, tmp_path)
    supervisor = MagicMock()
    supervisor.generate_detailed_report = AsyncMock(side_effect=OSError("disk full"))
    harness = _Harness(supervisor)
    state = _state()
    analysis = state["analysis_state"]
    analysis.detailed_report_fallback_pending = fallback
    analysis.sandbox_state = {"retained": "runtime metadata"}

    failed = await harness._generate_detailed_report(state)

    assert analysis.status == (TaskStatus.PAUSED if fallback else TaskStatus.FAILED)
    assert failed["failed_node"] == "generate_detailed_report"
    assert harness.persisted[-1].sandbox_state["failed_node"] == "generate_detailed_report"
    assert analysis.detailed_report_fallback_pending is fallback
    assert not any(call.args[0] == "completed" for call in harness._send_message.await_args_list)

    supervisor.generate_detailed_report.side_effect = None
    supervisor.generate_detailed_report.return_value = _report()
    recovered = await harness._generate_detailed_report(failed)

    assert recovered["failed_node"] is None
    assert analysis.status == TaskStatus.COMPLETED
    assert analysis.sandbox_state == {"retained": "runtime metadata"}
    assert supervisor.generate_detailed_report.await_args.kwargs["force_fallback"] is fallback
