import pytest
import asyncio
from backend.constants import WaitInputReason, LlmFailureAction
from backend.routes.lifecycle import resolve_hitl_skip_target


def test_marker_db_missing_enum_exists():
    assert WaitInputReason.MARKER_DB_MISSING.value == "marker_db_missing"


def test_resolve_skip_target_for_marker_db():
    class _S:
        wait_input_reason = "marker_db_missing"
        llm_failure_info = {"node_name": "bioinformatics_analysis", "generation": 7}
    assert resolve_hitl_skip_target(_S()) == ("bioinformatics_analysis", 7)


def _bare_workflow():
    from backend.workflow.core import SingleCellAnalysisWorkflow
    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "s1"
    wf._llm_failure_lock = asyncio.Lock()
    wf._llm_failure_events = {}
    wf._llm_failure_actions = {}
    wf._llm_failure_generations = {}
    wf._unattended_gate_retries = {}
    wf._settings_provider = None
    async def _noop(*a, **k): pass
    wf._send_message = _noop
    wf._send_state_sync = _noop
    wf._set_status = lambda s, st: setattr(s, "status", st)
    return wf


class _State:
    # AnalysisState always carries this; -1 keeps the gate in attended mode,
    # which is what these plumbing tests signal into.
    round_wait_seconds = -1
    wait_input_reason = None
    llm_failure_info = None
    status = None


@pytest.mark.asyncio
async def test_wait_for_marker_db_released_via_signal():
    """走真实 signal_llm_failure_action（验证 generation + skip 释放）。"""
    wf = _bare_workflow()
    state = _State()

    async def _drive():
        await asyncio.sleep(0.01)
        gen = wf._generation_for("bioinformatics_analysis")
        ok = wf.signal_llm_failure_action("bioinformatics_analysis", "skip", gen)
        assert ok is True

    asyncio.ensure_future(_drive())
    action = await wf._wait_for_marker_db_intervention("bioinformatics_analysis", state)
    assert action == LlmFailureAction.SKIP
    assert state.wait_input_reason is None  # 退出时已清


@pytest.mark.asyncio
async def test_marker_db_callback_auto_download_no_dialog(monkeypatch):
    """默认自动下载成功时直接返回 RETRY，不弹 HITL 窗口。"""
    wf = _bare_workflow()
    wf._current_analysis_state = _State()
    fetched = {"n": 0}
    waited = {"called": False}

    async def _wait(node, st):
        waited["called"] = True
        return LlmFailureAction.TERMINATE
    wf._wait_for_marker_db_intervention = _wait
    monkeypatch.setattr("backend.services.marker_priors.trigger_fetch",
                        lambda: fetched.__setitem__("n", fetched["n"] + 1) or True)
    action = await wf._marker_db_intervention_callback()
    assert action == LlmFailureAction.RETRY
    assert fetched["n"] == 1       # 自动下载触发了一次 fetch
    assert not waited["called"]    # 未弹窗


@pytest.mark.asyncio
async def test_marker_db_callback_retry_after_failed_download(monkeypatch):
    """自动下载失败 → 弹窗 → 用户重试，第二次成功返回 RETRY。"""
    wf = _bare_workflow()
    wf._current_analysis_state = _State()
    fetched = {"n": 0}

    # 第 1 次（自动）失败，第 2 次（用户重试）成功
    def _fetch():
        fetched["n"] += 1
        return fetched["n"] >= 2
    monkeypatch.setattr("backend.services.marker_priors.trigger_fetch", _fetch)

    async def _wait(node, st): return LlmFailureAction.RETRY
    wf._wait_for_marker_db_intervention = _wait
    action = await wf._marker_db_intervention_callback()
    assert action == LlmFailureAction.RETRY
    assert fetched["n"] == 2  # 自动 1 次 + 重试 1 次


@pytest.mark.asyncio
async def test_marker_db_callback_terminate_after_failed_download(monkeypatch):
    """自动下载失败 → 弹窗 → 用户终止，透传 TERMINATE。"""
    wf = _bare_workflow()
    wf._current_analysis_state = _State()

    monkeypatch.setattr("backend.services.marker_priors.trigger_fetch", lambda: False)

    async def _wait(node, st): return LlmFailureAction.TERMINATE
    wf._wait_for_marker_db_intervention = _wait
    action = await wf._marker_db_intervention_callback()
    assert action == LlmFailureAction.TERMINATE


@pytest.mark.asyncio
async def test_the_dialog_names_the_source_that_failed(monkeypatch):
    """The dialog asks the user to choose; it has to show what they are choosing on.

    On 2026-08-01 the prompt said only that the automatic download had failed.
    The actual fault — panglaodb.se dropping the TLS handshake, while CellMarker
    was reachable — appeared nowhere the user could see it, so "retry" and
    "skip (literature-only annotation)" looked equally arbitrary.
    """
    wf = _bare_workflow()
    captured = {}

    async def _gate(node_name, state, **kwargs):
        captured.update(kwargs)
        return LlmFailureAction.SKIP

    wf._drive_hitl_gate = _gate
    monkeypatch.setattr(
        "backend.services.marker_priors.last_fetch_errors",
        lambda: {"panglaodb.tsv": "ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING]"})

    await wf._wait_for_marker_db_intervention("bioinformatics_analysis", _State())

    error = captured["info_payload"]["error"]
    assert "panglaodb.tsv" in error
    assert "UNEXPECTED_EOF_WHILE_READING" in error
    assert "Retry the download" in error, "the offered choices must survive"


@pytest.mark.asyncio
async def test_the_dialog_says_nothing_extra_when_there_is_no_reason(monkeypatch):
    """No dangling "Last attempt:" header with nothing after it."""
    wf = _bare_workflow()
    captured = {}

    async def _gate(node_name, state, **kwargs):
        captured.update(kwargs)
        return LlmFailureAction.SKIP

    wf._drive_hitl_gate = _gate
    monkeypatch.setattr("backend.services.marker_priors.last_fetch_errors", dict)

    await wf._wait_for_marker_db_intervention("bioinformatics_analysis", _State())
    assert "Last attempt" not in captured["info_payload"]["error"]
