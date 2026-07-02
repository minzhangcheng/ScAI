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
    wf._settings_provider = None
    async def _noop(*a, **k): pass
    wf._send_message = _noop
    wf._send_state_sync = _noop
    wf._set_status = lambda s, st: setattr(s, "status", st)
    return wf


class _State:
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
