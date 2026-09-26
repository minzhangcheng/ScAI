"""Regressions for the plain-persistence review repairs."""
import asyncio
import importlib
import json
import logging
import uuid
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    from backend.tests.split_facade import reload_split
    import backend.config._config as cfg
    import backend.config as config
    import backend.services.session_store as store
    monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path))
    importlib.reload(cfg)
    importlib.reload(config)
    reload_split("backend.routes.export_import")
    importlib.reload(store)


def _stale_persistence(committed, current):
    return {
        "status": "live_dirty",
        "current_revision": current,
        "saved_revision": committed,
        "committed_revision": committed,
        "committed_source": "regular_save",
        "committed_digest": "a" * 64,
        "last_attempt_status": "succeeded",
        "last_attempt_revision": committed,
    }


@pytest.mark.asyncio
async def test_pair_start_restores_progress_before_publication_and_graph_resume(monkeypatch):
    import aiosqlite
    from typing import TypedDict
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, START, StateGraph
    from backend.config import get_session_states
    from backend.models.schemas import AnalysisPlan, AnalysisState, TaskStatus
    from backend.routes import lifecycle
    from backend.routes.websockets import _needs_report_only
    from backend.services.session_store import load_session_from_disk
    from backend.state import sessions
    from backend.utils import runtime_apply
    from backend.utils.checkpoint import DurableJsonPlusSerializer

    sid = str(uuid.uuid4())
    states = get_session_states(sid)
    pair = states / "generation" / "44"
    pair.mkdir(parents=True)
    record = _stale_persistence(44, 44)
    record["status"] = "current"
    (pair / "state.json").write_text(json.dumps({
        "session_id": sid, "runtime_persistence": record,
    }))
    (states / f"{sid}_state.json").write_text(json.dumps({
        "session_id": sid, "runtime_persistence": _stale_persistence(44, 61),
    }))
    runtime_apply._live_pkl(sid).write_bytes(b"committed")
    old = AnalysisState(
        id=sid, user_id="u1", current_round=1,
        current_plan=AnalysisPlan(id="paired-plan", phase="exploration"),
        runtime_persistence=record,
    )
    schema = TypedDict("PairGraphState", {"analysis_state": AnalysisState})
    builder = StateGraph(schema)
    seen = []

    def work(value):
        seen.append((value["analysis_state"].current_round,
                     value["analysis_state"].current_plan.id))
        return value

    builder.add_node("work", work)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    config = {"configurable": {"thread_id": sid}}
    async with aiosqlite.connect(pair / "checkpoints.sqlite") as conn:
        saver = AsyncSqliteSaver(conn, serde=DurableJsonPlusSerializer())
        graph = builder.compile(checkpointer=saver, interrupt_before=["work"])
        await graph.ainvoke({"analysis_state": old}, config)

    current = old.model_copy(deep=True)
    current.current_round = 3
    current.current_plan.id = "newer-plan"
    current.final_report_requested = True
    current.status = TaskStatus.PAUSED
    current.custom_models = {"pi": "current-model"}
    current.agent_server_configs = {"pi": {"api_key": "current-key"}}
    current.runtime_persistence = _stale_persistence(44, 61)
    monkeypatch.setitem(sessions, sid, current)
    monkeypatch.setattr(lifecycle, "require_session_ownership", AsyncMock(return_value=current))
    monkeypatch.setattr(lifecycle, "assert_recovery_clear", lambda *a, **k: None)
    original_restore = runtime_apply.restore_generation_pair

    def restore_after_progress_save(*args, **kwargs):
        persisted = load_session_from_disk(sid)
        assert persisted.current_round == 1
        assert persisted.current_plan.id == "paired-plan"
        assert persisted.runtime_persistence["current_revision"] == 61
        return original_restore(*args, **kwargs)

    monkeypatch.setattr(runtime_apply, "restore_generation_pair", restore_after_progress_save)
    await lifecycle.run_analysis(sid, SimpleNamespace(state=SimpleNamespace(user_id="u1")))
    assert sessions[sid] is current
    assert current.current_round == 1
    assert current.current_plan.id == "paired-plan"
    assert current.custom_models == {"pi": "current-model"}
    assert current.agent_server_configs["pi"]["api_key"] == "current-key"
    assert not _needs_report_only(current, 3, False)
    assert load_session_from_disk(sid).current_plan.id == "paired-plan"
    # Exercise the same injection followed by continuation that resume uses.
    async with aiosqlite.connect(states / "checkpoints.sqlite") as conn:
        saver = AsyncSqliteSaver(conn, serde=DurableJsonPlusSerializer())
        graph = builder.compile(checkpointer=saver, interrupt_before=["work"])
        await graph.aupdate_state(config, {"analysis_state": current})
        await graph.ainvoke(None, config)
    assert seen == [(1, "paired-plan")]


@pytest.mark.asyncio
@pytest.mark.parametrize("completed", [False, True], ids=["run", "completed-resume"])
async def test_rerun_quiesces_existing_workflow_before_retiring_checkpoint(monkeypatch, completed):
    from backend.config import get_session_states
    from backend.models.schemas import AnalysisState
    from backend.routes import lifecycle
    from backend.state import workflows
    from backend.utils.runtime_apply import _live_pkl
    sid = str(uuid.uuid4())
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    checkpoint = states / "checkpoints.sqlite"
    checkpoint.write_bytes(b"checkpoint owned by paused workflow")
    _live_pkl(sid).write_bytes(b"runtime")
    state = AnalysisState(id=sid, user_id="u1", current_round=2,
                          status="completed" if completed else "paused",
                          runtime_restore_status="unavailable")
    async def close(**kwargs):
        assert checkpoint.exists()
        assert _live_pkl(sid).exists()
        return True

    owner = SimpleNamespace(close=AsyncMock(side_effect=close))
    monkeypatch.setitem(workflows, sid, owner)
    monkeypatch.setattr(lifecycle, "require_session_ownership", AsyncMock(return_value=state))
    monkeypatch.setattr(lifecycle, "async_save_session", AsyncMock())
    monkeypatch.setattr(lifecycle, "assert_recovery_clear", lambda *a, **k: None)
    action = lifecycle.resume_analysis if completed else lifecycle.run_analysis
    await action(sid, SimpleNamespace(state=SimpleNamespace(user_id="u1")))
    assert not checkpoint.exists()
    assert not _live_pkl(sid).exists()
    assert state.runtime_recompute_pending
    assert state.restore_start_intent == (
        "completed_reanalysis" if completed else "explicit_start"
    )
    owner.close.assert_awaited_once_with(save_runtime=False)


def _live_paused_owner(monkeypatch, status="paused"):
    from backend.models.schemas import AnalysisState
    from backend.state import sessions, workflows
    sid = str(uuid.uuid4())
    state = AnalysisState(id=sid, user_id="u1", current_round=2, status=status,
                          runtime_persistence={"current_revision": 2, "saved_revision": 1,
                                               "last_attempt_status": "failed",
                                               "committed_revision": 1,
                                               "committed_source": "regular_save",
                                               "committed_digest": "d" * 64})
    sandbox = SimpleNamespace(
        _initialized=True, _kernel_dead=False, _session_restore_failed=False,
        _restarting=False, km=SimpleNamespace(is_alive=lambda: True), kc=object(),
        kernel_epoch=1, _expected_lifecycle_generation=state.lifecycle_generation,
        notebook_state=SimpleNamespace(lifecycle_generation=state.lifecycle_generation),
    )
    owner = SimpleNamespace(
        session_id=sid, sandbox=sandbox, _continuation_kernel_epoch=1,
        _current_analysis_state=state, _current_graph_state={"analysis_state": state},
        _pause_event=asyncio.Event(), _continue_event=asyncio.Event(),
        signal_resume=MagicMock(), signal_continue=MagicMock(),
    )
    if status != "paused":
        owner._pause_event.set()
    monkeypatch.setitem(sessions, sid, state)
    monkeypatch.setitem(workflows, sid, owner)
    return state, owner


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["paused", "waiting_input"])
async def test_http_resume_uses_intact_live_kernel_after_failed_save(monkeypatch, status):
    from backend.routes import lifecycle
    state, owner = _live_paused_owner(monkeypatch, status)
    monkeypatch.setattr(lifecycle, "require_session_ownership", AsyncMock(return_value=state))
    monkeypatch.setattr(lifecycle, "_update_session_cache", AsyncMock())
    monkeypatch.setattr(lifecycle, "async_save_session", AsyncMock())
    response = await lifecycle.resume_analysis(
        state.id, SimpleNamespace(state=SimpleNamespace(user_id="u1")),
    )
    assert response.success
    assert not (response.data or {}).get("runtime_unavailable")
    assert state.runtime_restore_status == "complete"
    (owner.signal_resume if status == "paused" else owner.signal_continue).assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("authorized", [False, True], ids=["refuse", "recompute"])
async def test_unavailable_cached_runtime_requires_authorized_recompute(authorized):
    """A stale graph copy cannot hide the cached runtime restoration failure."""
    from backend.agents.bioinformatics import executor_runtime
    from backend.models.schemas import AnalysisState
    from backend.state import sessions as live_sessions
    from backend.services.runtime_recompute import RuntimeUnavailable

    session_id = f"identity-probe-{uuid.uuid4().hex[:8]}"
    cached = AnalysisState(id=session_id, user_id="u")
    cached.runtime_restore_status = "unavailable"
    graph_state = AnalysisState(id=session_id, user_id="u")
    graph_state.runtime_recompute_pending = authorized
    assert graph_state is not cached
    assert graph_state.runtime_restore_status == "complete"

    agent = MagicMock()
    agent.sandbox.km = object()
    agent.sandbox.session_id = session_id
    agent.sandbox.sync_runtime_persistence_state.return_value = {}
    agent._notify = AsyncMock()

    live_sessions[session_id] = cached
    try:
        expectation = nullcontext() if authorized else pytest.raises(RuntimeUnavailable)
        with expectation:
            await executor_runtime.prepare_sandbox(
                agent, graph_state,
                bindings=SimpleNamespace(
                    logger=logging.getLogger(__name__),
                    CodeExecution=MagicMock(),
                ),
            )
    finally:
        live_sessions.pop(session_id, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "success, records_history, mutates_namespace",
    [(False, True, True), (True, False, True), (False, False, False)],
    ids=["failed-mutation", "unrecorded-mutation", "readonly-probe"],
)
async def test_execution_effect_keeps_persisted_revision_in_sync(
    tmp_path, success, records_history, mutates_namespace,
):
    from backend.utils.sandbox import CodeExecutionResult, LocalJupyterSandbox

    sandbox = LocalJupyterSandbox(
        session_id=f"dirty-mark-{uuid.uuid4().hex[:8]}",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    sandbox.runtime_revision = sandbox.saved_runtime_revision = 4
    sandbox._execute_unlocked = AsyncMock(return_value=CodeExecutionResult(
        success=success, error=None if success else "boom",
    ))
    sandbox._last_execution_reached_kernel = True
    await sandbox.execute(
        "adata_val = adata.copy()" if mutates_namespace else "print(adata.shape)",
        records_history=records_history, mutates_namespace=mutates_namespace,
        skip_safety_check=True,
    )
    record = sandbox.notebook_state.runtime_persistence
    assert sandbox.runtime_revision == (5 if mutates_namespace else 4)
    assert int(record.get("current_revision") or 0) == (5 if mutates_namespace else 0)
