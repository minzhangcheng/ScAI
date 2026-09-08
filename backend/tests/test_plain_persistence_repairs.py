"""Regressions for the plain-persistence review repairs."""
import asyncio
import hashlib
import importlib
import json
import logging
import uuid
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


def test_restart_with_work_after_last_save_is_not_current(tmp_path):
    from backend.models.schemas import AnalysisState
    from backend.utils.runtime_apply import state_runtime_coherence
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    sid = str(uuid.uuid4())
    notebook = NotebookState(sid, str(tmp_path / "states"))
    notebook.update_runtime_persistence(
        "current", current_revision=5, saved_revision=5,
        commit_source="regular_save", runtime_digest="d" * 64,
    )
    notebook.update_runtime_persistence("live_dirty", current_revision=6, saved_revision=5)
    # Load the persisted authority as a new process would, after a crash before
    # the next save was attempted. The state/checkpoint may already have moved.
    reloaded = NotebookState(sid, str(tmp_path / "states"))
    state = AnalysisState(id=sid, user_id="u1", current_round=2,
                          runtime_persistence=reloaded.runtime_persistence)
    assert state_runtime_coherence(state) == "stale", state.runtime_persistence


@pytest.mark.parametrize("malformed", [False, True], ids=["digest-mismatch", "malformed-manifest"])
def test_import_refused_generation_does_not_claim_runtime(tmp_path, malformed):
    from backend.config import DATA_BASE_DIR, get_session_states
    from backend.routes.export_import import _import_tar_gz
    from backend.tests.test_export_import_hardening import _build_archive, _session_json
    from backend.utils.runtime_apply import _live_pkl
    from backend.utils.runtime_manifest import build_runtime_manifest

    sid = str(uuid.uuid4())
    payload = b"imported pkl payload"
    manifest = build_runtime_manifest(
        revision=1, state_revision=1,
        pkl_digest=hashlib.sha256(payload if malformed else b"other generation").hexdigest(),
        saved_names=["adata"], skipped_names=[], serializer_version="dill-zstd-1",
    )
    if malformed:
        manifest["saved_names"] = ["not-an-identifier"]
    archive = _build_archive(
        tmp_path / "bad-runtime.tar.gz", sid,
        {
            f"{sid}/states/{sid}_session.pkl": payload,
            f"{sid}/states/{sid}_session.runtime.json": json.dumps(manifest).encode(),
            f"{sid}/states/checkpoints.sqlite": b"SQLITE",
        },
        data_base_dir=str(DATA_BASE_DIR),
        state_payload=_session_json(sid, current_round=2, status="paused"),
    )
    state = _import_tar_gz(str(archive), user_id="importer", trust=True)
    observed = {
        "pkl": _live_pkl(state.id).exists(),
        "checkpoint": (get_session_states(state.id) / "checkpoints.sqlite").exists(),
        "status": state.runtime_restore_status,
    }
    assert observed == {"pkl": False, "checkpoint": False, "status": "unavailable"}


@pytest.mark.asyncio
async def test_executor_stops_after_local_restore_marks_unavailable(monkeypatch):
    from backend.agents.bioinformatics.executor_runtime import prepare_sandbox
    from backend.config import get_session_states, get_session_workspace
    from backend.models.schemas import AnalysisState
    from backend.services.runtime_recompute import RuntimeUnavailable
    from backend.state import sessions
    from backend.utils.sandbox import LocalJupyterSandbox
    from backend.utils.sandbox_parts import kernel

    sid = str(uuid.uuid4())
    state = AnalysisState(id=sid, user_id="u1", current_round=2)
    monkeypatch.setitem(sessions, sid, state)
    sb = LocalJupyterSandbox(session_id=sid, state_dir=str(get_session_states(sid)),
                             working_dir=str(get_session_workspace(sid)))
    sb.notebook_state.update_runtime_persistence("live_dirty", current_revision=1, saved_revision=0)
    # Only process creation/bootstrap are replaced; start(), restore(),
    # persistence projection and executor preparation are production code.
    monkeypatch.setattr(kernel, "KernelManager", lambda **kwargs: MagicMock())
    monkeypatch.setattr(kernel, "_start_kernel_manager", AsyncMock())
    sb._install_isolated_kernel_launcher = lambda manager: None
    sb._kernel_launch_environment = lambda: {}
    sb._wait_for_ready = AsyncMock()
    sb._initialize_kernel = AsyncMock()
    agent = SimpleNamespace(sandbox=sb, _notify=AsyncMock(), code_history=[])
    with pytest.raises(RuntimeUnavailable):
        await prepare_sandbox(agent, state, bindings=SimpleNamespace(
            logger=logging.getLogger(__name__), CodeExecution=object,
        ))
    assert state.runtime_restore_status == "unavailable"


@pytest.mark.asyncio
async def test_send_failure_no_longer_marks_dirty():
    from backend.utils.sandbox import LocalJupyterSandbox
    sb = object.__new__(LocalJupyterSandbox)
    sb.session_id = "review-send"
    sb._exec_lock = asyncio.Lock()
    sb._kernel_dead = False
    sb.runtime_revision = 0
    sb.km = MagicMock()
    sb.kc = MagicMock()
    sb.kc.execute.side_effect = ValueError("Invalid Signature: b'zz'")
    sb._restart_kernel = AsyncMock()
    result = await sb.execute("x = 1", skip_safety_check=True)
    assert not result.success
    assert sb.runtime_revision == 0


def _committed_pair(sid, revision):
    """A generation pair as B2-1's commit leaves it."""
    import json as _json
    import sqlite3

    from backend.config import get_session_states

    states = get_session_states(sid)
    pair = states / "generation" / str(revision)
    pair.mkdir(parents=True, exist_ok=True)
    (pair / "state.json").write_text(_json.dumps({"session_id": sid}))
    with sqlite3.connect(pair / "checkpoints.sqlite") as conn:
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT, blob TEXT)")
        conn.execute("INSERT INTO checkpoints VALUES (?, 'paired')", (sid,))
    (pair / "executor_checkpoint.json").write_text(
        _json.dumps({"step_number": 20, "runtime_revision": revision})
    )
    return states


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
async def test_a_stale_generation_with_a_pair_is_restored_not_retired(monkeypatch):
    """Spec B2-2: the committed generation is healthy after a crash.

    What made it unusable is that the state and both positions moved past it,
    which after any interruption is always true (§3.1). Retiring it there is
    what turned "at most one round or N minutes at risk" into "re-run from the
    input files".
    """
    from backend.models.schemas import AnalysisState
    from backend.routes import lifecycle
    from backend.utils.runtime_apply import _live_pkl

    sid = str(uuid.uuid4())
    states = _committed_pair(sid, 44)
    live_checkpoint = states / "checkpoints.sqlite"
    live_checkpoint.write_bytes(b"position the crash left behind")
    _live_pkl(sid).write_bytes(b"the committed generation")

    state = AnalysisState(
        id=sid, user_id="u1", current_round=2, status="paused",
        runtime_persistence=_stale_persistence(44, 61),
    )
    monkeypatch.setattr(lifecycle, "require_session_ownership", AsyncMock(return_value=state))
    monkeypatch.setattr(lifecycle, "async_save_session", AsyncMock())
    monkeypatch.setattr(lifecycle, "assert_recovery_clear", lambda *a, **k: None)

    await lifecycle.run_analysis(sid, SimpleNamespace(state=SimpleNamespace(user_id="u1")))

    assert _live_pkl(sid).exists(), "a healthy generation must not be retired"
    assert not state.runtime_recompute_pending

    # On disk, not on `state`: the graph resumes with its own AnalysisState,
    # so this is the only place the executor and the UI can both read it.
    import json as _json

    restored_record = _json.loads(
        (states / f"{sid}_state.json").read_text()
    )["runtime_persistence"]
    assert restored_record["pair_restore"]["restored_revision"] == 44

    import sqlite3

    with sqlite3.connect(live_checkpoint) as conn:
        assert conn.execute("SELECT blob FROM checkpoints").fetchone()[0] == "paired"


@pytest.mark.asyncio
async def test_a_stale_generation_without_a_pair_still_retires(monkeypatch):
    """Retirement is the fallback, not the default, and must stay reachable."""
    from backend.config import get_session_states
    from backend.models.schemas import AnalysisState
    from backend.routes import lifecycle
    from backend.utils.runtime_apply import _live_pkl

    sid = str(uuid.uuid4())
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    (states / "checkpoints.sqlite").write_bytes(b"checkpoint")
    _live_pkl(sid).write_bytes(b"generation")

    state = AnalysisState(
        id=sid, user_id="u1", current_round=2, status="paused",
        runtime_persistence=_stale_persistence(44, 61),
    )
    monkeypatch.setattr(lifecycle, "require_session_ownership", AsyncMock(return_value=state))
    monkeypatch.setattr(lifecycle, "async_save_session", AsyncMock())
    monkeypatch.setattr(lifecycle, "assert_recovery_clear", lambda *a, **k: None)

    await lifecycle.run_analysis(sid, SimpleNamespace(state=SimpleNamespace(user_id="u1")))

    assert not _live_pkl(sid).exists()
    assert state.runtime_recompute_pending


@pytest.mark.asyncio
async def test_rerun_now_retires_checkpoint_before_choosing_intent(monkeypatch):
    from backend.config import get_session_states
    from backend.models.schemas import AnalysisState
    from backend.routes import lifecycle
    from backend.utils.runtime_apply import _live_pkl
    sid = str(uuid.uuid4())
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    checkpoint = states / "checkpoints.sqlite"
    checkpoint.write_bytes(b"checkpoint")
    _live_pkl(sid).write_bytes(b"generation")
    state = AnalysisState(id=sid, user_id="u1", current_round=2, status="paused",
                          runtime_restore_status="unavailable")
    monkeypatch.setattr(lifecycle, "require_session_ownership", AsyncMock(return_value=state))
    monkeypatch.setattr(lifecycle, "async_save_session", AsyncMock())
    monkeypatch.setattr(lifecycle, "assert_recovery_clear", lambda *a, **k: None)
    await lifecycle.run_analysis(sid, SimpleNamespace(state=SimpleNamespace(user_id="u1")))
    assert not checkpoint.exists()
    assert not _live_pkl(sid).exists()
    assert state.runtime_recompute_pending
    assert state.restore_start_intent == "explicit_start"


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
    assert state.restore_start_intent == "explicit_start"
    owner.close.assert_awaited_once_with(save_runtime=False)


def test_checkpoint_identity_includes_pending_writes(tmp_path):
    import sqlite3
    from backend.utils.runtime_pairing import checkpoint_identity
    path = tmp_path / "checkpoints.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE checkpoints(thread_id, checkpoint_ns, checkpoint_id, checkpoint)")
        conn.execute("CREATE TABLE writes(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, value)")
        conn.execute("INSERT INTO checkpoints VALUES('sid', '', 'c1', X'00')")
        conn.commit()
        before = checkpoint_identity("sid", tmp_path)
        conn.execute("INSERT INTO writes VALUES('sid', '', 'c1', 'task', 0, X'01')")
        conn.commit()
        assert checkpoint_identity("sid", tmp_path) != before


@pytest.mark.asyncio
async def test_commit_publication_preserves_pairing(monkeypatch):
    from backend.models.schemas import AnalysisState
    from backend.state import sessions
    from backend.workflow.callbacks import CallbacksMixin
    from backend.workflow.messaging import MessagingMixin
    from backend.services.session_store import async_save_session
    from backend.utils.runtime_apply import state_runtime_coherence
    from backend.utils.runtime_pairing import capture_pairing

    sid = str(uuid.uuid4())
    state = AnalysisState(id=sid, user_id="u1", state_version=7, status="completed")
    monkeypatch.setitem(sessions, sid, state)
    pairing = capture_pairing(sid, state)
    persistence = {
        "committed_source": "regular_save", "committed_digest": "d" * 64,
        "committed_revision": 5, "current_revision": 5,
        "last_attempt_revision": 5, "last_attempt_status": "succeeded",
        "committed_pairing": pairing,
    }

    class Workflow(CallbacksMixin, MessagingMixin):
        pass

    workflow = Workflow()
    workflow.session_id = sid
    workflow.message_callback = None

    async def persist(updated):
        await async_save_session(sid, updated, force=True)

    workflow.state_callback = persist
    await workflow._runtime_persistence_commit_callback({
        "session_id": sid, "current_revision": 5, "saved_revision": 5,
        "runtime_persistence": persistence,
    })
    # The subject: a persistence commit must not advance state_version, or it
    # would invalidate the pairing it just recorded.
    assert state.state_version == 7
    assert state_runtime_coherence(state) == "current"

    await workflow._send_state_sync(state, "round_completed")

    # And a sync that is not a persistence commit *does* advance it ...
    assert state.state_version == 8
    # ... without making the generation stale. This line previously asserted
    # "stale", which is what made crash recovery impossible unless the process
    # happened to die parked at a gate: `round_completed`, `round_start` and a
    # user's Continue all sync without running a single cell, and stale is a
    # hard block whose only escape retires the dump. Staleness is the kernel's
    # question, and `current_revision == committed_revision` answers it.
    assert state_runtime_coherence(state) == "current"


def test_manifest_roundtrip_preserves_control_pairing(tmp_path):
    from backend.utils.runtime_manifest import build_runtime_manifest, write_runtime_manifest, read_runtime_manifest
    path = tmp_path / "runtime.json"
    record = build_runtime_manifest(
        revision=5, state_revision=42, pkl_digest="d" * 64,
        saved_names=["x"], skipped_names=[], serializer_version="dill-zstd-1",
        pairing={"state_version": 42, "checkpoint": "a" * 64},
    )
    write_runtime_manifest(path, record)
    assert read_runtime_manifest(path) == record


@pytest.mark.asyncio
async def test_cold_resume_checks_availability_before_graph(monkeypatch):
    from backend.models.schemas import AnalysisState
    from backend.workflow import SingleCellAnalysisWorkflow
    from backend.config import get_session_states
    sid = str(uuid.uuid4())
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    (states / "checkpoints.sqlite").write_bytes(b"checkpoint")
    state = AnalysisState(id=sid, user_id="u1", current_round=2)
    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = sid
    workflow.compiled_graph = MagicMock()
    workflow.compiled_graph.aget_state = AsyncMock()
    assert await workflow.resume(sid, state) is state
    assert state.runtime_restore_status == "unavailable"
    workflow.compiled_graph.aget_state.assert_not_called()


def test_red_tree_validates_nodes_before_creating_worktree():
    from pathlib import Path
    script = (Path(__file__).resolve().parents[2] / "scripts/scai-red-tree").read_text()
    assert script.index('die "name each node') < script.index("git worktree add")


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


@pytest.mark.parametrize("status", ["paused", "waiting_input"])
def test_live_continuation_requires_the_original_reachable_kernel(monkeypatch, status):
    from backend.utils.runtime_pairing import is_live_runtime_continuation_safe
    state, owner = _live_paused_owner(monkeypatch, status)
    assert is_live_runtime_continuation_safe(state.id, state)
    owner.sandbox.kernel_epoch = 2
    assert not is_live_runtime_continuation_safe(state.id, state)
    owner.sandbox.kernel_epoch = 1
    owner.sandbox._session_restore_failed = True
    assert not is_live_runtime_continuation_safe(state.id, state)
    owner.sandbox._session_restore_failed = False
    owner.sandbox.km.is_alive = lambda: False
    assert not is_live_runtime_continuation_safe(state.id, state)
    owner.sandbox.km.is_alive = lambda: True
    owner._current_graph_state = None
    assert not is_live_runtime_continuation_safe(state.id, state)


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
async def test_cold_resume_refusal_persists_rerun_projection(monkeypatch):
    from backend.routes import lifecycle
    state, owner = _live_paused_owner(monkeypatch)
    owner.sandbox.kernel_epoch += 1
    monkeypatch.setattr(lifecycle, "require_session_ownership", AsyncMock(return_value=state))
    monkeypatch.setattr(lifecycle, "_update_session_cache", AsyncMock())
    save = AsyncMock()
    monkeypatch.setattr(lifecycle, "async_save_session", save)
    response = await lifecycle.resume_analysis(
        state.id, SimpleNamespace(state=SimpleNamespace(user_id="u1")),
    )
    assert response.data["runtime_unavailable"] is True
    assert response.data["runtime_restore_status"] == "unavailable"
    assert "Re-run" in response.data["runtime_restore_warning"]
    assert state.runtime_restore_status == "unavailable"
    save.assert_awaited_once_with(state.id, state, force=True)
    owner.signal_resume.assert_not_called()


@pytest.mark.asyncio
async def test_a_start_time_unavailable_verdict_reaches_the_graphs_own_state():
    """The mark and the guard must not read two different objects.

    `_mark_runtime_unavailable_if_work_exists` runs inside `sandbox.start()`
    and can only reach the cached AnalysisState in `backend.state.sessions`.
    The graph carries its own -- `_restore_entry` deep-copies one -- and the two
    coincide only from the next state sync onward. Reading the graph's object
    alone let the run continue over a namespace that does not exist.
    """
    from backend.agents.bioinformatics import executor_runtime
    from backend.models.schemas import AnalysisState
    from backend.services.runtime_recompute import RuntimeUnavailable
    from backend.state import sessions as live_sessions

    session_id = f"identity-probe-{uuid.uuid4().hex[:8]}"
    cached = AnalysisState(id=session_id, user_id="u")
    cached.runtime_restore_status = "unavailable"
    graph_state = AnalysisState(id=session_id, user_id="u")
    assert graph_state is not cached
    assert graph_state.runtime_restore_status == "complete"

    agent = MagicMock()
    agent.sandbox.km = object()
    agent.sandbox.session_id = session_id
    agent.sandbox.sync_runtime_persistence_state.return_value = {}
    agent._notify = AsyncMock()

    live_sessions[session_id] = cached
    try:
        with pytest.raises(RuntimeUnavailable):
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
async def test_an_authorized_recompute_still_runs_on_the_empty_namespace():
    """The re-run the user asked for is the one way past that guard."""
    from backend.agents.bioinformatics import executor_runtime
    from backend.models.schemas import AnalysisState
    from backend.state import sessions as live_sessions

    session_id = f"identity-probe-{uuid.uuid4().hex[:8]}"
    cached = AnalysisState(id=session_id, user_id="u")
    cached.runtime_restore_status = "unavailable"
    graph_state = AnalysisState(id=session_id, user_id="u")
    graph_state.runtime_recompute_pending = True

    agent = MagicMock()
    agent.sandbox.km = object()
    agent.sandbox.session_id = session_id
    agent.sandbox.sync_runtime_persistence_state.return_value = {}
    agent._notify = AsyncMock()

    live_sessions[session_id] = cached
    try:
        await executor_runtime.prepare_sandbox(
            agent, graph_state,
            bindings=SimpleNamespace(
                logger=logging.getLogger(__name__),
                CodeExecution=MagicMock(),
            ),
        )
    finally:
        live_sessions.pop(session_id, None)


def _mutating_sandbox(tmp_path, *, success: bool):
    """A sandbox whose next execution mutated the namespace and reached the kernel."""
    from backend.utils.sandbox import CodeExecutionResult, LocalJupyterSandbox

    sb = LocalJupyterSandbox(
        session_id=f"dirty-mark-{uuid.uuid4().hex[:8]}",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    sb.runtime_revision = 4
    sb.saved_runtime_revision = 4
    sb._execute_unlocked = AsyncMock(
        return_value=CodeExecutionResult(
            success=success, error=None if success else "boom"
        )
    )
    sb._last_execution_reached_kernel = True
    return sb


def _assert_record_moved_with_the_counter(sb):
    assert sb.runtime_revision == 5
    record = sb.notebook_state.runtime_persistence
    assert int(record.get("current_revision") or 0) == 5, record


@pytest.mark.asyncio
async def test_a_failed_mutating_cell_moves_the_persisted_revision(tmp_path):
    """The kernel does not roll a cell back.

    A statement that mutated `adata` before raising leaves it mutated, and the
    live counter says so -- but while the persisted record still read
    current == committed, a crash there came back as "current": the dump would
    have been restored, and reported coherent, without that mutation.
    """
    sb = _mutating_sandbox(tmp_path, success=False)

    await sb.execute("adata.X = adata.X * 2", skip_safety_check=True)

    _assert_record_moved_with_the_counter(sb)


@pytest.mark.asyncio
async def test_a_successful_cell_outside_the_notebook_moves_it_too(tmp_path):
    """`records_history=False` is about the notebook, not about effect.

    Several mutating internal cells stay out of it -- `adata_val = adata.copy()`
    among them -- so gating the record on it lost real, successful work.
    """
    sb = _mutating_sandbox(tmp_path, success=True)

    await sb.execute(
        "adata_val = adata.copy()", records_history=False, skip_safety_check=True,
    )

    _assert_record_moved_with_the_counter(sb)


@pytest.mark.asyncio
async def test_a_read_only_probe_still_leaves_the_record_alone(tmp_path):
    """The counterpart: nothing changed, so nothing is claimed."""
    from backend.utils.sandbox import CodeExecutionResult, LocalJupyterSandbox

    sb = LocalJupyterSandbox(
        session_id=f"probe-{uuid.uuid4().hex[:8]}",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    sb.runtime_revision = 4
    sb.saved_runtime_revision = 4
    sb._execute_unlocked = AsyncMock(return_value=CodeExecutionResult(success=False, error="boom"))
    sb._last_execution_reached_kernel = True

    await sb.execute(
        "print(adata.shape)",
        records_history=False,
        skip_safety_check=True,
        mutates_namespace=False,
    )

    assert sb.runtime_revision == 4
    assert not sb.notebook_state.runtime_persistence.get("current_revision")


def test_a_first_round_that_never_saved_still_has_something_to_lose():
    """Round 1, cells have run, the save floor has not come round yet.

    `current_round > 1`, `round_results` and a committed generation all read
    False here, and that session has the most to lose, not the least: none of
    its work is on disk. Without this the guard let the graph carry on over an
    empty kernel.
    """
    from types import SimpleNamespace

    from backend.utils.runtime_pairing import has_completed_runtime_work

    fresh = SimpleNamespace(current_round=1, round_results=[], runtime_persistence={})
    assert has_completed_runtime_work(fresh) is False

    worked = SimpleNamespace(
        current_round=1, round_results=[],
        runtime_persistence={"current_revision": 3},
    )
    assert has_completed_runtime_work(worked) is True

    degraded = SimpleNamespace(
        current_round=1, round_results=[],
        runtime_persistence={"current_revision": 0, "degraded_from_revision": 2},
    )
    assert has_completed_runtime_work(degraded) is True
