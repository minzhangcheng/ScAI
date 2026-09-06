"""execute() 与 save_session() 必须经同一把 asyncio.Lock 串行化，
防止 pause() 在 bioinformatics 在途 execute 期间并发读 kernel ZMQ socket。"""
import asyncio
import threading
import pytest

from backend.utils.sandbox import LocalJupyterSandbox
from backend.tests.split_facade import setattr_facade


def _bare_sandbox() -> LocalJupyterSandbox:
    sb = object.__new__(LocalJupyterSandbox)
    sb._exec_lock = asyncio.Lock()
    return sb


@pytest.mark.asyncio
async def test_execute_and_save_share_one_lock():
    sb = _bare_sandbox()
    order = []

    async def fake_long_exec():
        async with sb._exec_lock:
            order.append("exec-start")
            await asyncio.sleep(0.05)
            order.append("exec-end")

    async def fake_save():
        async with sb._exec_lock:
            order.append("save")

    await asyncio.gather(fake_long_exec(), fake_save())
    # save 必须等待 exec 完整结束后才进入临界区
    assert order == ["exec-start", "exec-end", "save"]


@pytest.mark.asyncio
async def test_save_captures_revision_after_waiting_for_execution_lock(
    tmp_path, monkeypatch,
):
    from pathlib import Path
    from backend.utils.sandbox import CodeExecutionResult

    sb = LocalJupyterSandbox(
        session_id="revision-after-lock",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    sb.runtime_budgets = {}
    sb.runtime_revision = 1

    async def fake_save_code(_code, **_kwargs):
        Path(str(sb.notebook_state.session_path) + ".tmp").write_bytes(b"generation")
        return CodeExecutionResult(
            success=True,
            output='Session saved: 1 variables\n__SCAI_SAVED_NAMES__:["adata"]\n__SCAI_EXCLUDED_KIND__:[]\n__SCAI_EXCLUDED_VOLATILE__:[]',
        )

    monkeypatch.setattr(sb, "_execute_unlocked", fake_save_code)
    await sb._exec_lock.acquire()
    task = asyncio.create_task(sb.save_session())
    await asyncio.sleep(0)
    sb.runtime_revision = 2
    assert not task.done()
    sb._exec_lock.release()

    assert await task is True
    assert sb.saved_runtime_revision == 2
    assert sb.notebook_state.runtime_persistence["committed_revision"] == 2


@pytest.mark.asyncio
async def test_save_fast_path_waits_for_inflight_execution(tmp_path, monkeypatch):
    from pathlib import Path
    from backend.utils.sandbox import CodeExecutionResult

    sb = LocalJupyterSandbox(
        session_id="fast-path-after-lock",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    sb.runtime_budgets = {}
    sb.runtime_revision = 1
    sb.saved_runtime_revision = 1
    live = sb.notebook_state.session_path
    live.write_bytes(b"old-generation")
    monkeypatch.setattr(sb.notebook_state, "has_session_dump", lambda: True)
    sb.notebook_state.runtime_persistence["commit_source"] = "regular_save"

    async def fake_save_code(_code, **_kwargs):
        Path(str(live) + ".tmp").write_bytes(b"new-generation")
        return CodeExecutionResult(
            success=True,
            output='Session saved: 1 variables\n__SCAI_SAVED_NAMES__:["adata"]\n__SCAI_EXCLUDED_KIND__:[]\n__SCAI_EXCLUDED_VOLATILE__:[]',
        )

    monkeypatch.setattr(sb, "_execute_unlocked", fake_save_code)
    await sb._exec_lock.acquire()
    task = asyncio.create_task(sb.save_session())
    await asyncio.sleep(0)
    assert not task.done()

    sb.runtime_revision = 2
    sb._exec_lock.release()

    assert await task is True
    assert sb.saved_runtime_revision == 2
    assert live.read_bytes() == b"new-generation"


@pytest.mark.asyncio
async def test_restore_refusal_waits_for_inflight_execution(tmp_path, monkeypatch):
    from pathlib import Path
    from backend.utils.sandbox import CodeExecutionResult

    sb = LocalJupyterSandbox(
        session_id="restore-refusal-after-lock",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    sb.runtime_budgets = {}
    sb.runtime_revision = 1
    sb.saved_runtime_revision = 1
    sb._session_restore_failed = True
    sb._restore_failed_at_revision = 1

    async def fake_rebuilt(_self):
        return {"adata"}

    async def fake_save_code(_code, **_kwargs):
        Path(str(sb.notebook_state.session_path) + ".tmp").write_bytes(b"rebuilt")
        return CodeExecutionResult(
            success=True,
            output='Session saved: 1 variables\n__SCAI_SAVED_NAMES__:["adata"]\n__SCAI_EXCLUDED_KIND__:[]\n__SCAI_EXCLUDED_VOLATILE__:[]',
        )

    monkeypatch.setattr(sb, "_rebuilt_since_restore_failure", fake_rebuilt.__get__(sb))
    monkeypatch.setattr(sb, "_execute_unlocked", fake_save_code)
    await sb._exec_lock.acquire()
    task = asyncio.create_task(sb.save_session())
    await asyncio.sleep(0)
    assert not task.done()

    sb.runtime_revision = 2
    sb._exec_lock.release()

    assert await task is True
    assert sb.saved_runtime_revision == 2


@pytest.mark.asyncio
async def test_concurrent_direct_async_saves_serialize_one_generation(
    tmp_path, monkeypatch,
):
    from pathlib import Path
    from backend.utils.sandbox import CodeExecutionResult

    sb = LocalJupyterSandbox(
        session_id="direct-save-coalesce",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    sb.runtime_budgets = {}
    sb.runtime_revision = 1
    calls = 0
    serialization_calls = 0
    entered = asyncio.Event()

    async def fake_save_code(_code, **_kwargs):
        nonlocal calls, serialization_calls
        calls += 1
        entered.set()
        await asyncio.sleep(0.02)
        if "SCAI_ZSTD_DILL_V1" in _code:
            serialization_calls += 1
        Path(str(sb.notebook_state.session_path) + ".tmp").write_bytes(
            f"generation-{calls}".encode()
        )
        return CodeExecutionResult(
            success=True,
            output='Session saved: 1 variables\n__SCAI_SAVED_NAMES__:["adata"]\n__SCAI_EXCLUDED_KIND__:[]\n__SCAI_EXCLUDED_VOLATILE__:[]',
        )

    monkeypatch.setattr(sb, "_execute_unlocked", fake_save_code)
    first = asyncio.create_task(sb.save_session())
    await entered.wait()
    second = asyncio.create_task(sb.save_session())

    assert await asyncio.gather(first, second) == [True, True]
    assert serialization_calls == 1
    assert sb.saved_runtime_revision == 1
    assert sb.notebook_state.runtime_persistence["committed_revision"] == 1
    assert sb.notebook_state.runtime_persistence["runtime_digest"]


@pytest.mark.asyncio
async def test_durable_host_operation_does_not_block_event_loop():
    from backend.utils.sandbox import _run_durable_host_operation

    started = threading.Event()
    release = threading.Event()

    def blocking_io():
        started.set()
        release.wait(timeout=2)
        return "committed"

    task = asyncio.create_task(_run_durable_host_operation(
        blocking_io,
        session_id="host-io",
        description="test commit",
    ))
    while not started.is_set():
        await asyncio.sleep(0)

    heartbeat = asyncio.Event()
    asyncio.get_running_loop().call_soon(heartbeat.set)
    await asyncio.wait_for(heartbeat.wait(), timeout=0.5)
    assert not task.done()

    release.set()
    assert await task == "committed"


@pytest.mark.asyncio
async def test_durable_host_operation_reaches_safe_boundary_before_cancellation():
    from backend.utils.sandbox import _run_durable_host_operation

    started = threading.Event()
    release = threading.Event()

    def blocking_io():
        started.set()
        release.wait(timeout=2)

    task = asyncio.create_task(_run_durable_host_operation(
        blocking_io,
        session_id="host-io-cancel",
        description="test commit",
    ))
    while not started.is_set():
        await asyncio.sleep(0)

    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


def _save_test_sandbox(tmp_path, session_id, monkeypatch):
    from pathlib import Path
    from backend.models.schemas import AnalysisState
    from backend.services import session_lifecycle
    from backend.utils.sandbox import _write_pkl_signature

    for name in (
        "SESSION_DIR",
        "UPLOAD_DIR",
        "WORKSPACE_DIR",
        "REPORTS_DIR",
        "STATE_DIR",
    ):
        monkeypatch.setattr(
            session_lifecycle,
            name,
            tmp_path / f"lifecycle-{name.lower()}",
        )
    session_lifecycle.reset_lifecycle_locks()
    lifecycle_state = AnalysisState(id=session_id, user_id="owner")
    assert session_lifecycle.register_new_generation(lifecycle_state) == 1

    sb = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    sb.runtime_budgets = {}
    sb.runtime_revision = 2
    sb.saved_runtime_revision = 1
    integrity = tmp_path / "integrity"
    # The readers live in backend/utils/sandbox_parts/shared.py, so patching
    # the facade alone rebinds a name nobody reads and the test silently
    # exercises the real dependency. setattr_facade binds every holder.
    setattr_facade(
        monkeypatch,
        "backend.utils.sandbox",
        "_integrity_dir",
        lambda _session_id: integrity,
    )
    old_pkl = Path(sb.notebook_state.session_path)
    old_pkl.write_bytes(b"old-generation")
    _write_pkl_signature(old_pkl, integrity)
    old_digest = __import__("hashlib").sha256(old_pkl.read_bytes()).hexdigest()
    sb.notebook_state.update_runtime_persistence(
        "current",
        current_revision=1,
        saved_revision=1,
        commit_source="regular_save",
        runtime_digest=old_digest,
    )

    async def fake_save_code(code, **_kwargs):
        if "__USER_GLOBALS__:" in code:
            return type("Result", (), {
                "success": True,
                "output": "__USER_GLOBALS__:[]",
                "error": None,
            })()
        Path(str(old_pkl) + ".tmp").write_bytes(b"new-generation")
        return type("Result", (), {
            "success": True,
            "output": (
                'Session saved: 1 variables\n'
                '__SCAI_SAVED_NAMES__:["adata"]\n__SCAI_EXCLUDED_KIND__:[]\n__SCAI_EXCLUDED_VOLATILE__:[]'
            ),
            "error": None,
        })()

    monkeypatch.setattr(sb, "_execute_unlocked", fake_save_code)
    return sb, old_pkl, integrity


@pytest.mark.asyncio
async def test_cancelled_digest_removes_unjournaled_staged_generation(
    tmp_path, monkeypatch,
):
    from backend.utils.sandbox import (
        _sig_path,
        _sha256_file,
    )

    sb, old_pkl, integrity = _save_test_sandbox(
        tmp_path, "cancelled-runtime-digest", monkeypatch
    )
    staged = old_pkl.with_name(old_pkl.name + ".tmp")
    old_sig = _sig_path(old_pkl, integrity)
    foreign_backup = old_pkl.with_name(f"{old_pkl.name}.regular-save.bak")
    foreign_temp = foreign_backup.parent / f".{foreign_backup.name}.foreign.tmp"
    foreign_backup.write_bytes(b"foreign-backup-sentinel")
    foreign_temp.write_bytes(b"foreign-temp-sentinel")
    old_pkl_bytes = old_pkl.read_bytes()
    old_sig_bytes = old_sig.read_bytes()
    old_metadata = sb.notebook_state.state_path.read_bytes()
    old_persistence = dict(sb.notebook_state.runtime_persistence)
    failed_updates = []
    original_update = sb.notebook_state.update_runtime_persistence

    def track_failed(status, *args, **kwargs):
        if status == "failed":
            failed_updates.append((args, kwargs))
        return original_update(status, *args, **kwargs)

    monkeypatch.setattr(
        sb.notebook_state,
        "update_runtime_persistence",
        track_failed,
    )
    started = threading.Event()
    release = threading.Event()

    def blocking_digest(_path):
        started.set()
        release.wait(timeout=2)
        return "a" * 64

    setattr_facade(
        monkeypatch, "backend.utils.sandbox", "_sha256_file", blocking_digest
    )
    task = asyncio.create_task(sb.save_session())
    while not started.is_set():
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0.01)
    assert staged.exists()
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert old_pkl.read_bytes() == old_pkl_bytes
    assert old_sig.read_bytes() == old_sig_bytes
    assert foreign_backup.read_bytes() == b"foreign-backup-sentinel"
    assert foreign_temp.read_bytes() == b"foreign-temp-sentinel"
    assert not staged.exists()
    persistence = sb.notebook_state.runtime_persistence
    assert len(failed_updates) == 1
    assert persistence["status"] == "failed"
    assert persistence["last_attempt_status"] == "failed"
    assert persistence["last_attempt_revision"] == 2
    assert persistence["saved_revision"] == old_persistence["saved_revision"]
    assert persistence["committed_revision"] == old_persistence["committed_revision"]
    assert persistence["committed_digest"] == old_persistence["committed_digest"]
    assert persistence["runtime_digest"] == old_persistence["runtime_digest"]
    assert persistence["last_attempt_error"]
    assert len(persistence["last_attempt_error"]) <= 500
    assert sb.notebook_state.state_path.read_bytes() != old_metadata
    assert _sha256_file(old_pkl) == old_persistence["runtime_digest"]


@pytest.mark.asyncio
async def test_cancelled_commit_after_worker_success_keeps_truthful_new_generation(
    tmp_path, monkeypatch,
):
    from backend.utils.sandbox import (
        _sig_path,
        _verify_pkl_integrity,
        _sha256_file,
    )

    sb, old_pkl, integrity = _save_test_sandbox(
        tmp_path, "cancelled-commit-success", monkeypatch
    )
    started = threading.Event()
    release = threading.Event()
    def hold_journal(point):
        if point == "journal_applying":
            started.set()
            assert release.wait(timeout=5)

    monkeypatch.setattr(sb, "_regular_save_fault_hook", hold_journal, raising=False)
    task = asyncio.create_task(sb.save_session())
    while not started.is_set():
        await asyncio.sleep(0)
    journal = old_pkl.with_name(f"{old_pkl.name}.regular-save.json")
    assert journal.exists()
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not journal.exists()
    assert old_pkl.read_bytes() == b"new-generation"
    assert _verify_pkl_integrity(old_pkl, integrity)
    persistence = sb.notebook_state.runtime_persistence
    assert persistence["status"] == "current"
    assert persistence["saved_revision"] == 2
    assert persistence["committed_revision"] == 2
    assert persistence["runtime_digest"] == _sha256_file(old_pkl)
    assert persistence["committed_digest"] == persistence["runtime_digest"]
    assert not old_pkl.with_name(old_pkl.name + ".tmp").exists()
    assert not _sig_path(old_pkl, integrity).with_name(
        f"{_sig_path(old_pkl, integrity).name}.regular-save.tmp"
    ).exists()
    assert not list(old_pkl.parent.glob("*.regular-save.bak"))
    assert not list(old_pkl.parent.glob(".*.regular-save.bak.*.tmp"))


@pytest.mark.asyncio
async def test_cancelled_commit_after_worker_rollback_preserves_previous_generation(
    tmp_path, monkeypatch,
):
    from backend.utils.sandbox import (
        _sig_path,
        _verify_pkl_integrity,
    )

    sb, old_pkl, integrity = _save_test_sandbox(
        tmp_path, "cancelled-commit-rollback", monkeypatch
    )
    old_bytes = old_pkl.read_bytes()
    old_sig = _sig_path(old_pkl, integrity).read_bytes()
    old_metadata = sb.notebook_state.state_path.read_bytes()
    old_identity = {
        key: sb.notebook_state.runtime_persistence.get(key)
        for key in (
            "committed_revision",
            "committed_digest",
            "saved_revision",
            "runtime_digest",
        )
    }
    failed_updates = []
    original_update = sb.notebook_state.update_runtime_persistence

    def track_failed(status, *args, **kwargs):
        if status == "failed":
            failed_updates.append((args, kwargs))
        return original_update(status, *args, **kwargs)

    monkeypatch.setattr(
        sb.notebook_state,
        "update_runtime_persistence",
        track_failed,
    )
    started = threading.Event()
    release = threading.Event()

    def fail_after_journal(point):
        if point == "journal_applying":
            started.set()
            assert release.wait(timeout=5)
        elif point == "pkl_replaced":
            raise RuntimeError("journal-owned failure")

    monkeypatch.setattr(
        sb,
        "_regular_save_fault_hook",
        fail_after_journal,
        raising=False,
    )
    task = asyncio.create_task(sb.save_session())
    while not started.is_set():
        await asyncio.sleep(0)
    journal = old_pkl.with_name(f"{old_pkl.name}.regular-save.json")
    assert journal.exists()
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await task

    assert isinstance(cancelled.value.__cause__, RuntimeError)
    assert str(cancelled.value.__cause__) == "journal-owned failure"
    assert old_pkl.read_bytes() == old_bytes
    assert _sig_path(old_pkl, integrity).read_bytes() == old_sig
    assert _verify_pkl_integrity(old_pkl, integrity)
    persistence = sb.notebook_state.runtime_persistence
    assert len(failed_updates) == 1
    assert persistence["status"] == "failed"
    assert persistence["last_attempt_status"] == "failed"
    assert persistence["last_attempt_revision"] == 2
    for key, value in old_identity.items():
        assert persistence[key] == value
    assert persistence["last_attempt_error"]
    assert len(persistence["last_attempt_error"]) <= 500
    assert sb.notebook_state.state_path.read_bytes() != old_metadata
    assert not journal.exists()
    assert not old_pkl.with_name(old_pkl.name + ".tmp").exists()
    staged_sig = _sig_path(old_pkl, integrity).with_name(
        f"{_sig_path(old_pkl, integrity).name}.regular-save.tmp"
    )
    assert not staged_sig.exists()
    assert not list(old_pkl.parent.glob("*.regular-save.bak"))
    assert not list(old_pkl.parent.glob(".*.regular-save.bak.*.tmp"))


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected_error", ["generation", "gone"])
async def test_rejected_lifecycle_generation_preserves_previous_runtime_and_cleans_staging(
    tmp_path, monkeypatch, rejected_error,
):
    from backend.services import session_lifecycle
    from backend.utils.sandbox import (
        _sig_path,
        _verify_pkl_integrity,
    )

    sb, old_pkl, integrity = _save_test_sandbox(
        tmp_path, "rejected-runtime-generation", monkeypatch
    )
    old_bytes = old_pkl.read_bytes()
    old_sig = _sig_path(old_pkl, integrity).read_bytes()
    old_metadata = sb.notebook_state.state_path.read_bytes()
    old_persistence = dict(sb.notebook_state.runtime_persistence)
    if rejected_error == "generation":
        with session_lifecycle.lifecycle_file_lock(sb.session_id):
            record = session_lifecycle.read_lifecycle_record(sb.session_id)
            replacement = dict(record)
            replacement.update({
                "generation": 2,
                "status": "active",
                "deleted_generations": [],
            })
            session_lifecycle._write_record(replacement)
        expected_error = session_lifecycle.SessionGenerationError
    else:
        session_lifecycle.tombstone_generation(
            sb.session_id,
            "owner",
            1,
        )
        expected_error = session_lifecycle.SessionGoneError

    failed_updates = []
    original_update = sb.notebook_state.update_runtime_persistence

    def track_failed(status, *args, **kwargs):
        if status == "failed":
            failed_updates.append((args, kwargs))
        return original_update(status, *args, **kwargs)

    monkeypatch.setattr(
        sb.notebook_state,
        "update_runtime_persistence",
        track_failed,
    )
    with pytest.raises(expected_error):
        await sb.save_session()

    staged = old_pkl.with_name(old_pkl.name + ".tmp")
    staged_sig = _sig_path(old_pkl, integrity).with_name(
        f"{_sig_path(old_pkl, integrity).name}.regular-save.tmp"
    )
    assert old_pkl.read_bytes() == old_bytes
    assert _sig_path(old_pkl, integrity).read_bytes() == old_sig
    assert old_pkl.with_name(old_pkl.name + ".regular-save.json").exists() is False
    assert not staged.exists()
    assert not staged_sig.exists()
    assert sb.notebook_state.state_path.read_bytes() == old_metadata
    assert sb.notebook_state.runtime_persistence == old_persistence
    assert failed_updates == []
    assert not list(old_pkl.parent.glob("*.regular-save.bak"))
    assert not list(old_pkl.parent.glob(".*.regular-save.bak.*.tmp"))
    assert _verify_pkl_integrity(old_pkl, integrity)


def test_sandbox_init_creates_exec_lock():
    sb = object.__new__(LocalJupyterSandbox)
    LocalJupyterSandbox._init_exec_lock(sb)
    assert isinstance(sb._exec_lock, asyncio.Lock)
