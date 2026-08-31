import asyncio
import contextvars
import gc
import threading
import weakref

import pytest

from backend.constants import VERSION
from backend.models.schemas import AnalysisState
from backend.services import session_lifecycle, session_store
from backend.services.session_lifecycle import (
    SessionGoneError,
    SessionGenerationError,
    abandon_import_generation,
    adopt_persisted_generation,
    bind_import_generation,
    claim_import_generation,
    complete_tombstone_cleanup,
    commit_import_generation,
    get_lifecycle_lock,
    lifecycle_file_lock,
    read_lifecycle_record,
    register_new_generation,
    tombstone_generation,
)
from backend.state import sessions


async def _wait_thread_event(event: threading.Event, timeout: float = 5.0) -> bool:
    """Wait without consuming a second default-executor worker in race tests."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if event.is_set():
            return True
        await asyncio.sleep(0.01)
    return event.is_set()


async def _joined_thread(func, /, *args, **kwargs):
    """Python 3.13/old-AnyIO-safe to_thread equivalent for these tests."""
    context = contextvars.copy_context()
    outcome = []

    def run():
        try:
            outcome.append((True, context.run(func, *args, **kwargs)))
        except BaseException as exc:
            outcome.append((False, exc))

    thread = threading.Thread(target=run)
    thread.start()
    try:
        while thread.is_alive():
            await asyncio.sleep(0.01)
    finally:
        thread.join()
    succeeded, value = outcome[0]
    if succeeded:
        return value
    raise value


def _configure_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(session_lifecycle, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(session_lifecycle, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(session_lifecycle, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.setattr(session_lifecycle, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(session_lifecycle, "STATE_DIR", tmp_path / "states")
    session_lifecycle.reset_lifecycle_locks()
    monkeypatch.setattr(asyncio, "to_thread", _joined_thread)
    monkeypatch.setattr(
        session_store,
        "get_session_metadata_path",
        lambda sid: tmp_path / sid / "session.json",
    )
    monkeypatch.setattr(
        session_store,
        "resolve_session_metadata_path",
        lambda sid: (
            tmp_path / sid / "session.json"
            if (tmp_path / sid / "session.json").exists()
            else None
        ),
    )


def test_lifecycle_lock_isolated_across_contended_event_loops():
    """A lock that acquired without waiting may hide stale loop affinity.

    Force a waiter in each independent loop so asyncio binds the lock before
    the next loop asks for the same session ID.
    """
    session_lifecycle.reset_lifecycle_locks()
    loop_refs = []
    lock_refs = []

    def run_one_loop():
        async def contend():
            loop_refs.append(weakref.ref(asyncio.get_running_loop()))
            lock = get_lifecycle_lock("cross-loop-contention")
            lock_refs.append(weakref.ref(lock))
            entered = asyncio.Event()
            release = asyncio.Event()

            async def holder():
                async with lock:
                    entered.set()
                    await release.wait()

            async def waiter():
                await entered.wait()
                async with get_lifecycle_lock("cross-loop-contention"):
                    return True

            holder_task = asyncio.create_task(holder())
            await entered.wait()
            waiter_task = asyncio.create_task(waiter())
            await asyncio.sleep(0)
            assert not waiter_task.done(), "the second acquisition did not contend"
            release.set()
            await asyncio.gather(holder_task, waiter_task)

        asyncio.run(contend())

    run_one_loop()
    run_one_loop()

    gc.collect()
    assert all(ref() is None for ref in lock_refs)
    assert all(ref() is None for ref in loop_refs)
    assert len(session_lifecycle._locks) == 0


@pytest.mark.asyncio
async def test_blocked_save_cannot_recreate_session_after_delete(monkeypatch, tmp_path):
    _configure_roots(monkeypatch, tmp_path)
    state = AnalysisState(id="race-save-delete", user_id="owner")
    register_new_generation(state)
    entered = threading.Event()
    release = threading.Event()
    real_write = session_store._externalize_and_write

    def blocked_write(session_id, dumped):
        entered.set()
        assert release.wait(timeout=5)
        real_write(session_id, dumped)

    monkeypatch.setattr(session_store, "_externalize_and_write", blocked_write)
    save_task = asyncio.create_task(
        session_store.async_save_session(state.id, state, force=True)
    )
    assert await _wait_thread_event(entered)

    delete_committed = asyncio.Event()

    async def delete_generation():
        async with get_lifecycle_lock(state.id):
            with lifecycle_file_lock(state.id):
                tombstone_generation(
                    state.id,
                    state.user_id,
                    state.lifecycle_generation,
                )
            metadata = tmp_path / state.id / "session.json"
            metadata.unlink(missing_ok=True)
            delete_committed.set()

    delete_task = asyncio.create_task(delete_generation())
    await asyncio.sleep(0)
    assert delete_committed.is_set() is False
    release.set()
    await asyncio.wait_for(save_task, timeout=5)
    await asyncio.wait_for(delete_task, timeout=5)

    assert not (tmp_path / state.id / "session.json").exists()
    with pytest.raises(SessionGoneError):
        session_lifecycle.assert_generation_writable(state.id, state)


@pytest.mark.asyncio
async def test_delete_during_load_io_cannot_reinstall_cache(monkeypatch, tmp_path):
    _configure_roots(monkeypatch, tmp_path)
    monkeypatch.setattr("backend.config.is_snapshot_v2_enabled", lambda: False)
    state = AnalysisState(id="race-load-delete", user_id="owner")
    register_new_generation(state)
    entered = threading.Event()
    release = threading.Event()

    def blocked_load(_session_id):
        entered.set()
        assert release.wait(timeout=5)
        return state.model_copy(deep=True)

    monkeypatch.setattr(session_store, "load_session_from_disk", blocked_load)
    sessions.pop(state.id, None)
    load_task = asyncio.create_task(session_store.ensure_session_loaded(state.id))
    assert await _wait_thread_event(entered)

    async with get_lifecycle_lock(state.id):
        with lifecycle_file_lock(state.id):
            tombstone_generation(
                state.id,
                state.user_id,
                state.lifecycle_generation,
            )
    release.set()

    with pytest.raises(SessionGoneError):
        await load_task
    assert state.id not in sessions


@pytest.mark.asyncio
async def test_disk_session_still_loads_after_process_cache_restart(monkeypatch, tmp_path):
    _configure_roots(monkeypatch, tmp_path)
    monkeypatch.setattr("backend.config.is_snapshot_v2_enabled", lambda: False)
    state = AnalysisState(id="restart-load", user_id="owner")
    assert state.scai_version is None
    register_new_generation(state)
    assert state.scai_version == VERSION
    session_store.save_session_to_disk(state.id, state)
    sessions.pop(state.id, None)

    loaded = await session_store.ensure_session_loaded(state.id)

    assert loaded is not None
    assert loaded.id == state.id
    assert loaded.lifecycle_generation == state.lifecycle_generation
    assert loaded.scai_version == VERSION


@pytest.mark.asyncio
async def test_recovery_is_inside_delete_lifecycle_boundary(monkeypatch, tmp_path):
    _configure_roots(monkeypatch, tmp_path)
    monkeypatch.setattr("backend.config.is_snapshot_v2_enabled", lambda: True)
    monkeypatch.setattr(session_store, "_snapshot_transaction_in_flight", lambda _sid: False)
    state = AnalysisState(id="race-recovery-delete", user_id="owner")
    register_new_generation(state)
    sessions.pop(state.id, None)
    recovery_entered = asyncio.Event()
    release_recovery = asyncio.Event()
    load_entered = threading.Event()
    release_load = threading.Event()
    recovered_artifact = tmp_path / state.id / "recovered.tmp"

    async def blocked_recovery(_session_id):
        recovery_entered.set()
        await release_recovery.wait()
        recovered_artifact.parent.mkdir(parents=True, exist_ok=True)
        recovered_artifact.write_text("recovered")
        return False, None

    def blocked_load(_session_id):
        load_entered.set()
        assert release_load.wait(timeout=5)
        return state.model_copy(deep=True)

    monkeypatch.setattr(session_store, "_recover_snapshot_v2_on_load", blocked_recovery)
    monkeypatch.setattr(session_store, "load_session_from_disk", blocked_load)
    load_task = asyncio.create_task(session_store.ensure_session_loaded(state.id))
    await recovery_entered.wait()

    delete_committed = asyncio.Event()

    async def delete_generation():
        async with get_lifecycle_lock(state.id):
            with lifecycle_file_lock(state.id):
                tombstone_generation(
                    state.id,
                    state.user_id,
                    state.lifecycle_generation,
                )
            if recovered_artifact.parent.exists():
                import shutil

                shutil.rmtree(recovered_artifact.parent)
            delete_committed.set()

    delete_task = asyncio.create_task(delete_generation())
    await asyncio.sleep(0)
    assert delete_committed.is_set() is False
    release_recovery.set()
    assert await _wait_thread_event(load_entered)
    await delete_committed.wait()
    release_load.set()

    with pytest.raises(SessionGoneError):
        await load_task
    await delete_task
    assert not recovered_artifact.exists()
    assert state.id not in sessions


def test_new_generation_rejects_pre_lifecycle_disk_collision(monkeypatch, tmp_path):
    _configure_roots(monkeypatch, tmp_path)
    state = AnalysisState(id="persisted-collision", user_id="owner")
    persisted = session_lifecycle.SESSION_DIR / state.id / "session.json"
    persisted.parent.mkdir(parents=True)
    persisted.write_text("{}", encoding="utf-8")

    with pytest.raises(SessionGenerationError, match="persisted data"):
        register_new_generation(state)


@pytest.mark.parametrize(
    "suffix",
    [
        ".ipynb",
        "_state.json",
        "_session.pkl",
        "_bioinfo_checkpoint.json",
        "_report_checkpoint.json",
    ],
)
def test_new_generation_rejects_each_exact_legacy_state_collision(
    monkeypatch, tmp_path, suffix,
):
    _configure_roots(monkeypatch, tmp_path)
    state = AnalysisState(id="legacy-state", user_id="owner")
    session_lifecycle.STATE_DIR.mkdir(parents=True)
    (session_lifecycle.STATE_DIR / f"{state.id}{suffix}").write_bytes(b"old")

    with pytest.raises(SessionGenerationError, match="persisted data"):
        register_new_generation(state)


def test_new_generation_ignores_near_prefix_legacy_state(
    monkeypatch, tmp_path,
):
    _configure_roots(monkeypatch, tmp_path)
    state = AnalysisState(id="abc", user_id="owner")
    session_lifecycle.STATE_DIR.mkdir(parents=True)
    (session_lifecycle.STATE_DIR / "abcd_session.pkl").write_bytes(b"other")
    (session_lifecycle.STATE_DIR / "abc_extra_state.json").write_bytes(b"other")

    assert register_new_generation(state) == 1


@pytest.mark.parametrize(
    ("session_id", "ambiguous_name"),
    [
        ("abc_state", "abc_state.json"),
        ("abc_session", "abc_session.pkl"),
    ],
)
def test_new_generation_ignores_ambiguous_shorter_id_legacy_file(
    monkeypatch, tmp_path, session_id, ambiguous_name,
):
    _configure_roots(monkeypatch, tmp_path)
    state = AnalysisState(id=session_id, user_id="owner")
    session_lifecycle.STATE_DIR.mkdir(parents=True)
    (session_lifecycle.STATE_DIR / ambiguous_name).write_bytes(b"shorter")

    assert register_new_generation(state) == 1


def test_save_id_mismatch_fails_before_lifecycle_mutation(monkeypatch, tmp_path):
    _configure_roots(monkeypatch, tmp_path)
    expected_id = "expected-session"
    state = AnalysisState(id="different-session", user_id="owner")

    with pytest.raises(SessionGenerationError, match="session id mismatch"):
        session_lifecycle.assert_generation_writable(expected_id, state)

    assert read_lifecycle_record(expected_id) is None
    assert read_lifecycle_record(state.id) is None
    assert state.lifecycle_generation == 0


def test_serialized_save_id_mismatch_fails_before_disk_or_lifecycle_mutation(
    monkeypatch, tmp_path,
):
    _configure_roots(monkeypatch, tmp_path)
    expected_id = "expected-serialized-session"
    state = AnalysisState(id="different-serialized-session", user_id="owner")
    metadata = tmp_path / expected_id / "session.json"

    with pytest.raises(SessionGenerationError, match="session id mismatch"):
        session_store.save_session_to_disk(expected_id, state)

    assert read_lifecycle_record(expected_id) is None
    assert read_lifecycle_record(state.id) is None
    assert not metadata.exists()
    assert state.lifecycle_generation == 0


def test_adopt_fails_closed_when_multiple_owner_records_exist(monkeypatch, tmp_path):
    _configure_roots(monkeypatch, tmp_path)
    state = AnalysisState(id="ambiguous-owner", user_id="owner-a")
    for owner in ("owner-a", "owner-b"):
        session_lifecycle._write_record({
            "session_id": state.id,
            "owner_id": owner,
            "generation": 1,
            "status": "active",
            "created_at": "now",
            "deleted_at": None,
        })

    with pytest.raises(SessionGenerationError, match="multiple lifecycle owner"):
        adopt_persisted_generation(state)


def test_deleted_id_import_claims_next_generation_and_stale_state_stays_gone(
    monkeypatch, tmp_path,
):
    _configure_roots(monkeypatch, tmp_path)
    stale = AnalysisState(id="same-id-import", user_id="owner")
    register_new_generation(stale)
    tombstone_generation(stale.id, stale.user_id, stale.lifecycle_generation)

    session_dir = session_lifecycle.SESSION_DIR / stale.id
    assert claim_import_generation(stale.id, stale.user_id, session_dir) is None
    assert complete_tombstone_cleanup(
        stale.id, stale.user_id, stale.lifecycle_generation,
    ) is True
    claimed = claim_import_generation(stale.id, stale.user_id, session_dir)
    assert claimed == 2

    importing = AnalysisState(
        id=stale.id,
        user_id=stale.user_id,
        # An archive may carry the source installation's old counter. The
        # destination reservation, not that untrusted value, is authoritative.
        lifecycle_generation=1,
    )
    assert bind_import_generation(importing) == 2
    record = read_lifecycle_record(stale.id)
    assert record["status"] == "importing"
    assert record["deleted_generations"] == [1]

    commit_import_generation(importing)
    record = read_lifecycle_record(stale.id)
    assert record["status"] == "active"
    assert record["generation"] == 2

    with pytest.raises(SessionGoneError) as delete_exc:
        tombstone_generation(
            stale.id, stale.user_id, stale.lifecycle_generation,
        )
    assert delete_exc.value.generation == 1
    assert read_lifecycle_record(stale.id)["status"] == "active"

    with pytest.raises(SessionGoneError) as exc_info:
        session_lifecycle.assert_generation_writable(stale.id, stale)
    assert exc_info.value.generation == 1
    assert_generation = adopt_persisted_generation(importing)
    assert assert_generation == 2


def test_deleted_id_cannot_be_reclaimed_by_another_owner(monkeypatch, tmp_path):
    _configure_roots(monkeypatch, tmp_path)
    state = AnalysisState(id="foreign-deleted-id", user_id="owner-a")
    register_new_generation(state)
    tombstone_generation(state.id, state.user_id, state.lifecycle_generation)

    session_dir = session_lifecycle.SESSION_DIR / state.id
    assert claim_import_generation(state.id, "owner-b", session_dir) is None
    assert not session_dir.exists()
    record = read_lifecycle_record(state.id)
    assert record["status"] == "deleted"
    assert record["owner_id"] == "owner-a"
    assert record["generation"] == 1


def test_failed_same_id_import_tombstones_reserved_generation(monkeypatch, tmp_path):
    _configure_roots(monkeypatch, tmp_path)
    state = AnalysisState(id="failed-same-id-import", user_id="owner")
    register_new_generation(state)
    tombstone_generation(state.id, state.user_id, state.lifecycle_generation)

    session_dir = session_lifecycle.SESSION_DIR / state.id
    assert claim_import_generation(state.id, state.user_id, session_dir) is None
    assert complete_tombstone_cleanup(
        state.id, state.user_id, state.lifecycle_generation,
    ) is True
    assert claim_import_generation(state.id, state.user_id, session_dir) == 2
    assert abandon_import_generation(state.id) is True

    record = read_lifecycle_record(state.id)
    assert record["status"] == "deleted"
    assert record["generation"] == 2
    assert record["deleted_generations"] == [1, 2]
    assert record["cleanup_complete"] is False
