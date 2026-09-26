import asyncio
import contextvars
import threading

import pytest

from backend.models.schemas import AnalysisState
from backend.services import session_lifecycle, session_store
from backend.services.session_lifecycle import (
    SessionGoneError,
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
