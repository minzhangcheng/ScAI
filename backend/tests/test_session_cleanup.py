"""Unit tests for backend.tasks.session_cleanup.

These characterization tests lock in current behavior so that the
upcoming refactor (Task 3) is provably behavior-preserving.
"""
import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from backend.tasks import session_cleanup as sc_mod
from backend import state as state_mod


@pytest.fixture(autouse=True)
def _reset_state():
    """Clear module-level dicts/locks between tests."""
    state_mod.sessions.clear()
    state_mod.workflows.clear()
    state_mod.report_workflows.clear()
    state_mod.initializing_sessions.clear()
    state_mod.initializing_session_owners.clear()
    state_mod.session_locks.clear()
    state_mod.session_access_times.clear()
    state_mod.last_save_times.clear()
    state_mod.report_websockets.clear()
    state_mod.detailed_reports.clear()
    yield
    state_mod.sessions.clear()
    state_mod.workflows.clear()
    state_mod.report_workflows.clear()
    state_mod.initializing_sessions.clear()
    state_mod.initializing_session_owners.clear()
    state_mod.session_locks.clear()
    state_mod.session_access_times.clear()
    state_mod.last_save_times.clear()


async def _run_one_cleanup_cycle(chat_task_lock, chat_sandboxes, import_jobs, import_jobs_lock):
    """Run cleanup_stale_sessions for exactly one iteration, then cancel.

    CLEANUP_INTERVAL is patched to 0 so the loop enters immediately.
    """
    task = asyncio.create_task(
        sc_mod.cleanup_stale_sessions(
            chat_task_lock, chat_sandboxes, import_jobs, import_jobs_lock,
        )
    )
    # Let the task run one loop iteration
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_idle_session_is_saved_and_evicted():
    state_mod.sessions["s1"] = {"data": "hello"}
    state_mod.session_access_times["s1"] = time.time() - 10_000  # long idle

    mock_save = AsyncMock()
    with patch.object(sc_mod, "CLEANUP_INTERVAL", 0), \
         patch.object(sc_mod, "SESSION_CACHE_TTL", 1), \
         patch.object(sc_mod, "MAX_CACHED_SESSIONS", 100), \
         patch.object(sc_mod, "_async_save_session_under_lifecycle", mock_save):
        await _run_one_cleanup_cycle(
            asyncio.Lock(), {}, {}, asyncio.Lock(),
        )

    mock_save.assert_awaited_once()
    assert "s1" not in state_mod.sessions
    assert "s1" not in state_mod.session_access_times


@pytest.mark.asyncio
async def test_initializing_session_not_evicted():
    state_mod.sessions["s1"] = {"data": "x"}
    state_mod.session_access_times["s1"] = time.time() - 10_000
    state_mod.initializing_sessions.add("s1")

    mock_save = AsyncMock()
    with patch.object(sc_mod, "CLEANUP_INTERVAL", 0), \
         patch.object(sc_mod, "SESSION_CACHE_TTL", 1), \
         patch.object(sc_mod, "MAX_CACHED_SESSIONS", 100), \
         patch.object(sc_mod, "_async_save_session_under_lifecycle", mock_save):
        await _run_one_cleanup_cycle(
            asyncio.Lock(), {}, {}, asyncio.Lock(),
        )

    mock_save.assert_not_awaited()
    assert "s1" in state_mod.sessions


@pytest.mark.asyncio
async def test_report_only_workflow_not_selected_for_eviction():
    state_mod.sessions["s1"] = {"data": "x"}
    state_mod.session_access_times["s1"] = time.time() - 10_000
    state_mod.report_workflows["s1"] = object()

    with patch.object(sc_mod, "SESSION_CACHE_TTL", 1), \
         patch.object(sc_mod, "MAX_CACHED_SESSIONS", 100):
        selected = await sc_mod._collect_sessions_to_evict(time.time())

    assert selected == []


@pytest.mark.asyncio
async def test_report_owner_installed_after_selection_blocks_eviction():
    state_mod.sessions["s1"] = {"data": "x"}
    state_mod.session_access_times["s1"] = time.time() - 10_000
    state_mod.report_workflows["s1"] = object()
    save = AsyncMock()

    with patch.object(sc_mod, "_async_save_session_under_lifecycle", save):
        await sc_mod._evict_single_session("s1", asyncio.Lock(), {})

    save.assert_not_awaited()
    assert "s1" in state_mod.sessions


@pytest.mark.asyncio
async def test_chat_sandbox_stopped_on_eviction():
    state_mod.sessions["s1"] = {"data": "x"}
    state_mod.session_access_times["s1"] = time.time() - 10_000

    sandbox = AsyncMock()
    chat_sandboxes = {"s1": sandbox}

    with patch.object(sc_mod, "CLEANUP_INTERVAL", 0), \
         patch.object(sc_mod, "SESSION_CACHE_TTL", 1), \
         patch.object(sc_mod, "MAX_CACHED_SESSIONS", 100), \
         patch.object(sc_mod, "_async_save_session_under_lifecycle", AsyncMock()):
        await _run_one_cleanup_cycle(
            asyncio.Lock(), chat_sandboxes, {}, asyncio.Lock(),
        )

    sandbox.stop.assert_awaited_once()
    assert "s1" not in chat_sandboxes


@pytest.mark.asyncio
async def test_stale_import_jobs_cleaned():
    now = time.time()
    import_jobs = {
        "j1": {"status": "completed", "created_at": now - 10_000},
        "j2": {"status": "running", "created_at": now - 10_000},
        "j3": {"status": "failed", "created_at": now - 1},
    }

    with patch.object(sc_mod, "CLEANUP_INTERVAL", 0), \
         patch.object(sc_mod, "IMPORT_JOB_TTL", 100), \
         patch.object(sc_mod, "SESSION_CACHE_TTL", 1_000_000), \
         patch.object(sc_mod, "MAX_CACHED_SESSIONS", 100), \
         patch.object(sc_mod, "_async_save_session_under_lifecycle", AsyncMock()):
        await _run_one_cleanup_cycle(
            asyncio.Lock(), {}, import_jobs, asyncio.Lock(),
        )

    assert "j1" not in import_jobs   # completed + old → cleaned
    assert "j2" in import_jobs        # running → kept
    assert "j3" in import_jobs        # failed but fresh → kept
