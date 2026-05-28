"""C5 — cleanup uses asyncio.to_thread; ttl query covers completed leftovers."""
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.mark.asyncio
async def test_delete_user_data_offloads_blocking_io(tmp_path, monkeypatch):
    """shutil.rmtree, glob, and unlink must all go through asyncio.to_thread."""
    from backend.services import user_cleanup

    # Track every call to asyncio.to_thread
    original_to_thread = asyncio.to_thread
    calls = []

    async def tracking_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return await original_to_thread(func, *args, **kwargs)

    # Setup: one fake session belonging to user-X with a directory on disk
    sess_id = "sess-abc"
    session_dir = tmp_path / "sessions" / sess_id
    session_dir.mkdir(parents=True)
    (session_dir / "marker.txt").write_text("x")
    state_dir = tmp_path / "states"
    state_dir.mkdir()
    (state_dir / f"{sess_id}.pkl").write_text("blob")

    monkeypatch.setattr(user_cleanup, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(user_cleanup, "STATE_DIR", state_dir)

    monkeypatch.setattr(
        user_cleanup, "list_saved_sessions",
        lambda: [{"id": sess_id, "user_id": "user-X"}],
    )

    fake_db = MagicMock()
    fake_db.delete_api_keys_by_user = AsyncMock()
    fake_db.delete_user_llm_config = AsyncMock()
    fake_db.delete_user = AsyncMock()
    monkeypatch.setattr(user_cleanup, "get_db", AsyncMock(return_value=fake_db))

    monkeypatch.setattr(asyncio, "to_thread", tracking_to_thread)

    result = await user_cleanup.delete_user_data("user-X")

    assert result["deleted_sessions"] == 1
    assert result["deleted_state_files"] == 1
    # Both the rmtree and the unlink path must have been routed via to_thread.
    assert any("rmtree" in n or "_rmtree" in n for n in calls), \
        f"shutil.rmtree not offloaded to thread; to_thread calls: {calls}"
    assert any("unlink" in n or "_unlink" in n or "scan_and_unlink" in n for n in calls), \
        f"state-file unlink not offloaded; to_thread calls: {calls}"
    # And the session directory + state file are actually gone.
    assert not session_dir.exists()
    assert not (state_dir / f"{sess_id}.pkl").exists()


@pytest.mark.asyncio
async def test_list_expired_uploads_includes_completed(app_db):
    """The query used by ttl_cleanup must include completed-but-expired records."""
    from datetime import datetime, timezone, timedelta

    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    # Completed-and-expired: should still be returned for cleanup
    uid_completed = await app_db.create_upload_session(
        session_id="s1",
        filename="x",
        expected_sha256="abc",
        total_chunks=1,
        chunk_size=1024,
        expires_at_iso=past,
        placeholder_file_id="pf1",
    )
    await app_db.complete_upload_session(uid_completed)

    # Pending-but-expired: should be returned
    uid_pending_expired = await app_db.create_upload_session(
        session_id="s1",
        filename="y",
        expected_sha256="abc",
        total_chunks=1,
        chunk_size=1024,
        expires_at_iso=past,
        placeholder_file_id="pf2",
    )

    # Pending-but-not-expired: should NOT be returned
    await app_db.create_upload_session(
        session_id="s1",
        filename="z",
        expected_sha256="abc",
        total_chunks=1,
        chunk_size=1024,
        expires_at_iso=future,
        placeholder_file_id="pf3",
    )

    expired = await app_db.list_expired_uploads(
        datetime.now(timezone.utc).isoformat()
    )
    ids = {e["id"] for e in expired}
    assert uid_completed in ids, \
        "completed-but-expired upload must be returned for cleanup"
    assert uid_pending_expired in ids
    assert len(expired) == 2


@pytest.mark.asyncio
async def test_ttl_cleanup_offloads_unlink(monkeypatch, tmp_path):
    """ttl_cleanup_task must use asyncio.to_thread for chunk file unlinks."""
    from backend.tasks import ttl_cleanup

    monkeypatch.setattr(ttl_cleanup, "UPLOAD_TEMP_DIR", tmp_path)
    chunk = tmp_path / "uX.chunk.0"
    chunk.write_text("data")

    fake_db = MagicMock()
    fake_db.list_expired_uploads = AsyncMock(return_value=[
        {"id": "uX", "total_chunks": 1, "completed": 0}
    ])
    fake_db.delete_upload_session = AsyncMock()
    monkeypatch.setattr(ttl_cleanup, "get_db", AsyncMock(return_value=fake_db))

    calls = []
    original_to_thread = asyncio.to_thread

    async def tracking_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", tracking_to_thread)

    # Run one iteration only — patch the sleep to raise, breaking the loop.
    class _Stop(Exception): ...
    async def _fail_sleep(*_a, **_kw):
        raise _Stop
    monkeypatch.setattr(ttl_cleanup.asyncio, "sleep", _fail_sleep)

    with pytest.raises(_Stop):
        await ttl_cleanup.ttl_cleanup_task()

    assert not chunk.exists()
    assert any("unlink" in n or "_unlink" in n for n in calls), \
        f"chunk unlink not offloaded; to_thread calls: {calls}"