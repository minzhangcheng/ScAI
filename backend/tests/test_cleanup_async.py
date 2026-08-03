"""C5 — cleanup uses asyncio.to_thread; ttl query covers completed leftovers."""
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


async def _inline_to_thread(func, *args, **kwargs):
    """Run a thread-boundary call inline on hosts with a blocked executor."""
    return func(*args, **kwargs)


@pytest.mark.asyncio
async def test_delete_user_data_offloads_blocking_io(tmp_path, monkeypatch):
    """Cascade discovery is offloaded and deletion uses the unified tombstone path."""
    from backend.services import user_cleanup
    from backend.models.schemas import AnalysisState
    from backend.services import session_deletion

    # Track every call to asyncio.to_thread
    calls = []

    async def tracking_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        # The contract under test is that blocking work is routed through this
        # boundary. Execute inline so Python 3.13 + old AnyIO test hosts do not
        # hang while shutting down the default executor.
        return func(*args, **kwargs)

    sess_id = "sess-abc"
    state = AnalysisState(id=sess_id, user_id="user-X")

    monkeypatch.setattr(
        user_cleanup, "list_saved_sessions",
        lambda: [{"id": sess_id, "user_id": "user-X"}],
    )
    monkeypatch.setattr(user_cleanup, "load_session_from_disk", lambda _sid: state)
    monkeypatch.setattr(user_cleanup, "_owned_lifecycle_records", lambda _uid: {})
    monkeypatch.setattr(user_cleanup, "import_jobs", {})
    monkeypatch.setattr(user_cleanup, "import_jobs_lock", asyncio.Lock())
    monkeypatch.setattr(user_cleanup, "import_generation_tasks", {})
    monkeypatch.setattr(user_cleanup, "users_pending_deletion", set())
    unified_delete = AsyncMock(return_value={
        "deleted_sessions": 1,
        "deleted_state_files": 1,
    })
    monkeypatch.setattr(
        session_deletion, "delete_session_generation", unified_delete,
    )

    fake_db = MagicMock()
    fake_db.set_user_status = AsyncMock(return_value=True)
    fake_db.delete_api_keys_by_user = AsyncMock()
    fake_db.delete_user_llm_config = AsyncMock()
    fake_db.delete_user = AsyncMock()
    monkeypatch.setattr(user_cleanup, "get_db", AsyncMock(return_value=fake_db))

    monkeypatch.setattr(asyncio, "to_thread", tracking_to_thread)

    result = await user_cleanup.delete_user_data("user-X")

    assert result["deleted_sessions"] == 1
    assert result["deleted_state_files"] == 1
    assert calls.count("<lambda>") >= 2
    unified_delete.assert_awaited_once_with(
        sess_id, state, "user-X", force_stop=True,
    )
    fake_db.set_user_status.assert_awaited_once_with("user-X", "deleting")
    fake_db.delete_user.assert_awaited_once_with("user-X")


@pytest.mark.asyncio
async def test_delete_user_cancels_unlisted_importing_generation(monkeypatch):
    """An unlisted import blocks account removal and cannot later commit."""
    from backend.services import user_cleanup

    session_id = "importing-session"
    records = {
        session_id: {
            "session_id": session_id,
            "owner_id": "user-X",
            "generation": 1,
            "status": "importing",
        }
    }

    def owned_records(_user_id):
        return {sid: dict(record) for sid, record in records.items()}

    def request_cancel(sid, owner_id):
        assert owner_id == "user-X"
        records[sid].update({
            "import_cancel_requested": True,
        })
        return 1

    jobs = {
        "job-1": {
            "user_id": "user-X",
            "status": "extracting",
            "session_id": session_id,
            "cancel_requested": False,
        }
    }
    monkeypatch.setattr(user_cleanup, "_owned_lifecycle_records", owned_records)
    monkeypatch.setattr(
        user_cleanup,
        "request_import_generation_cancellation",
        request_cancel,
    )
    monkeypatch.setattr(user_cleanup, "list_saved_sessions", lambda: [])
    monkeypatch.setattr(user_cleanup, "import_jobs", jobs)
    monkeypatch.setattr(user_cleanup, "import_jobs_lock", asyncio.Lock())
    monkeypatch.setattr(user_cleanup, "import_generation_tasks", {})
    monkeypatch.setattr(user_cleanup, "users_pending_deletion", set())
    monkeypatch.setattr(asyncio, "to_thread", _inline_to_thread)

    fake_db = MagicMock()
    fake_db.set_user_status = AsyncMock(return_value=True)
    fake_db.delete_api_keys_by_user = AsyncMock()
    fake_db.delete_user_llm_config = AsyncMock()
    fake_db.delete_user = AsyncMock()
    monkeypatch.setattr(user_cleanup, "get_db", AsyncMock(return_value=fake_db))

    with pytest.raises(user_cleanup.UserCleanupPendingError) as caught:
        await user_cleanup.delete_user_data("user-X")

    assert jobs["job-1"]["cancel_requested"] is True
    assert caught.value.pending_sessions == [session_id]
    assert caught.value.pending_import_jobs == ["job-1"]
    fake_db.set_user_status.assert_awaited_once_with("user-X", "deleting")
    fake_db.delete_api_keys_by_user.assert_awaited_once_with("user-X")
    fake_db.delete_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_user_retries_incomplete_tombstone_before_removing_user(
    monkeypatch,
):
    """A hidden cleanup_complete=False lifecycle record is actively swept."""
    from backend.services import session_deletion, user_cleanup

    session_id = "old-tombstone"
    record = {
        "session_id": session_id,
        "owner_id": "user-X",
        "generation": 3,
        "status": "deleted",
        "cleanup_complete": False,
    }

    monkeypatch.setattr(
        user_cleanup,
        "_owned_lifecycle_records",
        lambda _uid: {session_id: dict(record)},
    )
    monkeypatch.setattr(user_cleanup, "list_saved_sessions", lambda: [])
    monkeypatch.setattr(user_cleanup, "import_jobs", {})
    monkeypatch.setattr(user_cleanup, "import_jobs_lock", asyncio.Lock())
    monkeypatch.setattr(user_cleanup, "import_generation_tasks", {})
    monkeypatch.setattr(user_cleanup, "users_pending_deletion", set())
    monkeypatch.setattr(asyncio, "to_thread", _inline_to_thread)

    async def cleanup(sid, owner_id=None):
        assert sid == session_id
        assert owner_id == "user-X"
        record["cleanup_complete"] = True
        return {
            "deleted_sessions": 1,
            "deleted_state_files": 2,
            "pending_producers": 0,
        }

    monkeypatch.setattr(session_deletion, "cleanup_tombstoned_generation", cleanup)

    fake_db = MagicMock()
    fake_db.set_user_status = AsyncMock(return_value=True)
    fake_db.delete_api_keys_by_user = AsyncMock()
    fake_db.delete_user_llm_config = AsyncMock()
    fake_db.delete_user = AsyncMock()
    monkeypatch.setattr(user_cleanup, "get_db", AsyncMock(return_value=fake_db))

    result = await user_cleanup.delete_user_data("user-X")

    assert result["deleted_sessions"] == 1
    assert result["deleted_state_files"] == 2
    fake_db.delete_user.assert_awaited_once_with("user-X")


@pytest.mark.asyncio
async def test_delete_user_retires_cancelled_import_left_by_prior_process(
    monkeypatch,
):
    """A durable import cancel marker is safely cleanable after restart."""
    from backend.services import session_deletion, user_cleanup

    session_id = "cancelled-import-leftover"
    record = {
        "session_id": session_id,
        "owner_id": "user-X",
        "generation": 2,
        "status": "importing",
        "import_cancel_requested": True,
    }
    monkeypatch.setattr(
        user_cleanup,
        "_owned_lifecycle_records",
        lambda _uid: {session_id: dict(record)},
    )
    monkeypatch.setattr(user_cleanup, "list_saved_sessions", lambda: [])
    monkeypatch.setattr(user_cleanup, "import_jobs", {})
    monkeypatch.setattr(user_cleanup, "import_jobs_lock", asyncio.Lock())
    monkeypatch.setattr(user_cleanup, "import_generation_tasks", {})
    monkeypatch.setattr(user_cleanup, "users_pending_deletion", set())
    monkeypatch.setattr(asyncio, "to_thread", _inline_to_thread)

    def cleanup_import(sid):
        assert sid == session_id
        record.update({"status": "deleted", "cleanup_complete": True})
        return True

    monkeypatch.setattr(
        session_deletion,
        "cleanup_failed_import_generation",
        cleanup_import,
    )

    fake_db = MagicMock()
    fake_db.set_user_status = AsyncMock(return_value=True)
    fake_db.delete_api_keys_by_user = AsyncMock()
    fake_db.delete_user_llm_config = AsyncMock()
    fake_db.delete_user = AsyncMock()
    monkeypatch.setattr(user_cleanup, "get_db", AsyncMock(return_value=fake_db))

    result = await user_cleanup.delete_user_data("user-X")

    assert result["user_id"] == "user-X"
    fake_db.delete_user.assert_awaited_once_with("user-X")


@pytest.mark.asyncio
async def test_delete_user_continues_after_one_session_producer_stop_error(
    monkeypatch,
):
    """One stalled session does not prevent independent sessions progressing."""
    from backend.models.schemas import AnalysisState
    from backend.services import session_deletion, user_cleanup

    states = {
        sid: AnalysisState(id=sid, user_id="user-X")
        for sid in ("session-a", "session-b")
    }
    records = {
        sid: {
            "session_id": sid,
            "owner_id": "user-X",
            "generation": 1,
            "status": "active",
        }
        for sid in states
    }

    monkeypatch.setattr(
        user_cleanup,
        "_owned_lifecycle_records",
        lambda _uid: {sid: dict(record) for sid, record in records.items()},
    )
    monkeypatch.setattr(
        user_cleanup,
        "list_saved_sessions",
        lambda: [{"id": sid, "user_id": "user-X"} for sid in states],
    )
    monkeypatch.setattr(user_cleanup, "load_session_from_disk", states.get)
    monkeypatch.setattr(user_cleanup, "import_jobs", {})
    monkeypatch.setattr(user_cleanup, "import_jobs_lock", asyncio.Lock())
    monkeypatch.setattr(user_cleanup, "import_generation_tasks", {})
    monkeypatch.setattr(user_cleanup, "users_pending_deletion", set())
    monkeypatch.setattr(asyncio, "to_thread", _inline_to_thread)

    attempted = []

    async def delete_generation(sid, _state, _owner, *, force_stop=False):
        attempted.append(sid)
        assert force_stop is True
        records[sid].update({"status": "deleted", "cleanup_complete": sid == "session-b"})
        if sid == "session-a":
            raise session_deletion.SessionProducerStopError("producer still running")
        return {"deleted_sessions": 1, "deleted_state_files": 1}

    monkeypatch.setattr(session_deletion, "delete_session_generation", delete_generation)

    fake_db = MagicMock()
    fake_db.set_user_status = AsyncMock(return_value=True)
    fake_db.delete_api_keys_by_user = AsyncMock()
    fake_db.delete_user_llm_config = AsyncMock()
    fake_db.delete_user = AsyncMock()
    monkeypatch.setattr(user_cleanup, "get_db", AsyncMock(return_value=fake_db))

    with pytest.raises(user_cleanup.UserCleanupPendingError) as caught:
        await user_cleanup.delete_user_data("user-X")

    assert attempted == ["session-a", "session-b"]
    assert caught.value.pending_sessions == ["session-a"]
    assert "session:session-a" in caught.value.failures
    fake_db.delete_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_delete_reports_retryable_cleanup_conflict(monkeypatch):
    """The admin API exposes pending cleanup as a retryable 409 contract."""
    from fastapi import HTTPException
    from backend.routes import admin
    from backend.services.user_cleanup import UserCleanupPendingError

    target = MagicMock(role="user")
    fake_db = MagicMock()
    fake_db.get_user_by_id = AsyncMock(return_value=target)
    fake_db.mark_user_deleting_if_not_last_admin = AsyncMock(
        return_value="updated"
    )
    monkeypatch.setattr(admin, "get_db", AsyncMock(return_value=fake_db))
    pending = UserCleanupPendingError(
        "user-X",
        pending_sessions=["session-a"],
        pending_import_jobs=["job-1"],
        failures={"session:session-a": "producer still running"},
    )
    monkeypatch.setattr(admin, "delete_user_data", AsyncMock(side_effect=pending))

    with pytest.raises(HTTPException) as caught:
        await admin.delete_user("user-X", admin_id="admin-X")

    assert caught.value.status_code == 409
    assert caught.value.detail["code"] == "user_cleanup_pending"
    assert caught.value.detail["retryable"] is True


@pytest.mark.asyncio
async def test_pending_account_cleanup_revokes_live_websockets_first(monkeypatch):
    """A resistant producer may delay erasure but cannot retain a transport."""
    from types import SimpleNamespace

    from backend.services import session_deletion, user_cleanup
    from backend.utils.websocket_delivery import (
        register_analysis_websocket,
        register_report_websocket,
        reset_websocket_delivery_for_tests,
        send_connection_json,
    )

    class _Socket:
        def __init__(self):
            self.closed = []
            self.sent = []

        async def close(self, code=1000):
            self.closed.append(code)

        async def send_json(self, payload):
            self.sent.append(payload)

    session_id = "resistant-session"
    state = SimpleNamespace(
        id=session_id,
        user_id="user-X",
        lifecycle_generation=1,
    )
    records = {
        session_id: {
            "session_id": session_id,
            "owner_id": "user-X",
            "generation": 1,
            "status": "active",
        }
    }
    monkeypatch.setattr(
        user_cleanup,
        "_owned_lifecycle_records",
        lambda _uid: {sid: dict(record) for sid, record in records.items()},
    )
    monkeypatch.setattr(user_cleanup, "sessions", {session_id: state})
    monkeypatch.setattr(user_cleanup, "state_lock", asyncio.Lock())
    monkeypatch.setattr(user_cleanup, "list_saved_sessions", lambda: [])
    monkeypatch.setattr(user_cleanup, "import_jobs", {})
    monkeypatch.setattr(user_cleanup, "import_jobs_lock", asyncio.Lock())
    monkeypatch.setattr(user_cleanup, "import_generation_tasks", {})
    monkeypatch.setattr(user_cleanup, "users_pending_deletion", set())
    monkeypatch.setattr(asyncio, "to_thread", _inline_to_thread)

    async def resistant_delete(*_args, **_kwargs):
        raise session_deletion.SessionProducerStopError("producer resisted")

    monkeypatch.setattr(
        session_deletion, "delete_session_generation", resistant_delete,
    )
    fake_db = MagicMock()
    fake_db.set_user_status = AsyncMock(return_value=True)
    fake_db.delete_api_keys_by_user = AsyncMock()
    fake_db.delete_user_llm_config = AsyncMock()
    fake_db.delete_user = AsyncMock()
    monkeypatch.setattr(user_cleanup, "get_db", AsyncMock(return_value=fake_db))

    reset_websocket_delivery_for_tests()
    analysis_socket = _Socket()
    report_socket = _Socket()
    analysis = await register_analysis_websocket(session_id, analysis_socket)
    report = await register_report_websocket(session_id, report_socket)
    try:
        with pytest.raises(user_cleanup.UserCleanupPendingError):
            await user_cleanup.delete_user_data("user-X")

        assert analysis_socket.closed == [4001]
        assert report_socket.closed == [4001]
        assert await send_connection_json(analysis, {"late": True}) is False
        assert await send_connection_json(report, {"late": True}) is False
        fake_db.delete_user.assert_not_awaited()
    finally:
        reset_websocket_delivery_for_tests()


@pytest.mark.asyncio
async def test_admin_cannot_reactivate_deleting_user(monkeypatch):
    """A persisted deletion state is not an approval state."""
    from types import SimpleNamespace
    from fastapi import HTTPException
    from backend.routes import admin

    target = SimpleNamespace(id="user-X", role="user", status="deleting")
    fake_db = MagicMock()
    fake_db.get_user_by_id = AsyncMock(return_value=target)
    fake_db.set_user_status = AsyncMock(return_value=True)
    monkeypatch.setattr(admin, "get_db", AsyncMock(return_value=fake_db))

    with pytest.raises(HTTPException) as caught:
        await admin.set_user_status(
            "user-X",
            admin.SetStatusRequest(status="active"),
            admin_id="admin-X",
        )

    assert caught.value.status_code == 409
    assert "deletion is in progress" in str(caught.value.detail).lower()
    fake_db.set_user_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_status_update_loses_race_to_deletion(monkeypatch):
    """The database predicate prevents a stale approval from reviving deletion."""
    from types import SimpleNamespace
    from fastapi import HTTPException
    from backend.routes import admin

    pending = SimpleNamespace(id="user-X", role="user", status="pending")
    deleting = SimpleNamespace(id="user-X", role="user", status="deleting")
    fake_db = MagicMock()
    fake_db.get_user_by_id = AsyncMock(side_effect=[pending, deleting])
    fake_db.set_user_status = AsyncMock(return_value=False)
    fake_db.is_usable_admin = AsyncMock(return_value=True)
    monkeypatch.setattr(admin, "get_db", AsyncMock(return_value=fake_db))

    with pytest.raises(HTTPException) as caught:
        await admin.set_user_status(
            "user-X",
            admin.SetStatusRequest(status="active"),
            admin_id="admin-X",
        )

    assert caught.value.status_code == 409
    fake_db.set_user_status.assert_awaited_once_with(
        "user-X",
        "active",
        unless_current_status="deleting",
        forbid_current_role=None,
        required_admin_id="admin-X",
    )


@pytest.mark.asyncio
async def test_admin_password_update_loses_race_to_deletion(monkeypatch):
    from types import SimpleNamespace
    from fastapi import HTTPException
    from backend.routes import admin

    active = SimpleNamespace(id="user-X", role="user", status="active")
    deleting = SimpleNamespace(id="user-X", role="user", status="deleting")
    fake_db = MagicMock()
    fake_db.get_user_by_id = AsyncMock(side_effect=[active, deleting])
    fake_db.update_user_password = AsyncMock(return_value=False)
    fake_db.is_usable_admin = AsyncMock(return_value=True)
    monkeypatch.setattr(admin, "get_db", AsyncMock(return_value=fake_db))
    monkeypatch.setattr(admin, "hash_password", lambda _password: "hash")

    with pytest.raises(HTTPException) as caught:
        await admin.set_user_password(
            "user-X",
            admin.SetPasswordRequest(new_password="valid-password"),
            admin_id="admin-X",
        )

    assert caught.value.status_code == 409
    assert "deletion is in progress" in str(caught.value.detail).lower()
    fake_db.update_user_password.assert_awaited_once_with(
        "user-X",
        "hash",
        unless_current_status="deleting",
        required_admin_id="admin-X",
    )


@pytest.mark.asyncio
async def test_self_service_password_update_rejects_stale_verified_hash(
    monkeypatch,
):
    from types import SimpleNamespace

    from fastapi import HTTPException

    from backend.routes import auth

    original = SimpleNamespace(
        id="user-X", status="active", password_hash="old-hash",
    )
    reset = SimpleNamespace(
        id="user-X", status="active", password_hash="admin-reset",
    )
    fake_db = MagicMock()
    fake_db.get_user_by_id = AsyncMock(side_effect=[original, reset])
    fake_db.compare_and_swap_user_password = AsyncMock(return_value=False)
    monkeypatch.setattr(auth, "get_db", AsyncMock(return_value=fake_db))
    monkeypatch.setattr(auth, "verify_password", lambda *_args: True)
    monkeypatch.setattr(auth, "hash_password", lambda _password: "self-hash")

    with pytest.raises(HTTPException) as caught:
        await auth.update_password(
            auth.ChangePasswordRequest(
                current_password="old-password",
                new_password="new-password",
            ),
            user_id="user-X",
        )

    assert caught.value.status_code == 409
    fake_db.compare_and_swap_user_password.assert_awaited_once_with(
        "user-X",
        "old-hash",
        "self-hash",
        allowed_statuses=("active", "readonly"),
    )


@pytest.mark.asyncio
async def test_admin_demotion_loses_race_to_deletion(monkeypatch):
    from types import SimpleNamespace
    from fastapi import HTTPException
    from backend.routes import admin

    active = SimpleNamespace(id="user-X", role="admin", status="active")
    deleting = SimpleNamespace(id="user-X", role="admin", status="deleting")
    fake_db = MagicMock()
    fake_db.get_user_by_id = AsyncMock(side_effect=[active, deleting])
    fake_db.demote_admin_if_not_last = AsyncMock(return_value=False)
    fake_db.is_usable_admin = AsyncMock(return_value=True)
    monkeypatch.setattr(admin, "get_db", AsyncMock(return_value=fake_db))

    with pytest.raises(HTTPException) as caught:
        await admin.set_user_role(
            "user-X",
            admin.SetRoleRequest(role="user"),
            admin_id="admin-X",
        )

    assert caught.value.status_code == 409
    assert "deletion is in progress" in str(caught.value.detail).lower()
    fake_db.demote_admin_if_not_last.assert_awaited_once_with(
        "user-X",
        "user",
        unless_current_status="deleting",
        required_admin_id="admin-X",
    )


@pytest.mark.asyncio
async def test_deleting_user_cannot_log_in(monkeypatch):
    """The durable deletion state revokes password-based bearer issuance."""
    from types import SimpleNamespace
    from fastapi import HTTPException
    from starlette.requests import Request
    from backend.routes import auth

    target = SimpleNamespace(
        id="user-X",
        username="cleanup-target",
        email=None,
        role="user",
        status="deleting",
        password_hash="hash",
    )
    fake_db = MagicMock()
    fake_db.get_user_by_username = AsyncMock(return_value=target)
    monkeypatch.setattr(auth, "get_db", AsyncMock(return_value=fake_db))
    monkeypatch.setattr(auth, "verify_password", lambda *_args: True)

    request = Request({"type": "http", "client": ("127.0.0.1", 1234)})
    with pytest.raises(HTTPException) as caught:
        await auth.login(
            auth.LoginRequest(username="cleanup-target", password="secret"),
            request,
        )

    assert caught.value.status_code == 403
    assert caught.value.detail == "Account deletion is in progress."


@pytest.mark.asyncio
async def test_inline_import_refuses_generation_claim_after_account_revocation(
    monkeypatch,
):
    """A pre-authenticated inline request cannot outrun account cleanup."""
    from backend.routes.export_import_parts import import_routes

    class Request:
        headers = {
            "x-filename": "session.json",
            "content-length": "2",
        }
        state = MagicMock(user_id="user-X")
        client = None

        async def stream(self):
            yield b"{}"

    monkeypatch.setattr(import_routes, "users_pending_deletion", {"user-X"})

    with pytest.raises(Exception) as caught:
        await import_routes.import_state(Request(), None)

    assert getattr(caught.value, "status_code", None) == 409
    assert "deletion is in progress" in str(caught.value.detail)


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
        total_size=1024,
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
        total_size=1024,
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
        total_size=1024,
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
