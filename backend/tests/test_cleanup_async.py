"""C5 — cleanup uses asyncio.to_thread; ttl query covers completed leftovers."""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock


async def _inline_to_thread(func, *args, **kwargs):
    """Run a thread-boundary call inline on hosts with a blocked executor."""
    return func(*args, **kwargs)


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
