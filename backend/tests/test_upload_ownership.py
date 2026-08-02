"""P0-1: chunk upload and status endpoints must enforce session ownership.

Without these checks, a user who learns another user's upload_id can write
chunks into the victim's temp directory or read upload progress.
"""
import asyncio
import logging
import re
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.models.schemas import AnalysisState
from backend.state import sessions
from backend.utils import auth as auth_module
from backend.utils.auth import create_access_token, hash_password
from backend.utils.db import AppDB


@pytest.fixture
def client(tmp_path):
    """Test client backed by an isolated AppDB, mirroring test_auth_flow.py."""
    old_db = auth_module._db

    db_path = tmp_path / "test_upload_ownership.db"
    new_db = AppDB(db_path)

    loop = asyncio.new_event_loop()
    loop.run_until_complete(new_db.init())
    loop.close()

    auth_module._db = new_db

    from backend.main import app
    test_client = TestClient(app)
    yield test_client

    auth_module._db = old_db


@pytest.fixture
def users_ab(client):
    """Create user_a and user_b in the isolated auth DB."""
    db = auth_module._db
    pw = hash_password("x")

    loop = asyncio.new_event_loop()
    uid_a = loop.run_until_complete(
        db.create_user("user_a", pw, role="user", status="active")
    )
    uid_b = loop.run_until_complete(
        db.create_user("user_b", pw, role="user", status="active")
    )
    loop.close()
    return uid_a, uid_b


@pytest.fixture
def user_a_session(users_ab):
    """Register an in-memory session owned by user_a."""
    uid_a, _ = users_ab
    # Lifecycle ownership records are intentionally durable.  Reusing one
    # fixed ID while the auth fixture creates a new random owner per test makes
    # later cases look like a cross-owner resurrection attempt.
    session_id = f"test-upload-own-{uuid.uuid4().hex}"
    state = AnalysisState(id=session_id, user_id=uid_a, name="A")
    sessions[session_id] = state
    yield session_id, uid_a, users_ab[1]
    sessions.pop(session_id, None)


def _token(user_id: str) -> str:
    return create_access_token(user_id, user_id)


def test_upload_chunk_rejects_non_owner(client, user_a_session):
    """User B uploading a chunk to user A's session must be rejected (404 leak-oracle safe)."""
    session_id, _uid_a, uid_b = user_a_session
    upload_id = "any-upload-id"
    headers = {"Authorization": f"Bearer {_token(uid_b)}"}
    files = {"file": ("chunk.bin", b"hello", "application/octet-stream")}
    resp = client.post(
        f"/api/session/{session_id}/upload/chunk/{upload_id}/0",
        headers=headers,
        files=files,
    )
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_upload_status_rejects_non_owner(client, user_a_session):
    """User B requesting upload status on user A's session must be rejected (404).

    Use a *real* upload_id (init via upload_manager) so that without the fix
    the request would succeed with 200; with the fix, ownership check fires first.
    """
    from backend.services.upload import upload_manager
    session_id, _uid_a, uid_b = user_a_session
    upload_id = await upload_manager.init_upload(
        session_id=session_id,
        filename="data.h5ad",
        total_size=100,
        total_chunks=1,
        file_type="snrna_seq",
        group="control",
    )
    try:
        headers = {"Authorization": f"Bearer {_token(uid_b)}"}
        resp = client.get(
            f"/api/session/{session_id}/upload/status/{upload_id}",
            headers=headers,
        )
        assert resp.status_code == 404, resp.text
    finally:
        await upload_manager.cancel_upload(upload_id)


def test_upload_chunk_owner_passes_ownership_check(client, user_a_session):
    """Owner reaches the manager, which hides an unknown upload as not found."""
    session_id, uid_a, _ = user_a_session
    upload_id = "unknown-upload-id"
    headers = {"Authorization": f"Bearer {_token(uid_a)}"}
    files = {"file": ("chunk.bin", b"hello", "application/octet-stream")}
    resp = client.post(
        f"/api/session/{session_id}/upload/chunk/{upload_id}/0",
        headers=headers,
        files=files,
    )
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_cancel_chunked_upload_owner_uses_current_generation(
    client, user_a_session,
):
    """The owner can cancel under the lifecycle barrier without NameError."""
    from backend.services.upload import upload_manager

    session_id, uid_a, _ = user_a_session
    upload_id = await upload_manager.init_upload(
        session_id=session_id,
        filename="cancel-me.h5ad",
        total_size=100,
        total_chunks=1,
        file_type="snrna_seq",
        group="control",
    )
    headers = {"Authorization": f"Bearer {_token(uid_a)}"}

    resp = client.delete(
        f"/api/session/{session_id}/upload/cancel/{upload_id}",
        headers=headers,
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "success": True,
        "message": "Upload cancelled",
    }


def test_upload_data_validation_has_safe_error_code_and_request_id(
    client, user_a_session, caplog
):
    session_id, uid_a, _ = user_a_session
    headers = {"Authorization": f"Bearer {_token(uid_a)}"}
    files = {
        "file": (
            "../../private-study.h5ad",
            b"secret-upload-body",
            "application/octet-stream",
        )
    }
    data = {"section": "invalid\nsection", "data_type": "single-cell"}

    with caplog.at_level(logging.WARNING, logger="backend.routes.upload"):
        resp = client.post(
            f"/api/session/{session_id}/upload/data",
            headers=headers,
            files=files,
            data=data,
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == (
        "Section must be 'pathology', 'sequencing', or 'biochemistry'"
    )
    assert resp.headers["x-scai-error-code"] == "UPLOAD_INVALID_SECTION"
    assert re.fullmatch(r"[0-9a-f]{32}", resp.headers["x-request-id"])
    assert f"request_id={resp.headers['x-request-id']}" in caplog.text
    assert "stage=validation status=400" in caplog.text
    assert "invalid_section" in caplog.text
    assert "private-study.h5ad" in caplog.text
    assert "../../private-study.h5ad" not in caplog.text
    assert "secret-upload-body" not in caplog.text
