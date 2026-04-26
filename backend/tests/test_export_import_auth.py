"""Regression test for the import 401 bug.

Background: POST /api/session/import was returning 401 because the frontend
XHR upload omitted the Authorization header. The fix is in the frontend
(attach the Bearer token), but this test pins the backend contract: no
token → 401, so the auth-optional middleware branch must not be relaxed
to silently accept anonymous imports.
"""
import io

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_import_without_auth_returns_401(client):
    # Minimal multipart payload: tiny tar.gz stub is fine — auth check happens first.
    files = {"file": ("session.tar.gz", io.BytesIO(b"\x1f\x8b\x08\x00" + b"\x00" * 8), "application/gzip")}
    resp = client.post("/api/session/import", files=files)
    assert resp.status_code == 401
    body = resp.json()
    assert "auth" in body.get("detail", "").lower()
