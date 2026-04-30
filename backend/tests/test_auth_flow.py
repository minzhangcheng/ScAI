"""Integration tests for username-based auth flow: setup, register, login."""
import pytest
from fastapi.testclient import TestClient

from backend.utils import auth as auth_module
from backend.utils.db import AppDB


@pytest.fixture
def client(tmp_path):
    """Create a test client with a fresh database."""
    # Save the old singleton
    old_db = auth_module._db

    # Create a fresh AppDB pointing to the temp directory and assign it as the singleton
    db_path = tmp_path / "test_auth.db"
    new_db = AppDB(db_path)

    # We need to init the DB synchronously for the TestClient
    import asyncio
    loop = asyncio.new_event_loop()
    loop.run_until_complete(new_db.init())
    loop.close()

    # Override the singleton so get_db() returns our test DB
    auth_module._db = new_db

    from backend.main import app
    client = TestClient(app)
    yield client

    # Restore
    auth_module._db = old_db


class TestAuthFlow:
    def test_setup_register_login(self, client):
        """Full flow: setup admin, register user, login."""
        # 1. Setup admin (no email)
        resp = client.post("/api/auth/setup/admin", json={"password": "adminpass123"})
        assert resp.status_code == 201

        # 2. Login as admin with username
        resp = client.post("/api/auth/login", json={"username": "admin", "password": "adminpass123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "admin"
        assert data["role"] == "admin"
        assert data["access_token"] != ""

        # 3. Register a user with username only (no email)
        resp = client.post("/api/auth/register", json={"username": "testuser", "password": "userpass123"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "testuser"
        assert data["status"] == "pending"
        assert data["access_token"] == ""  # Pending users don't get tokens

        # 4. Duplicate username should fail
        resp = client.post("/api/auth/register", json={"username": "testuser", "password": "otherpass123"})
        assert resp.status_code == 409

        # 5. Pending user cannot login
        resp = client.post("/api/auth/login", json={"username": "testuser", "password": "userpass123"})
        assert resp.status_code == 403

        # 6. Wrong password should fail
        resp = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    def test_setup_with_email(self, client):
        """Setup admin with optional email."""
        resp = client.post("/api/auth/setup/admin", json={"password": "adminpass123", "email": "admin@example.com"})
        assert resp.status_code == 201

        resp = client.post("/api/auth/login", json={"username": "admin", "password": "adminpass123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "admin@example.com"

    def test_register_with_email(self, client):
        """Register a user with both username and email."""
        client.post("/api/auth/setup/admin", json={"password": "adminpass123"})

        resp = client.post("/api/auth/register", json={
            "username": "user2",
            "password": "userpass123",
            "email": "user2@example.com",
        })
        assert resp.status_code == 201
        assert resp.json()["email"] == "user2@example.com"

    def test_username_validation(self, client):
        """Username must be 3-64 chars, alphanumeric/underscore/hyphen only."""
        client.post("/api/auth/setup/admin", json={"password": "adminpass123"})

        # Too short
        resp = client.post("/api/auth/register", json={"username": "ab", "password": "userpass123"})
        assert resp.status_code == 422

        # Invalid characters
        resp = client.post("/api/auth/register", json={"username": "user@name", "password": "userpass123"})
        assert resp.status_code == 422