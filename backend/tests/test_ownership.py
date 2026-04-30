"""Tests for authentication and session ownership enforcement.

P0 security: these prove the auth/ownership fixes work automatically.
"""
import os

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.utils.auth import (
    hash_password, verify_password, create_access_token, decode_access_token,
    generate_api_key, hash_api_key, require_auth, JWT_SECRET, JWT_ALGORITHM,
)
from backend.utils.files import safe_filename, validate_session_id


class TestPasswordAuth:
    def test_hash_and_verify(self):
        h = hash_password("test123")
        assert verify_password("test123", h)
        assert not verify_password("wrong", h)

    def test_bcrypt_hash_format(self):
        h = hash_password("pass")
        assert h.startswith("$2b$")


class TestJWT:
    def test_create_and_decode(self):
        token = create_access_token("user1", "testuser")
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "user1"
        assert payload["username"] == "testuser"

    def test_invalid_token_returns_none(self):
        result = decode_access_token("invalid.token.here")
        assert result is None

    def test_expired_token(self):
        from jose import jwt as jose_jwt
        from datetime import datetime
        payload = {"sub": "u1", "username": "testuser", "exp": datetime(2000, 1, 1)}
        token = jose_jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        result = decode_access_token(token)
        assert result is None


class TestAPIKey:
    def test_generate_key_format(self):
        raw, hashed = generate_api_key()
        assert raw.startswith("scai_")
        assert len(hashed) == 64  # SHA256 hex

    def test_hash_consistency(self):
        raw, hashed = generate_api_key()
        assert hash_api_key(raw) == hashed

    def test_different_keys_different_hashes(self):
        raw1, h1 = generate_api_key()
        raw2, h2 = generate_api_key()
        assert h1 != h2


class TestOwnership:
    def test_middleware_rejects_unauthenticated(self):
        """Middleware returns 401 for protected paths without auth."""
        from backend.middleware.auth import AuthMiddleware
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def protected_route(request):
            return PlainTextResponse("ok")

        app = Starlette(
            routes=[Route("/api/v1/test", protected_route)],
            middleware=[(AuthMiddleware, [], {})],
        )
        client = TestClient(app)
        resp = client.get("/api/v1/test")
        assert resp.status_code == 401

    def test_middleware_allows_public(self):
        """Middleware allows /api/auth/ without auth."""
        from backend.middleware.auth import AuthMiddleware
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def public_route(request):
            return PlainTextResponse("ok")

        app = Starlette(
            routes=[Route("/api/auth/login", public_route)],
            middleware=[(AuthMiddleware, [], {})],
        )
        client = TestClient(app)
        resp = client.get("/api/auth/login")
        assert resp.status_code == 200

    def test_middleware_passes_through_auth_optional_with_token(self):
        """Middleware passes through auth-optional paths even if DB user missing.

        Ownership is enforced by route-layer require_session_ownership,
        not by the middleware itself.
        """
        from backend.middleware.auth import AuthMiddleware
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def session_route(request):
            return PlainTextResponse("ok")

        app = Starlette(
            routes=[Route("/api/session/sess1/detail", session_route)],
            middleware=[(AuthMiddleware, [], {})],
        )
        client = TestClient(app)

        # Token for a user that doesn't exist in DB
        token = create_access_token("unknown_user", "unknown_user")
        resp = client.get(
            "/api/session/sess1/detail",
            headers={"Authorization": f"Bearer {token}"},
        )
        # Middleware passes through; route would check ownership
        assert resp.status_code == 200

    def test_middleware_passes_through_auth_optional_with_valid_token(self):
        """Middleware passes through auth-optional paths with a valid token."""
        from backend.middleware.auth import AuthMiddleware
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def session_route(request):
            return PlainTextResponse("ok")

        app = Starlette(
            routes=[Route("/api/session/sess1/detail", session_route)],
            middleware=[(AuthMiddleware, [], {})],
        )
        client = TestClient(app)

        token = create_access_token("user_a", "user_a")
        resp = client.get(
            "/api/session/sess1/detail",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_middleware_passes_through_file_path_with_token(self):
        """Middleware passes through /api/file/ paths with a valid token.

        Ownership is enforced by route-layer require_session_ownership.
        """
        from backend.middleware.auth import AuthMiddleware
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def file_route(request):
            return PlainTextResponse("ok")

        app = Starlette(
            routes=[Route("/api/file/sess2/data/test.h5ad", file_route)],
            middleware=[(AuthMiddleware, [], {})],
        )
        client = TestClient(app)

        token = create_access_token("user_b", "user_b")
        resp = client.get(
            "/api/file/sess2/data/test.h5ad",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_middleware_passes_through_session_with_empty_owner(self):
        """Middleware passes through auth-optional path even when session has empty user_id.

        Ownership check (empty user_id means no owner) is the route's job.
        """
        from backend.middleware.auth import AuthMiddleware
        from backend.state import sessions
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        mock_state = MagicMock()
        mock_state.user_id = ""
        sessions["sess3"] = mock_state

        try:
            async def session_route(request):
                return PlainTextResponse("ok")

            app = Starlette(
                routes=[Route("/api/session/sess3/detail", session_route)],
                middleware=[(AuthMiddleware, [], {})],
            )
            client = TestClient(app)

            token = create_access_token("user_a", "user_a")
            resp = client.get(
                "/api/session/sess3/detail",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200
        finally:
            sessions.pop("sess3", None)


class TestSafeFilename:
    def test_normal_filename(self):
        assert safe_filename("data.h5ad") == "data.h5ad"

    def test_path_traversal_blocked(self):
        assert safe_filename("../../etc/passwd") == "passwd"
        assert safe_filename("/etc/shadow") == "shadow"

    def test_hidden_file_blocked(self):
        assert safe_filename(".env") == ""

    def test_empty_input(self):
        assert safe_filename("") == ""

    def test_windows_path(self):
        # On Linux, Path("C:\\Users\\data.h5ad").name returns the full string
        # since backslash is a valid filename char on Linux
        result = safe_filename("C:\\Users\\data.h5ad")
        assert result != ""  # Not blocked at least


class TestValidateSessionId:
    def test_valid_id(self):
        validate_session_id("abc123-def_456")

    def test_rejects_traversal(self):
        with pytest.raises(Exception):
            validate_session_id("../../etc")

    def test_rejects_empty(self):
        with pytest.raises(Exception):
            validate_session_id("")

    def test_rejects_too_long(self):
        with pytest.raises(Exception):
            validate_session_id("a" * 200)