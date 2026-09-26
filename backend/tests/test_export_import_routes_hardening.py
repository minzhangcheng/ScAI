"""Route- and wiring-level guards for the export/import hardening pass.

These pin behaviour a user meets directly: which endpoints authenticate, what a
mis-named upload reports, whether an imported session can be started again, and
whether a credential-free custom endpoint can capture the installation's key.
"""
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


# --- H7: only the token-redeeming leg may skip authentication --------------


# --- M8: prepared archives nobody downloaded are reclaimed -----------------


@pytest.mark.asyncio
async def test_stale_prepared_exports_are_deleted(tmp_path):
    import asyncio

    from backend.config import PREPARED_EXPORT_TTL
    from backend.services.export_lifecycle import (
        create_export_ownership_marker,
    )
    from backend.tasks.session_cleanup import _cleanup_stale_prepared_exports

    fresh = tmp_path / "fresh.tar.gz"
    stale = tmp_path / "stale.tar.gz"
    fresh.write_bytes(b"a")
    stale.write_bytes(b"b")
    fresh_marker = create_export_ownership_marker(fresh)
    stale_marker = create_export_ownership_marker(stale)
    now = 1_000_000.0
    exports = {
        "fresh": {
            "token": "fresh",
            "path": str(fresh),
            "path_owned": True,
            "ownership_marker_required": True,
            "ownership_marker_owned": True,
            "phase": "ready",
            "created_at": now - 10,
        },
        "stale": {
            "token": "stale",
            "path": str(stale),
            "path_owned": True,
            "ownership_marker_required": True,
            "ownership_marker_owned": True,
            "phase": "ready",
            "created_at": now - PREPARED_EXPORT_TTL - 1,
        },
    }

    await _cleanup_stale_prepared_exports(now, exports, asyncio.Lock())

    assert set(exports) == {"fresh"}
    assert fresh.exists()
    assert fresh_marker.exists()
    assert not stale.exists()
    assert not stale_marker.exists()


# --- H4 / L2: what the upload header may carry ----------------------------


# --- H1: an imported session must have a way forward ----------------------


# --- H6: a custom endpoint with no credential must not borrow one ---------


# --- H8: the endpoint key config_io writes is the key the runtime reads ---


# --- M12: ConfigV1 must describe the sessions the product creates ---------


# --- M2: chat delegation respects the same runtime gate -------------------


# --- An import job belongs to the user who started it ---------------------


# --- Extraction stops on a full disk instead of filling it ----------------


def test_import_refuses_to_start_without_free_space(tmp_path, monkeypatch):
    import tarfile

    from fastapi import HTTPException

    from backend.routes.export_import_parts import import_impl

    archive = tmp_path / "any.tar.gz"
    with tarfile.open(archive, mode="w:gz"):
        pass

    monkeypatch.setattr(import_impl, "free_disk_bytes", lambda _p: 1024)

    with pytest.raises(HTTPException) as exc:
        import_impl._import_tar_gz(str(archive), user_id="owner")

    assert exc.value.status_code == 507
    assert "disk space" in str(exc.value.detail).lower()


class TestExportRequiresAuthentication:
    def test_export_without_a_token_is_401(self, client):
        resp = client.get(f"/api/session/{uuid.uuid4()}/export")
        assert resp.status_code == 401


    @pytest.mark.asyncio
    async def test_a_token_is_bound_to_the_session_that_minted_it(self, tmp_path):
        from backend.routes.export_import import export_download
        from backend.state import prepared_exports, prepared_exports_lock

        archive = tmp_path / "other.tar.gz"
        archive.write_bytes(b"payload")
        token = "token-for-session-a"
        async with prepared_exports_lock:
            prepared_exports[token] = {
                "path": str(archive),
                "filename": "a.tar.gz",
                "size": 7,
                "session_id": "session-a",
                "user_id": "owner",
                "created_at": 0.0,
            }
        try:
            request = MagicMock()
            request.state = MagicMock()
            request.state.user_id = None

            with pytest.raises(Exception) as exc:
                await export_download("session-b", request, token=token)
            assert getattr(exc.value, "status_code", None) == 404

            # Redeeming for the wrong session must not consume the entry.
            async with prepared_exports_lock:
                assert token in prepared_exports
        finally:
            async with prepared_exports_lock:
                prepared_exports.pop(token, None)


class TestImportFilenameHandling:


    @staticmethod
    def _request(filename, body=b"{}"):
        """The route's own view of a request, without the auth stack in front.

        The sibling test that posts through the TestClient never reaches this
        route -- the stub user is not in the database and the answer is 401 --
        so it can only assert "400 or 401" and passes whatever the route does.
        Calling the route is the only way to see what it actually answers.
        """
        import types

        async def _stream():
            yield body
            yield b""

        return types.SimpleNamespace(
            state=types.SimpleNamespace(user_id="user-1"),
            headers={
                "x-filename": filename,
                "content-length": str(len(body)),
            },
            client=None,
            stream=_stream,
        )

    @pytest.mark.asyncio
    async def test_a_large_body_under_a_name_this_route_cannot_read_reports_the_name(
        self, monkeypatch,
    ):
        """The message has to name the cause, and the cause is the name.

        A session export is a gzipped tar. Saved as `.tar` rather than
        `.tar.gz` it fell past the dispatch into the inline branch, where the
        first thing it met was the request-body cap -- so a 1.6 GB export came
        back as "Import file too large (max 104857600 bytes)". The same bytes
        under the same server imported fine once renamed. A caller told the
        file was too large goes looking for a limit to raise, and there is no
        limit to raise: the tar path streams to disk uncapped.
        """
        from fastapi import BackgroundTasks, HTTPException

        from backend.routes.export_import_parts import import_routes

        # Stand in for a body larger than the cap without sending one.
        monkeypatch.setattr(import_routes, "MAX_INLINE_IMPORT_BYTES", 8)

        with pytest.raises(HTTPException) as raised:
            await import_routes.import_state(
                self._request("session.tar", b"x" * 4096), BackgroundTasks(),
            )

        assert raised.value.status_code == 400
        detail = str(raised.value.detail).lower()
        assert "unsupported import format" in detail
        assert "too large" not in detail


class TestForeignEndpointNeverGetsTheLocalKey:
    def _provider(self, agent_cfg):
        from backend.config import SystemConfig
        from backend.models.schemas import AnalysisState
        from backend.utils.settings_provider import AgentSettingsProvider

        state = AnalysisState(id="endpoint", user_id="u1")
        state.agent_server_configs = {"pi": agent_cfg}
        config = SystemConfig()
        config.api.llm_base_url = "https://global.example/v1"
        config.api.api_key = "sk-installation"
        sentinel = object()
        provider = AgentSettingsProvider(
            state, config, global_client=sentinel, session_id="endpoint",
        )
        return provider, sentinel


    def _agent_client(self, agent_cfg):
        from backend.config import SystemConfig
        from backend.models.schemas import AnalysisState
        from backend.utils.agent_llm import get_agent_llm_client

        config = SystemConfig()
        config.api.llm_base_url = "https://global.example/v1"
        config.api.api_key = "sk-installation"
        state = AnalysisState(id="endpoint", user_id="u1")
        state.agent_server_configs = {"pi": agent_cfg}
        return get_agent_llm_client(state, config, "pi", "endpoint")

    def test_agent_llm_empty_custom_key_falls_back_to_global_endpoint(self):
        client = self._agent_client(
            {"llm_base_url": "https://archive.example/v1", "api_key": ""}
        )

        assert client.base_url.startswith("https://global.example")
        assert client.api_key == "sk-installation"


    def test_agent_llm_distinct_own_key_still_uses_custom_endpoint(self):
        client = self._agent_client(
            {"llm_base_url": "https://archive.example/v1", "api_key": "sk-its-own"}
        )

        assert client.base_url.startswith("https://archive.example")
        assert client.api_key == "sk-its-own"


class TestImportJobsAreScopedToTheirOwner:
    @pytest.fixture
    def _job(self):
        from backend.state import import_jobs

        job_id = f"job-{uuid.uuid4()}"
        import_jobs[job_id] = {
            "status": "extracting",
            "phase": "extracting",
            "user_id": "owner",
            "created_at": 0.0,
            "cancel_requested": False,
        }
        yield job_id
        import_jobs.pop(job_id, None)

    def _request(self, user_id):
        request = MagicMock()
        request.state = MagicMock()
        request.state.user_id = user_id
        return request


    @pytest.mark.asyncio
    async def test_another_user_gets_404_not_the_progress(self, _job):
        from backend.routes.export_import import get_import_status

        with pytest.raises(Exception) as exc:
            await get_import_status(_job, self._request("stranger"))
        assert getattr(exc.value, "status_code", None) == 404

    @pytest.mark.asyncio
    async def test_another_user_cannot_cancel(self, _job):
        """X-Import-Id is client-chosen, so an id is not a capability."""
        from backend.routes.export_import import cancel_import
        from backend.state import import_jobs

        with pytest.raises(Exception) as exc:
            await cancel_import(_job, self._request("stranger"))
        assert getattr(exc.value, "status_code", None) == 404
        assert import_jobs[_job]["cancel_requested"] is False

        result = await cancel_import(_job, self._request("owner"))
        assert result["cancelled"] is True
        assert import_jobs[_job]["cancel_requested"] is True
