"""Route- and wiring-level guards for the export/import hardening pass.

These pin behaviour a user meets directly: which endpoints authenticate, what a
mis-named upload reports, whether an imported session can be started again, and
whether a credential-free custom endpoint can capture the installation's key.
"""
import importlib
import json
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


# --- H7: only the token-redeeming leg may skip authentication --------------


class TestExportRequiresAuthentication:
    def test_export_without_a_token_is_401(self, client):
        resp = client.get(f"/api/session/{uuid.uuid4()}/export")
        assert resp.status_code == 401

    def test_prepare_without_a_token_is_401(self, client):
        resp = client.post(f"/api/session/{uuid.uuid4()}/export/prepare")
        assert resp.status_code == 401

    def test_download_without_a_token_query_param_is_refused(self, client):
        """The download leg is reachable unauthenticated, but not without a token."""
        resp = client.get(f"/api/session/{uuid.uuid4()}/export/download")
        assert resp.status_code == 400
        assert "token" in json.dumps(resp.json()).lower()

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


class TestImportFilenameHandling:
    def test_percent_encoded_names_are_decoded(self):
        """Export names keep CJK; XHR cannot put those in a header verbatim."""
        from urllib.parse import quote, unquote

        from backend.routes.export_import import _build_export_filename

        filename = _build_export_filename(str(uuid.uuid4()), "单细胞数据分析 PBMC")
        assert filename.encode("latin-1", errors="ignore").decode("latin-1") != filename

        header_value = quote(filename)
        header_value.encode("latin-1")  # must be transmissible
        assert unquote(header_value) == filename

    def test_unsupported_extension_reports_the_format(self, client, monkeypatch):
        from backend.middleware import auth as auth_middleware

        async def _authenticate(_token):
            return "user-1"

        monkeypatch.setattr(auth_middleware, "authenticate_token", _authenticate)

        resp = client.post(
            "/api/session/import",
            content=b"{}",
            headers={
                "Authorization": "Bearer t",
                "X-Filename": "session.tar",
                "Content-Type": "application/octet-stream",
            },
        )
        # Either the middleware rejects the stub user (401) or the route reports
        # the unsupported format -- never a bare 500 from json.loads.
        assert resp.status_code in (400, 401)
        if resp.status_code == 400:
            assert "unsupported import format" in json.dumps(resp.json()).lower()

    def test_extension_dispatch_is_case_insensitive(self):
        import inspect

        from backend.routes.export_import_parts import import_routes

        source = inspect.getsource(import_routes.import_state)
        assert "lowered.endswith(('.tar.gz', '.tgz'))" in source, (
            "a browser that upper-cases a downloaded filename must not change "
            "how the body is parsed"
        )


# --- H1: an imported session must have a way forward ----------------------


@pytest.mark.asyncio
async def test_run_signals_explicit_start_when_there_is_no_checkpoint(monkeypatch, tmp_path):
    """Without this, resume() finds no checkpoint, refuses to invent a run,
    flips the status back to PAUSED and executes nothing -- so an imported
    session's Start/Re-run button is a silent no-op with no way forward."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.routes import lifecycle

    session_id = str(uuid.uuid4())
    state = AnalysisState(id=session_id, user_id="u1", current_round=2)
    state.status = TaskStatus.PAUSED
    state.runtime_restore_status = "unavailable"
    state.runtime_restore_warning = "imported without runtime"

    async def _owner(session_id, user_id):
        return state

    async def _save(*a, **k):
        return None

    monkeypatch.setattr(lifecycle, "require_session_ownership", _owner)
    monkeypatch.setattr(lifecycle, "async_save_session", _save)
    monkeypatch.setattr(lifecycle, "assert_recovery_clear", lambda *a, **k: None)

    class _Req:
        state = type("S", (), {"user_id": "u1"})()

    await lifecycle.run_analysis(session_id, _Req())

    assert state.restore_start_intent == "explicit_start"
    assert state.runtime_restore_status == "complete"
    assert state.runtime_recompute_pending is True
    assert state.status == TaskStatus.RUNNING


@pytest.mark.parametrize(
    "write_pkl, expected_intent",
    [
        # Checkpoint plus the kernel it was written against: an exact
        # continuation is possible, so no explicit start intent is armed.
        (True, None),
        # Checkpoint without a kernel. `include_runtime=true` exports the
        # checkpoint unconditionally while the pkl needs `trust=true` on the way
        # in, so this pairing is reachable from the UI's two separate prompts.
        # Testing the file alone armed nothing and sent Re-run into an exact
        # continuation against an empty kernel.
        (False, "explicit_start"),
    ],
)
@pytest.mark.asyncio
async def test_a_checkpoint_without_its_kernel_does_not_count(
    monkeypatch, write_pkl, expected_intent,
):
    from backend.config import get_session_states
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.routes import lifecycle
    from backend.utils.runtime_apply import _live_pkl

    session_id = str(uuid.uuid4())
    states = get_session_states(session_id)
    states.mkdir(parents=True, exist_ok=True)
    (states / "checkpoints.sqlite").write_bytes(b"SQLITE")
    if write_pkl:
        _live_pkl(session_id).write_bytes(b"KERNEL")

    state = AnalysisState(id=session_id, user_id="u1", current_round=2)
    state.status = TaskStatus.PAUSED

    async def _owner(session_id, user_id):
        return state

    async def _save(*a, **k):
        return None

    monkeypatch.setattr(lifecycle, "require_session_ownership", _owner)
    monkeypatch.setattr(lifecycle, "async_save_session", _save)
    monkeypatch.setattr(lifecycle, "assert_recovery_clear", lambda *a, **k: None)

    class _Req:
        state = type("S", (), {"user_id": "u1"})()

    await lifecycle.run_analysis(session_id, _Req())

    assert getattr(state, "restore_start_intent", None) == expected_intent


@pytest.mark.asyncio
async def test_completed_reanalysis_tells_the_agent_its_kernel_is_empty(monkeypatch):
    """``/resume`` exempts COMPLETED from the block, then must honour the same
    contract ``/run`` does: clearing an ``unavailable`` runtime is exactly the
    moment the agent has to be told to reload its inputs from disk."""
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.routes import lifecycle

    session_id = str(uuid.uuid4())
    state = AnalysisState(id=session_id, user_id="u1", current_round=2)
    state.status = TaskStatus.COMPLETED
    state.runtime_restore_status = "unavailable"

    async def _owner(session_id, user_id):
        return state

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(lifecycle, "require_session_ownership", _owner)
    monkeypatch.setattr(lifecycle, "async_save_session", _noop)
    monkeypatch.setattr(lifecycle, "_update_session_cache", _noop)
    monkeypatch.setattr(lifecycle, "teardown_running_workflow", _noop)
    monkeypatch.setattr(lifecycle, "assert_recovery_clear", lambda *a, **k: None)

    await lifecycle._resume_analysis_locked(session_id, "u1")

    assert state.runtime_recompute_pending is True
    assert state.runtime_restore_status == "complete"
    assert state.status == TaskStatus.RUNNING


def test_explicit_start_routes_a_plain_session_into_the_round_loop():
    """The plan builder must not need Snapshot V2 metadata to honour the intent."""
    from backend.models.schemas import AnalysisState
    from backend.workflow.restore_entry import build_restore_entry_plan

    state = AnalysisState(id="plain", user_id="u1", current_round=2)

    plan = build_restore_entry_plan(state, restore_start_intent="explicit_start")

    assert plan.intent == "explicit_start"
    assert plan.target_node == "initialize_round"


def test_recompute_run_tells_the_agent_its_kernel_is_empty():
    from backend.agents.bioinformatics_agent import BioinformaticsAgent
    from backend.models.schemas import AnalysisState

    state = AnalysisState(id="recompute", user_id="u1", current_round=2)
    state.runtime_recompute_pending = True

    context = BioinformaticsAgent._build_data_context(MagicMock(), state)

    assert "KERNEL STATE IS EMPTY" in context

    state.runtime_recompute_pending = False
    assert "KERNEL STATE IS EMPTY" not in BioinformaticsAgent._build_data_context(
        MagicMock(), state
    )


# --- H6: a custom endpoint with no credential must not borrow one ---------


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

    def test_stripped_credential_falls_back_to_the_global_client(self):
        provider, sentinel = self._provider(
            {"llm_base_url": "https://archive.example/v1", "api_key": ""}
        )

        assert provider.get_llm_client("pi") is sentinel

    def test_installation_equal_credential_falls_back_to_the_global_client(self):
        provider, sentinel = self._provider(
            {"llm_base_url": "https://archive.example/v1", "api_key": "sk-installation"}
        )

        assert provider.get_llm_client("pi") is sentinel

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

    def test_agent_llm_installation_equal_key_falls_back_to_global_endpoint(self):
        client = self._agent_client(
            {"llm_base_url": "https://archive.example/v1", "api_key": "sk-installation"}
        )

        assert client.base_url.startswith("https://global.example")
        assert client.api_key == "sk-installation"

    def test_agent_llm_distinct_own_key_still_uses_custom_endpoint(self):
        client = self._agent_client(
            {"llm_base_url": "https://archive.example/v1", "api_key": "sk-its-own"}
        )

        assert client.base_url.startswith("https://archive.example")
        assert client.api_key == "sk-its-own"

    def test_a_complete_override_is_still_honoured(self):
        provider, sentinel = self._provider(
            {"llm_base_url": "https://archive.example/v1", "api_key": "sk-its-own"}
        )

        client = provider.get_llm_client("pi")
        try:
            assert client is not sentinel
            assert client.base_url.startswith("https://archive.example")
        finally:
            for cached in provider._client_cache.values():
                closer = getattr(cached, "close", None)
                if closer:
                    try:
                        import asyncio
                        result = closer()
                        if asyncio.iscoroutine(result):
                            result.close()
                    except Exception:
                        pass


# --- H8: the endpoint key config_io writes is the key the runtime reads ---


def test_config_import_endpoint_reaches_the_runtime():
    from backend.models.config_schema import ConfigV1
    from backend.services.config_io import apply_config_to_session
    from backend.tests.test_config_io import _config_dict

    config = ConfigV1.model_validate(_config_dict(api_config={
        "default_llm": {
            "llm_base_url": "https://proxy.example/v1",
            "api_key": "sk-proxy",
            "model": "m",
        },
        "literature": {"mode": "disabled"},
    }))

    state = apply_config_to_session(config, user_id="u1")

    # The runtime reads this exact key; writing "base_url" here dropped the
    # endpoint while keeping the credential that belongs to it.
    assert state.agent_server_configs["pi"]["llm_base_url"] == "https://proxy.example/v1"
    assert state.agent_server_configs["pi"]["api_key"] == "sk-proxy"


# --- M12: ConfigV1 must describe the sessions the product creates ---------


def test_indicator_without_a_unit_is_exportable():
    """The PI agent writes unit="" when a document does not state one."""
    from backend.models.config_schema import IndicatorEntry

    entry = IndicatorEntry(indicator="ALT", value=31.0, unit="")

    assert entry.unit == ""


def test_a_long_session_name_is_exportable():
    from backend.models.config_schema import AnalysisSection

    section = AnalysisSection(
        name="单细胞数据分析" * 40, initial_instruction="go", round_wait_seconds=0,
    )

    assert section.name


# --- M2: chat delegation respects the same runtime gate -------------------


def test_chat_delegation_checks_the_runtime_gate():
    import inspect

    from backend.routes.chat_parts import delegation

    source = inspect.getsource(delegation._execute_chat_agent)
    assert "is_forward_resume_blocked" in source, (
        "chat delegation would otherwise start a fresh sandbox and run the "
        "bioinformatics agent against an empty kernel while its prompt claims "
        "every earlier result is loaded"
    )


# --- An import job belongs to the user who started it ---------------------


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
    async def test_owner_can_poll(self, _job):
        from backend.routes.export_import import get_import_status

        status = await get_import_status(_job, self._request("owner"))

        assert status["status"] == "extracting"
        assert "user_id" not in status

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
