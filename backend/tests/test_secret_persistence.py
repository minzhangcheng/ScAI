import asyncio
import copy
import io
import json
import logging
import stat
import tarfile

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from backend.models.schemas import AnalysisState
from backend.services import session_store
from backend.utils.secret_redaction import (
    SEALED_SECRET_PREFIX,
    open_system_config_payload,
)
from backend.utils.db import AppDB


RAW_KEY = "sk-123456789"


def _state() -> AnalysisState:
    state = AnalysisState(id="secret-test", user_id="owner")
    state.agent_server_configs = {
        "pi": {"llm_base_url": "https://example.test/v1", "api_key": RAW_KEY}
    }
    state.literature_config["google_api_key"] = "google-abcdef"
    state.config_snapshot = {
        "api_config": {
            "default_llm": {"api_key": "snapshot-default-key"},
            "literature": {"google_api_key": "snapshot-google-key"},
        },
        "agents": {"pi": {"api_key": "snapshot-agent-key"}},
    }
    return state


def test_application_logging_redacts_keys_and_keeps_rotated_files_owner_only(
    tmp_path, monkeypatch
):
    from backend.config import _config as config_module

    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    aiosqlite_logger = logging.getLogger("aiosqlite")
    previous_aiosqlite_level = aiosqlite_logger.level
    monkeypatch.setattr(config_module, "LOG_DIR", tmp_path)
    old_rotated_log = tmp_path / "debug.log.3"
    old_rotated_log.write_text("historic log", encoding="utf-8")
    old_rotated_log.chmod(0o644)
    try:
        config_module.setup_logging()
        logging.getLogger("secret-probe").warning("api_key=%s", RAW_KEY)
        for handler in root.handlers:
            handler.flush()

        app_log = tmp_path / "app.log"
        rendered = app_log.read_text(encoding="utf-8")
        assert RAW_KEY not in rendered
        assert "sk-******789" in rendered
        assert stat.S_IMODE(app_log.stat().st_mode) == 0o600
        assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
        assert stat.S_IMODE(old_rotated_log.stat().st_mode) == 0o600
        assert aiosqlite_logger.level == logging.WARNING

        file_handler = next(
            handler for handler in root.handlers
            if getattr(handler, "baseFilename", "").endswith("app.log")
        )
        file_handler.doRollover()
        logging.getLogger("secret-probe").warning("Bearer %s", RAW_KEY)
        file_handler.flush()
        assert stat.S_IMODE(app_log.stat().st_mode) == 0o600
        assert RAW_KEY not in app_log.read_text(encoding="utf-8")
    finally:
        for handler in list(root.handlers):
            handler.close()
        root.handlers = previous_handlers
        root.setLevel(previous_level)
        aiosqlite_logger.setLevel(previous_aiosqlite_level)


def test_analysis_state_dump_seals_keys_and_round_trip_opens_them():
    dumped = _state().model_dump(mode="json")
    serialized = json.dumps(dumped)

    assert RAW_KEY not in serialized
    assert "snapshot-default-key" not in serialized
    assert "snapshot-google-key" not in serialized
    assert "snapshot-agent-key" not in serialized
    assert dumped["agent_server_configs"]["pi"]["api_key"].startswith(
        SEALED_SECRET_PREFIX
    )
    restored = AnalysisState(**dumped)
    assert restored.agent_server_configs["pi"]["api_key"] == RAW_KEY
    assert restored.literature_config["google_api_key"] == "google-abcdef"
    assert (
        restored.config_snapshot["api_config"]["default_llm"]["api_key"]
        == "snapshot-default-key"
    )


def test_langgraph_checkpoint_serializer_never_contains_raw_key():
    serializer = JsonPlusSerializer(allowed_json_modules=[
        ("backend.models.schemas", "AgentType"),
        ("backend.models.schemas", "AnalysisState"),
        ("backend.models.schemas", "AnalysisPhase"),
        ("backend.models.schemas", "RoundResult"),
        ("backend.models.schemas", "TaskStatus"),
        ("backend.models.schemas", "MessageRole"),
    ])
    type_name, payload = serializer.dumps_typed({"analysis_state": _state()})

    assert type_name == "msgpack"
    assert RAW_KEY.encode() not in payload
    assert b"snapshot-default-key" not in payload
    restored = serializer.loads_typed((type_name, payload))["analysis_state"]
    assert restored.agent_server_configs["pi"]["api_key"] == RAW_KEY


def test_portable_export_state_omits_credentials(tmp_path):
    from backend.routes.export_import import _add_portable_state_json

    source = tmp_path / "session.json"
    source.write_text(json.dumps(_state().model_dump(mode="json")), encoding="utf-8")
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        _add_portable_state_json(archive, source, "session/session.json")

    archive_buffer.seek(0)
    with tarfile.open(fileobj=archive_buffer, mode="r") as archive:
        member = archive.extractfile("session/session.json")
        assert member is not None
        payload = member.read()

    assert RAW_KEY.encode() not in payload
    assert SEALED_SECRET_PREFIX.encode() not in payload
    decoded = json.loads(payload)
    assert decoded["agent_server_configs"]["pi"]["api_key"] == ""


def test_session_json_is_encrypted_and_owner_only(tmp_path, monkeypatch):
    session_path = tmp_path / "secret-test" / "session.json"
    monkeypatch.setattr(
        session_store,
        "get_session_metadata_path",
        lambda _session_id: session_path,
    )

    session_store.save_session_to_disk("secret-test", _state())
    rendered = session_path.read_text(encoding="utf-8")

    assert RAW_KEY not in rendered
    assert stat.S_IMODE(session_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(session_path.parent.stat().st_mode) == 0o700


@pytest.mark.asyncio
async def test_user_config_preview_is_exact_and_resubmission_preserves_key(tmp_path):
    from backend.routes.user_config import (
        UserConfigRequest,
        get_user_config,
        set_user_config,
    )
    from backend.utils import crypto

    database = AppDB(tmp_path / "user-config.db")
    await database.init()
    user_id = await database.create_user(
        "secret-user",
        "test-password-hash",
        email="secret@example.test",
        role="user",
        status="active",
    )
    await set_user_config(
        UserConfigRequest(
            api_key=RAW_KEY,
            llm_base_url="https://one.test/v1",
            proxy_url="https://proxy.test",
        ),
        user_id,
        database,
    )

    status = await get_user_config(user_id, database)
    assert status["masked_api_key"] == "sk-******789"

    await set_user_config(
        UserConfigRequest(
            api_key=status["masked_api_key"],
            llm_base_url="https://two.test/v1",
            proxy_url="•••••",
        ),
        user_id,
        database,
    )
    stored = await database.get_user_llm_config(user_id)
    assert crypto.decrypt_value(stored["encrypted_api_key"]) == RAW_KEY
    assert crypto.decrypt_value(stored["encrypted_proxy_url"]) == "https://proxy.test"
    assert stored["llm_base_url"] == "https://two.test/v1"


@pytest.mark.asyncio
async def test_agent_default_secret_updates_for_one_user_do_not_overwrite_each_other(monkeypatch):
    from backend.routes import user_config

    stored = {"agent_api_keys": {}, "literature": {}}

    async def load_defaults(_db, _user_id):
        snapshot = copy.deepcopy(stored)
        await asyncio.sleep(0)
        return snapshot

    async def save_defaults(_db, _user_id, payload):
        nonlocal stored
        await asyncio.sleep(0)
        stored = copy.deepcopy(payload)

    monkeypatch.setattr(user_config, "load_agent_default_secrets", load_defaults)
    monkeypatch.setattr(user_config, "save_agent_default_secrets", save_defaults)

    await asyncio.gather(
        user_config.update_agent_default_secrets(
            user_config.AgentDefaultSecretsRequest(agent="pi", api_key="pi-secret"),
            "user-1",
            object(),
        ),
        user_config.update_agent_default_secrets(
            user_config.AgentDefaultSecretsRequest(
                agent="bioinformatics", api_key="bio-secret"
            ),
            "user-1",
            object(),
        ),
    )

    assert stored["agent_api_keys"] == {
        "pi": "pi-secret",
        "bioinformatics": "bio-secret",
    }


@pytest.mark.asyncio
async def test_global_config_route_persists_encrypted_api_key(tmp_path, monkeypatch):
    from backend.models.api_requests import ConfigRequest
    from backend.routes import config as config_route
    from backend.state import system_config

    database = AppDB(tmp_path / "global-config.db")
    await database.init()

    async def _get_database():
        return database

    monkeypatch.setattr(config_route, "get_db", _get_database)
    previous_config = system_config.get("current")
    try:
        await config_route.set_config(
            ConfigRequest(
                api_key=RAW_KEY,
                llm_base_url="https://global.test/v1",
            ),
            "admin-user",
        )
        status = await config_route.get_config("admin-user")
        assert status["llm_base_url"] == "https://global.test/v1"
        assert status["masked_api_key"] == "sk-******789"

        await config_route.set_config(
            ConfigRequest(
                api_key=status["masked_api_key"],
                llm_base_url="http://localhost:8000/v1",
            ),
            "admin-user",
        )
        persisted = await database.get_meta("global_system_config")
    finally:
        system_config["current"] = previous_config

    assert RAW_KEY not in persisted
    payload = json.loads(persisted)
    assert payload["api"]["api_key"].startswith(SEALED_SECRET_PREFIX)
    assert open_system_config_payload(payload)["api"]["api_key"] == RAW_KEY
    assert payload["api"]["llm_base_url"] == "http://localhost:8000/v1"
