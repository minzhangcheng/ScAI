from backend.tests.split_facade import setattr_facade
import io
import json
import logging
import stat
import tarfile

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from backend.models.schemas import AnalysisState
from backend.services import session_store
from backend.services.snapshot_v2.service import SnapshotService
from backend.utils.secret_redaction import (
    SEALED_SECRET_PREFIX,
    SecretRedactionFilter,
    mask_api_key,
    open_system_config_payload,
    preserve_runtime_secrets,
    seal_system_config_payload,
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


def test_api_key_preview_has_exact_required_shape():
    assert mask_api_key(RAW_KEY) == "sk-******789"


def test_log_filter_replaces_raw_key_with_required_preview():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(SecretRedactionFilter())
    logger = logging.getLogger("scai-secret-redaction-test")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    logger.info("api_key=%s Authorization: Bearer %s", RAW_KEY, RAW_KEY)
    rendered = stream.getvalue()

    assert RAW_KEY not in rendered
    assert rendered.count("sk-******789") == 2


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


def test_snapshot_business_state_never_contains_raw_key():
    payload = SnapshotService._serialize_analysis_state(_state())
    assert RAW_KEY.encode() not in payload
    assert b"snapshot-default-key" not in payload
    assert SEALED_SECRET_PREFIX.encode() not in payload
    decoded = json.loads(payload)
    assert decoded["agent_server_configs"]["pi"]["api_key"] == ""
    assert decoded["config_snapshot"]["agents"]["pi"]["api_key"] == ""


def test_snapshot_restore_preserves_current_runtime_credentials():
    current = _state()
    restored = AnalysisState(id="secret-test", user_id="owner")
    restored.agent_server_configs = {
        "pi": {"llm_base_url": "https://old.test/v1", "api_key": ""}
    }

    preserve_runtime_secrets(current, restored)

    assert restored.agent_server_configs["pi"]["api_key"] == RAW_KEY
    assert restored.literature_config["google_api_key"] == "google-abcdef"


def test_global_system_config_seals_api_credentials():
    payload = {
        "api": {
            "api_key": RAW_KEY,
            "llm_base_url": "https://example.test/v1",
            "proxy_url": "https://proxy-user:proxy-password@example.test",
        }
    }

    sealed = seal_system_config_payload(payload)
    serialized = json.dumps(sealed)

    assert RAW_KEY not in serialized
    assert "proxy-password" not in serialized
    assert sealed["api"]["api_key"].startswith(SEALED_SECRET_PREFIX)
    assert open_system_config_payload(sealed) == payload


def test_global_system_config_discards_legacy_deployment_runtime_budgets():
    payload = {
        "api": {"api_key": "", "llm_base_url": "https://example.test/v1"},
        "runtime_max_namespace_count": 1500,
        "runtime_max_single_object_bytes": 4 * 1024 ** 3,
        "runtime_max_anndata_total_bytes": 12 * 1024 ** 3,
        "runtime_max_rss_bytes": 16 * 1024 ** 3,
        "runtime_max_pkl_bytes": 8 * 1024 ** 3,
        "runtime_max_save_seconds": 300,
    }

    sealed = seal_system_config_payload(payload)
    opened = open_system_config_payload(payload)

    for key in payload.keys() - {"api"}:
        assert key not in sealed
        assert key not in opened


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


def test_portable_export_omits_installation_bound_checkpoint(tmp_path, monkeypatch):
    from backend.routes import export_import

    states = tmp_path / "states"
    states.mkdir()
    (states / "checkpoints.sqlite").write_bytes(b"encrypted local checkpoint")
    (states / "checkpoints.sqlite-wal").write_bytes(b"wal")
    (states / "analysis.ipynb").write_text("{}", encoding="utf-8")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    setattr_facade(monkeypatch, "backend.routes.export_import", "get_session_states", lambda _session_id: states)
    setattr_facade(monkeypatch, "backend.routes.export_import", "STATE_DIR", legacy)

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        export_import._add_states_to_tar(
            archive, "portable-session", include_runtime=False
        )

    archive_buffer.seek(0)
    with tarfile.open(fileobj=archive_buffer, mode="r") as archive:
        names = set(archive.getnames())

    assert "portable-session/states/analysis.ipynb" in names
    assert "portable-session/states/checkpoints.sqlite" not in names
    assert "portable-session/states/checkpoints.sqlite-wal" not in names


def test_legacy_state_export_excludes_near_prefix_sessions(tmp_path, monkeypatch):
    from backend.routes import export_import

    session_id = "abc"
    states = tmp_path / "states"
    states.mkdir()
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "abc.ipynb").write_bytes(b"owned")
    (legacy / "abc_session.pkl").write_bytes(b"owned-runtime")
    (legacy / "abcd_session.pkl").write_bytes(b"other-session")
    (legacy / "abc-extra_state.json").write_bytes(b"other-session")
    (legacy / "abc_extra_state.json").write_bytes(b"other-session")
    setattr_facade(
        monkeypatch, "backend.routes.export_import", "get_session_states",
        lambda _session_id: states,
    )
    setattr_facade(
        monkeypatch, "backend.routes.export_import", "STATE_DIR", legacy,
    )

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        export_import._add_states_to_tar(
            archive, session_id, include_runtime=True,
        )
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode="r") as archive:
        names = set(archive.getnames())

    assert f"{session_id}/states/abc.ipynb" in names
    assert f"{session_id}/states/abc_session.pkl" in names
    assert f"{session_id}/states/abcd_session.pkl" not in names
    assert f"{session_id}/states/abc-extra_state.json" not in names
    assert f"{session_id}/states/abc_extra_state.json" not in names


@pytest.mark.parametrize(
    ("session_id", "ambiguous_name"),
    [
        ("abc_state", "abc_state.json"),
        ("abc_session", "abc_session.pkl"),
    ],
)
def test_legacy_state_export_excludes_ambiguous_shorter_id_file(
    tmp_path, monkeypatch, session_id, ambiguous_name,
):
    from backend.routes import export_import

    states = tmp_path / "states"
    states.mkdir()
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / ambiguous_name).write_bytes(b"shorter-session")
    setattr_facade(
        monkeypatch, "backend.routes.export_import", "get_session_states",
        lambda _session_id: states,
    )
    setattr_facade(
        monkeypatch, "backend.routes.export_import", "STATE_DIR", legacy,
    )

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        export_import._add_states_to_tar(
            archive, session_id, include_runtime=True,
        )
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode="r") as archive:
        assert f"{session_id}/states/{ambiguous_name}" not in archive.getnames()


def _states_export_names(tmp_path, monkeypatch, include_runtime):
    """Pack a states/ dir through the real builder and return the member names."""
    from backend.routes import export_import

    states = tmp_path / "states"
    states.mkdir()
    (states / "analysis.ipynb").write_text("{}", encoding="utf-8")
    (states / "runtime_h5ad_manifest.json").write_text(
        '{"version": 1, "entries": {"adata": {"path": "adata.h5ad", "sha256": "%s"}}}'
        % ("0" * 64),
        encoding="utf-8",
    )
    (states / "adata.h5ad").write_bytes(b"H5AD")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    setattr_facade(
        monkeypatch, "backend.routes.export_import", "get_session_states",
        lambda _session_id: states,
    )
    setattr_facade(monkeypatch, "backend.routes.export_import", "STATE_DIR", legacy)

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        export_import._add_states_to_tar(
            archive, "sidecar-session", include_runtime=include_runtime
        )
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode="r") as archive:
        return set(archive.getnames())


def test_portable_export_omits_the_h5ad_runtime_sidecars(tmp_path, monkeypatch):
    """They are runtime, and this is the export that declares the runtime absent.

    Matching neither ``is_sandbox_pickle`` nor the checkpoint filter, the pair used
    to travel in every ordinary archive -- shipping a multi-GB AnnData that the
    importer then marks ``runtime_restore_status=unavailable``.
    """
    names = _states_export_names(tmp_path, monkeypatch, include_runtime=False)

    assert "sidecar-session/states/analysis.ipynb" in names
    assert "sidecar-session/states/runtime_h5ad_manifest.json" not in names
    assert "sidecar-session/states/adata.h5ad" not in names


def test_a_runtime_export_keeps_the_h5ad_sidecars_as_a_pair(tmp_path, monkeypatch):
    """Atomicity is the property that matters, in both directions.

    A manifest whose sidecars were dropped makes every later kernel start raise
    "h5ad sidecar missing" -- a brick rather than a degrade -- and sidecars with no
    manifest are inert, since nothing reads ``states/*.h5ad`` except through it.
    """
    names = _states_export_names(tmp_path, monkeypatch, include_runtime=True)

    assert "sidecar-session/states/runtime_h5ad_manifest.json" in names
    assert "sidecar-session/states/adata.h5ad" in names


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
async def test_application_database_is_owner_only(tmp_path):
    database_path = tmp_path / "private" / "app.db"
    database = AppDB(database_path)

    await database.init()

    assert stat.S_IMODE(database_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(database_path.parent.stat().st_mode) == 0o700


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
        persisted = await database.get_meta("global_system_config")
    finally:
        system_config["current"] = previous_config

    assert RAW_KEY not in persisted
    payload = json.loads(persisted)
    assert payload["api"]["api_key"].startswith(SEALED_SECRET_PREFIX)
    assert open_system_config_payload(payload)["api"]["api_key"] == RAW_KEY
