"""Feature-gate tests for the default-disabled Snapshot V2 subsystem."""

from backend.tests.split_facade import setattr_facade
import asyncio
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "False"),
        ("", "False"),
        ("0", "False"),
        ("false", "False"),
        ("1", "True"),
        ("true", "True"),
        ("TRUE", "True"),
        ("yes", "True"),
    ],
)
def test_snapshot_v2_environment_gate(value, expected):
    env = {**os.environ, "SCAI_DATA_DIR": "/tmp/scai_snapshot_gate_test"}
    if value is None:
        env.pop("SCAI_ENABLE_SNAPSHOT_V2", None)
    else:
        env["SCAI_ENABLE_SNAPSHOT_V2"] = value

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from backend.config._config import "
                "is_snapshot_v2_enabled; print(is_snapshot_v2_enabled())"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == expected


def test_snapshot_v2_system_config_accepts_explicit_disabled():
    from backend.config._config import SystemConfig

    assert SystemConfig(snapshot_v2_enabled=False).snapshot_v2_enabled is False


def test_config_status_exposes_snapshot_v2_capability(monkeypatch):
    from backend.routes import config as config_routes

    monkeypatch.setattr(config_routes, "is_snapshot_v2_enabled", lambda: False)
    monkeypatch.setitem(config_routes.system_config, "current", None)

    result = asyncio.run(config_routes.config_status())

    assert result == {
        "configured": False,
        "has_api_key": False,
        "snapshot_v2_enabled": False,
    }


@pytest.mark.asyncio
async def test_disabled_workflow_snapshot_producer_is_a_noop(monkeypatch):
    from backend.config import _config
    from backend.models.schemas import AnalysisState
    from backend.workflow.core import SingleCellAnalysisWorkflow

    monkeypatch.setattr(_config, "SNAPSHOT_V2_ENABLED", False)

    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "snapshot-disabled-workflow"
    workflow._current_graph_state = {}
    workflow.sandbox = None
    workflow.snapshot_service = MagicMock()
    workflow.snapshot_service.capture_required_snapshot = AsyncMock()
    state = AnalysisState(
        id="snapshot-disabled-workflow",
        name="Disabled",
        user_id="u1",
        initial_instruction="analyze",
    )

    result = await workflow._create_auto_snapshot(state, "pre_round_wait")

    assert result is None
    workflow.snapshot_service.capture_required_snapshot.assert_not_awaited()
    assert state.snapshot_failure is None


@pytest.mark.asyncio
async def test_disabled_post_chat_snapshot_producer_is_a_noop(monkeypatch):
    from backend.config import _config
    from backend.models.schemas import AnalysisState
    from backend.routes import chat as chat_routes

    monkeypatch.setattr(_config, "SNAPSHOT_V2_ENABLED", False)
    capture = AsyncMock()
    setattr_facade(monkeypatch, "backend.routes.chat", "capture_post_chat_snapshot", capture)
    state = AnalysisState(
        id="snapshot-disabled-chat",
        name="Disabled",
        user_id="u1",
        initial_instruction="analyze",
    )

    result = await chat_routes._create_chat_snapshot(state.id, state)

    assert result is None
    capture.assert_not_awaited()
    assert state.snapshot_failure is None


@pytest.mark.asyncio
async def test_disabled_post_report_snapshot_producer_clears_stale_pending(monkeypatch):
    from backend.config import _config
    from backend.models.schemas import AnalysisState
    from backend.workflow.reporting import ReportingMixin

    monkeypatch.setattr(_config, "SNAPSHOT_V2_ENABLED", False)
    workflow = object.__new__(ReportingMixin)
    workflow._capture_post_report_snapshot = AsyncMock()
    workflow._capture_required_snapshot = AsyncMock()
    state = AnalysisState(
        id="snapshot-disabled-report",
        name="Disabled",
        user_id="u1",
        initial_instruction="analyze",
        pending_required_snapshot={
            "phase": "post_report",
            "report_generation_id": "report-1",
            "attempts": 1,
        },
    )

    result = await workflow._capture_post_report_if_enabled(state, "report-1")

    assert result is None
    assert state.pending_required_snapshot is None
    workflow._capture_post_report_snapshot.assert_not_awaited()
    workflow._capture_required_snapshot.assert_not_awaited()


def test_disabled_snapshot_v2_http_gate_returns_stable_404(monkeypatch):
    from fastapi import HTTPException

    from backend.config import _config
    from backend.services.snapshot_v2.feature_gate import (
        require_snapshot_v2_enabled,
    )

    monkeypatch.setattr(_config, "SNAPSHOT_V2_ENABLED", False)

    with pytest.raises(HTTPException) as exc:
        require_snapshot_v2_enabled()

    assert exc.value.status_code == 404
    assert exc.value.detail == {
        "code": "SNAPSHOT_V2_DISABLED",
        "message": "Snapshot V2 is disabled",
    }


def test_enabled_snapshot_v2_http_gate_allows_request(monkeypatch):
    from backend.config import _config
    from backend.services.snapshot_v2.feature_gate import (
        require_snapshot_v2_enabled,
    )

    monkeypatch.setattr(_config, "SNAPSHOT_V2_ENABLED", True)

    assert require_snapshot_v2_enabled() is None


def test_all_snapshot_v2_http_operations_use_central_dependency():
    from backend.main import app
    from backend.services.snapshot_v2.feature_gate import (
        require_snapshot_v2_enabled,
    )

    expected = {
        ("GET", "/api/session/{session_id}/snapshots-v2"),
        ("GET", "/api/session/{session_id}/snapshots-v2/{snapshot_id}"),
        ("DELETE", "/api/session/{session_id}/snapshots-v2/{snapshot_id}"),
        ("POST", "/api/session/{session_id}/snapshots/{snapshot_id}/restore-v2"),
        ("GET", "/api/session/{session_id}/snapshot-transactions/{transaction_id}"),
        ("POST", "/api/session/{session_id}/snapshot-transactions/{transaction_id}/cancel"),
        ("POST", "/api/session/{session_id}/export-v2"),
        ("POST", "/api/session/{session_id}/import-v2"),
        ("POST", "/api/session/{session_id}/snapshot-failure/retry"),
        ("POST", "/api/session/{session_id}/snapshot-failure/skip"),
        ("POST", "/api/session/{session_id}/snapshot-failure/terminate"),
    }

    gated = set()
    for route in app.routes:
        dependencies = getattr(route, "dependencies", [])
        if not any(
            getattr(dep, "dependency", None) is require_snapshot_v2_enabled
            for dep in dependencies
        ):
            continue
        for method in getattr(route, "methods", set()):
            gated.add((method, route.path))

    assert expected <= gated


def _snapshot_blocked_state(producer: str, previous_status: str = "running"):
    from backend.models.schemas import AnalysisState, TaskStatus

    state = AnalysisState(
        id=f"stale-{producer}",
        name="Stale snapshot gate",
        user_id="u1",
        initial_instruction="analyze",
        status=TaskStatus.WAITING_INPUT,
        wait_input_reason="snapshot_failure",
        snapshot_failure={
            "code": "SNAPSHOT_FAILED",
            "message": "old failure",
            "retry_context": {
                "producer": producer,
                "previous_status": previous_status,
                "snapshot_request": {
                    "phase": "post_report" if producer == "post_report" else producer,
                },
            },
        },
    )
    return state


@pytest.mark.parametrize(
    ("producer", "previous_status", "workflow_active", "expected_status"),
    [
        ("workflow_boundary", "running", False, "paused"),
        ("workflow_boundary", "running", True, "running"),
        ("chat", "completed", False, "completed"),
        ("chat", "paused", False, "paused"),
        ("pre_restore", "running", False, "paused"),
    ],
)
def test_stale_snapshot_failure_is_normalized(
    producer,
    previous_status,
    workflow_active,
    expected_status,
):
    from backend.services.snapshot_v2.feature_gate import (
        normalize_disabled_snapshot_state,
    )

    state = _snapshot_blocked_state(producer, previous_status)
    transaction = {"transaction_id": "restore-audit", "status": "failed"}
    state.snapshot_transaction = transaction

    changed = normalize_disabled_snapshot_state(
        state,
        workflow_active=workflow_active,
    )

    assert changed is True
    assert state.snapshot_failure is None
    assert state.wait_input_reason is None
    assert state.pending_required_snapshot is None
    assert state.status.value == expected_status
    assert state.snapshot_transaction == transaction
    assert normalize_disabled_snapshot_state(
        state,
        workflow_active=workflow_active,
    ) is False


def test_stale_post_report_snapshot_failure_completes_report():
    from backend.services.snapshot_v2.feature_gate import (
        normalize_disabled_snapshot_state,
    )

    state = _snapshot_blocked_state("post_report", "generating_report")
    state.pending_required_snapshot = {
        "phase": "post_report",
        "report_generation_id": "report-1",
    }

    assert normalize_disabled_snapshot_state(state) is True
    assert state.status.value == "completed"
    assert state.pending_required_snapshot is None


def test_recovery_block_is_cleared_when_the_subsystem_is_off():
    """Nothing else can ever clear it once V2 is disabled.

    ``recovery_blocked`` is only ever set by journal recovery, which runs only
    while V2 is enabled, and ``assert_recovery_clear`` refuses every start,
    resume, WebSocket attach and kernel launch while it is set. Turning the
    subsystem off would otherwise promote a recoverable session to a permanently
    unstartable one, with every snapshot endpoint answering 404.
    """
    from backend.models.schemas import AnalysisState
    from backend.services.session_store import assert_recovery_clear
    from backend.services.snapshot_v2.feature_gate import (
        normalize_disabled_snapshot_state,
    )

    state = AnalysisState(id="recovery-blocked", user_id="owner")
    state.recovery_blocked = True
    state.snapshot_transaction = {
        "transaction_id": "restore-audit",
        "recovery_error": {"code": "RECOVERY_UNEXPECTED", "message": "boom"},
    }

    assert normalize_disabled_snapshot_state(state) is True
    assert state.recovery_blocked is False
    # The audit trail survives; only the blocker is dropped.
    assert state.snapshot_transaction == {"transaction_id": "restore-audit"}
    assert_recovery_clear(state, session_id=state.id)
    assert normalize_disabled_snapshot_state(state) is False


def test_clearing_a_recovery_block_leaves_the_status_alone():
    """A recovery block says nothing about where the run had got to.

    The status reconciliation below belongs to the snapshot-failure gate, which
    carries a ``previous_status``. Running it for a session that only has a
    recovery block would read that missing field as unparseable and downgrade a
    finished session to paused.
    """
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2.feature_gate import (
        normalize_disabled_snapshot_state,
    )

    state = AnalysisState(id="finished", user_id="owner")
    state.status = TaskStatus.COMPLETED
    state.recovery_blocked = True

    assert normalize_disabled_snapshot_state(state) is True
    assert state.recovery_blocked is False
    assert state.status == TaskStatus.COMPLETED


def test_a_session_with_nothing_stale_is_untouched():
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.snapshot_v2.feature_gate import (
        normalize_disabled_snapshot_state,
    )

    state = AnalysisState(id="clean", user_id="owner")
    state.status = TaskStatus.PAUSED

    assert normalize_disabled_snapshot_state(state) is False
    assert state.status == TaskStatus.PAUSED


def test_stale_untyped_snapshot_failure_falls_back_to_paused():
    from backend.services.snapshot_v2.feature_gate import (
        normalize_disabled_snapshot_state,
    )

    state = _snapshot_blocked_state("workflow_boundary")
    state.snapshot_failure.pop("retry_context")

    assert normalize_disabled_snapshot_state(state) is True
    assert state.status.value == "paused"


@pytest.mark.asyncio
async def test_run_lifecycle_clears_stale_snapshot_gate_when_disabled(monkeypatch):
    from backend.config import _config
    from backend.routes import lifecycle

    monkeypatch.setattr(_config, "SNAPSHOT_V2_ENABLED", False)
    state = _snapshot_blocked_state("workflow_boundary", "running")
    monkeypatch.setattr(
        lifecycle,
        "require_session_ownership",
        AsyncMock(return_value=state),
    )
    monkeypatch.setattr(lifecycle, "assert_recovery_clear", lambda *_a, **_kw: None)
    save = AsyncMock()
    monkeypatch.setattr(lifecycle, "async_save_session", save)
    request = SimpleNamespace(state=SimpleNamespace(user_id="u1"))

    response = await lifecycle.run_analysis(state.id, request)

    assert response.success is True
    assert state.status.value == "running"
    assert state.snapshot_failure is None
    assert state.wait_input_reason is None
    # Forced: the write carries the cleared gate and the start intent, and
    # RUNNING does not bypass the save throttle on its own.
    save.assert_awaited_once_with(state.id, state, force=True)


@pytest.mark.asyncio
async def test_workflow_entry_clears_stale_snapshot_gate_when_disabled(monkeypatch):
    from backend.config import _config
    from backend.workflow.core import SingleCellAnalysisWorkflow

    monkeypatch.setattr(_config, "SNAPSHOT_V2_ENABLED", False)
    state = _snapshot_blocked_state("workflow_boundary", "running")
    state.restore_resume_kind = "terminal"
    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = state.id
    workflow._send_state_sync = AsyncMock()

    result = await workflow.run(state, "thread-disabled-snapshot")

    assert result is state
    assert state.status.value == "completed"
    assert state.snapshot_failure is None
    workflow._send_state_sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_load_skips_snapshot_recovery_and_normalizes_cached_state(
    monkeypatch,
):
    from backend.config import _config
    from backend.services import session_store
    from backend.services.snapshot_v2 import recovery
    from backend.state import sessions

    monkeypatch.setattr(_config, "SNAPSHOT_V2_ENABLED", False)
    state = _snapshot_blocked_state("chat", "completed")
    sessions[state.id] = state
    recover = AsyncMock()
    monkeypatch.setattr(recovery, "recover_snapshot_transactions", recover)
    persist = AsyncMock()
    monkeypatch.setattr(session_store, "_persist_recovery_marker", persist)

    try:
        loaded = await session_store.ensure_session_loaded(state.id)
    finally:
        sessions.pop(state.id, None)

    assert loaded is state
    recover.assert_not_awaited()
    assert state.snapshot_failure is None
    assert state.status.value == "completed"
    persist.assert_awaited_once_with(state.id, state)


@pytest.mark.asyncio
async def test_startup_skips_snapshot_recovery_when_disabled(monkeypatch):
    from backend.config import _config
    from backend import main
    from backend.services.snapshot_v2 import recovery

    monkeypatch.setattr(_config, "SNAPSHOT_V2_ENABLED", False)
    recover = AsyncMock()
    monkeypatch.setattr(recovery, "recover_for_all_sessions", recover)

    await main._recover_snapshots_on_startup()

    recover.assert_not_awaited()


def test_ordinary_import_does_not_materialize_v2_bundle_when_disabled(
    monkeypatch,
    tmp_path,
):
    from fastapi import HTTPException

    from backend.config import _config
    from backend.routes import export_import

    monkeypatch.setattr(_config, "SNAPSHOT_V2_ENABLED", False)
    object_hash = "a" * 64
    leaked_path = export_import.DATA_BASE_DIR / object_hash
    leaked_path.unlink(missing_ok=True)
    archive = tmp_path / "snapshot-v2.tar.gz"

    with tarfile.open(archive, mode="w:gz") as tf:
        payload = b"snapshot-object"
        object_info = tarfile.TarInfo(name=f"objects/{object_hash}")
        object_info.size = len(payload)
        tf.addfile(object_info, io.BytesIO(payload))

        metadata = b'{"format_version": 2}'
        metadata_info = tarfile.TarInfo(name="snapshot-v2-export.json")
        metadata_info.size = len(metadata)
        tf.addfile(metadata_info, io.BytesIO(metadata))

    with pytest.raises(HTTPException) as exc:
        export_import._import_tar_gz(str(archive), user_id="u1")

    assert exc.value.status_code == 400
    assert not leaked_path.exists()


# --- The recovery block is a claim about a *restore*, not about journals ------
#
# ``journals/`` is one directory for every kind (paths.py), so classifying by
# status alone read a stranded delete or import transaction as a half-applied
# restore. With Snapshot V2 off nothing can ever clear ``recovery_blocked``
# again and no snapshot endpoint is reachable to try, so the session became
# permanently unstartable — over work that never touched its state, notebook,
# checkpoint or workspace, which is the whole justification for fail-closing.


def _write_journal(session_id, *, kind, status, transaction_id="tx-1"):
    from backend.services.snapshot_v2.journal_store import JournalStore, make_envelope
    from backend.services.snapshot_v2.paths import SnapshotV2Paths

    paths = SnapshotV2Paths.for_session(session_id)
    paths.journals.mkdir(parents=True, exist_ok=True)
    for stale in paths.journals.iterdir():
        stale.unlink()
    JournalStore(paths).write(
        make_envelope(
            kind=kind,
            transaction_id=transaction_id,
            session_id=session_id,
            status=status,
            payload={},
        )
    )
    return paths


def _write_raw_journal(session_id, name, text):
    from backend.services.snapshot_v2.paths import SnapshotV2Paths

    paths = SnapshotV2Paths.for_session(session_id)
    paths.journals.mkdir(parents=True, exist_ok=True)
    for stale in paths.journals.iterdir():
        stale.unlink()
    (paths.journals / name).write_text(text, encoding="utf-8")
    return paths


@pytest.mark.parametrize(
    ("kind", "status"),
    [
        # deletion.py stages then applies inside a try whose except deliberately
        # leaves the journal on disk for recovery.
        ("delete", "STAGED"),
        ("delete", "APPLYING"),
        # An import writes objects/ and committed/ only.
        ("import", "STAGING"),
    ],
)
def test_a_stranded_non_restore_journal_does_not_block_a_disabled_session(
    kind, status,
):
    from backend.models.schemas import AnalysisState
    from backend.services.snapshot_v2.feature_gate import (
        has_unfinished_restore_journal,
        normalize_disabled_snapshot_state,
    )

    session_id = f"journal-{kind}-{status.lower()}"
    _write_journal(session_id, kind=kind, status=status)

    assert has_unfinished_restore_journal(session_id) is False, (
        f"a {kind} journal was read as a half-applied restore"
    )

    state = AnalysisState(id=session_id, user_id="owner")
    state.recovery_blocked = True

    assert normalize_disabled_snapshot_state(state) is True
    assert state.recovery_blocked is False, (
        "the session stays unstartable with Snapshot V2 off, and the operator is "
        "told to re-enable it so recovery can finish a restore that never ran"
    )


@pytest.mark.parametrize(
    ("kind", "status"),
    [
        ("restore_apply", "STAGED"),
        ("restore_apply", "APPLYING"),
        # jobs.py writes restore_job statuses in lower case.
        ("restore_job", "applying"),
    ],
)
def test_an_unfinished_restore_journal_still_fails_closed(kind, status):
    from backend.models.schemas import AnalysisState
    from backend.services.snapshot_v2.feature_gate import (
        has_unfinished_restore_journal,
        normalize_disabled_snapshot_state,
    )

    session_id = f"journal-{kind}-{status.lower()}"
    _write_journal(session_id, kind=kind, status=status)

    assert has_unfinished_restore_journal(session_id) is True

    state = AnalysisState(id=session_id, user_id="owner")
    state.recovery_blocked = True

    assert normalize_disabled_snapshot_state(state) is False
    assert state.recovery_blocked is True


def test_a_terminal_restore_journal_is_only_waiting_to_be_reaped():
    from backend.services.snapshot_v2.feature_gate import (
        has_unfinished_restore_journal,
    )

    session_id = "journal-restore-committed"
    _write_journal(session_id, kind="restore_apply", status="COMMITTED")

    assert has_unfinished_restore_journal(session_id) is False


@pytest.mark.parametrize(
    ("name", "text"),
    [
        # A truncated write is itself evidence of an interrupted one.
        ("truncated.json", "{not json"),
        # Not an object: doc.get would raise, and unknown means fail closed.
        ("notadict.json", '["a list"]'),
        # Task-1 restore journals predate the kind envelope entirely.
        (
            "legacy.json",
            '{"transaction_id": "t", "session_id": "s", "status": "APPLYING",'
            ' "target_snapshot_id": "snap", "recovery_mode": "rollback"}',
        ),
    ],
)
def test_unreadable_or_kindless_journals_still_fail_closed(name, text):
    from backend.services.snapshot_v2.feature_gate import (
        has_unfinished_restore_journal,
    )

    session_id = f"journal-raw-{name.split('.')[0]}"
    _write_raw_journal(session_id, name, text)

    assert has_unfinished_restore_journal(session_id) is True
