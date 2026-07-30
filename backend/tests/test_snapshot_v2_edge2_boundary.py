"""Regression tests for the Snapshot V2 boundary defects (second edge pass).

Boundary = the two places a session crosses from one durable state into the
next: the HITL forward-motion endpoints in ``backend/routes/interaction.py``,
and the import/export/deletion transactions that publish or reclaim objects.

One test per defect, named for the failure it prevents rather than for the
function it calls. Every one of these was verified to fail against the source
as it stood before this pass.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import types
from pathlib import Path

import pytest

from backend.models.schemas import AnalysisPhase, AnalysisState, TaskStatus


# ── shared minimal paths object (mirrors test_snapshot_v2_edge_hardening) ────


class _Paths:
    def __init__(self, root: Path):
        self.root = root / "v2"
        self.objects = root / "objects"
        self.staging = root / "staging"
        self.journals = root / "journals"
        self.integrity = root / "integrity"
        self.committed = root / "committed"

    def ensure_dirs(self):
        for d in (
            self.root, self.objects, self.staging,
            self.journals, self.integrity, self.committed,
        ):
            d.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def v2_paths(tmp_path):
    paths = _Paths(tmp_path)
    paths.ensure_dirs()
    return paths


def _armed_state() -> AnalysisState:
    """A session parked on an armed, unresolved required-snapshot gate."""
    state = AnalysisState(name="t", user_id="u", initial_instruction="o")
    state.status = TaskStatus.WAITING_INPUT
    state.wait_input_reason = "snapshot_failure"
    state.snapshot_failure = {
        "producer": "post_chat",
        "code": "snapshot_quota_exceeded",
        "retry_context": {"message": "hi"},
    }
    return state


def _install_state(monkeypatch, state, workflow):
    import backend.routes.interaction as ix

    async def fake_owner(session_id, user_id):
        return state

    async def fake_noop(*args, **kwargs):
        return None

    monkeypatch.setattr(ix, "require_session_ownership", fake_owner)
    monkeypatch.setattr(ix, "_update_session_cache", fake_noop)
    monkeypatch.setattr(ix, "async_save_session", fake_noop)
    monkeypatch.setattr(ix, "_pause_and_restart_ws", fake_noop)
    monkeypatch.setattr(ix, "workflows", {"sess": workflow})
    return ix


class _RecordingWorkflow:
    def __init__(self):
        self.supplementary_calls = []
        self.continue_calls = 0

    def signal_supplementary_analysis(self, request_text, target):
        self.supplementary_calls.append((request_text, target))

    def signal_continue(self):
        self.continue_calls += 1


# ── F17. /supplementary-analysis cannot release a fail-closed snapshot gate ──
#
# `signal_supplementary_analysis(target="direction"|"round")` sets the very
# `_continue_event` the round/direction gate waits on, so it drives a full
# PI-planned supplementary run — agent, kernel, workspace mutation — past a
# boundary the design guarantees is snapshotted.


@pytest.mark.parametrize("target", ["direction", "round"])
def test_supplementary_analysis_refuses_while_the_snapshot_gate_is_armed(
    monkeypatch, target,
):
    from fastapi import HTTPException

    from backend.models.api_requests import SupplementaryAnalysisRequest

    state = _armed_state()
    workflow = _RecordingWorkflow()
    ix = _install_state(monkeypatch, state, workflow)

    with pytest.raises(HTTPException) as caught:
        asyncio.run(ix.request_supplementary_analysis(
            "sess",
            types.SimpleNamespace(state=types.SimpleNamespace(user_id="u")),
            SupplementaryAnalysisRequest(request_text="x", target=target),
        ))

    assert caught.value.status_code == 409
    assert workflow.supplementary_calls == [], (
        "supplementary analysis set the same _continue_event the round gate "
        "waits on, so the run advanced with snapshot_failure still armed"
    )


def test_supplementary_analysis_still_runs_with_no_gate_armed(monkeypatch):
    from backend.models.api_requests import SupplementaryAnalysisRequest

    state = AnalysisState(name="t", user_id="u", initial_instruction="o")
    state.status = TaskStatus.WAITING_INPUT
    workflow = _RecordingWorkflow()
    ix = _install_state(monkeypatch, state, workflow)

    response = asyncio.run(ix.request_supplementary_analysis(
        "sess",
        types.SimpleNamespace(state=types.SimpleNamespace(user_id="u")),
        SupplementaryAnalysisRequest(request_text="x", target="direction"),
    ))

    assert response.success is True
    assert workflow.supplementary_calls == [("x", "direction")]


# ── F18. /finalize-from-integration cannot release the gate either ───────────
#
# The gate leaves `analysis_phase == INTEGRATION` and `status ==
# WAITING_INPUT` intact, so the endpoint's only guard still passes and the
# graph runs straight to a COMPLETED final report with the gate armed.


def test_finalize_from_integration_refuses_while_the_snapshot_gate_is_armed(
    monkeypatch,
):
    from fastapi import HTTPException

    state = _armed_state()
    state.analysis_phase = AnalysisPhase.INTEGRATION
    workflow = _RecordingWorkflow()
    ix = _install_state(monkeypatch, state, workflow)

    with pytest.raises(HTTPException) as caught:
        asyncio.run(ix.finalize_from_integration(
            "sess",
            types.SimpleNamespace(state=types.SimpleNamespace(user_id="u")),
        ))

    assert caught.value.status_code == 409
    assert workflow.continue_calls == 0
    assert state.final_report_requested is False, (
        "the finalize request was recorded and force-persisted, so the WS "
        "reconnect path would still try to generate the report"
    )
    assert state.status == TaskStatus.WAITING_INPUT


def test_finalize_from_integration_still_runs_with_no_gate_armed(monkeypatch):
    state = AnalysisState(name="t", user_id="u", initial_instruction="o")
    state.status = TaskStatus.WAITING_INPUT
    state.analysis_phase = AnalysisPhase.INTEGRATION
    workflow = _RecordingWorkflow()
    ix = _install_state(monkeypatch, state, workflow)

    response = asyncio.run(ix.finalize_from_integration(
        "sess",
        types.SimpleNamespace(state=types.SimpleNamespace(user_id="u")),
    ))

    assert response.success is True
    assert workflow.continue_calls == 1
    assert state.final_report_requested is True


# ── F12. The import rollback honours every root, not just visible manifests ──


def _publish(paths, snapshot_id, digest, size):
    from backend.services.snapshot_v2.models import ObjectRef
    from backend.tests.snapshot_v2_fixtures import (
        make_snapshot_v2_manifest, write_manifest_to_store,
    )

    manifest = make_snapshot_v2_manifest(
        session_id="s1",
        snapshot_id=snapshot_id,
        objects={
            digest: ObjectRef(
                sha256=digest, size=size, media_type="application/octet-stream",
            )
        },
    )
    return write_manifest_to_store(manifest, paths.committed, paths.integrity)


def _pending(paths, snapshot_id, digest, size):
    from backend.services.snapshot_v2.object_store import ManifestStore

    manifest = _publish(paths, snapshot_id, digest, size)
    ManifestStore._pending_marker_path(paths.committed, snapshot_id).touch()
    return manifest


def test_import_rollback_keeps_objects_an_audit_bundle_still_references(v2_paths):
    from backend.services.snapshot_v2.deletion import DeletionService

    body = b"discarded-state"
    digest = hashlib.sha256(body).hexdigest()
    (v2_paths.objects / digest).write_bytes(body)

    # The only record of the state a `discard_current` restore threw away.
    _publish(v2_paths, "audit-deadbeefdeadbeef", digest, len(body))
    # A same-content object arriving from an archive: content addressing means
    # the pending manifest declares a hash the store already held.
    _pending(v2_paths, "pending-1", digest, len(body))

    DeletionService("s1", v2_paths).delete_pending_import_manifests(["pending-1"])

    assert (v2_paths.objects / digest).exists(), (
        "the rollback derived reachability from visible manifests alone, so it "
        "unlinked an object the audit bundle is the last reference to"
    )
    assert not (v2_paths.committed / "pending-1.json").exists()
    assert not (v2_paths.integrity / "pending-1.sig").exists()


def test_import_rollback_keeps_objects_an_active_journal_still_references(v2_paths):
    from backend.services.snapshot_v2.deletion import DeletionService
    from backend.services.snapshot_v2.journal_store import JournalStore, make_envelope

    body = b"journal-pinned"
    digest = hashlib.sha256(body).hexdigest()
    (v2_paths.objects / digest).write_bytes(body)

    JournalStore(v2_paths).write(make_envelope(
        kind="restore_apply",
        transaction_id="restore-live",
        session_id="s1",
        status="applying",
        payload={"object_refs": {digest: {"sha256": digest}}},
    ))
    _pending(v2_paths, "pending-1", digest, len(body))

    DeletionService("s1", v2_paths).delete_pending_import_manifests(["pending-1"])

    assert (v2_paths.objects / digest).exists(), (
        "an in-flight transaction declared this object as a root and the "
        "rollback unlinked it out from under the running restore"
    )


# ── F19. A rollback never publishes what it failed to delete ─────────────────


def test_import_rollback_never_publishes_a_manifest_it_failed_to_delete(v2_paths):
    from backend.services.snapshot_v2.importer import SnapshotImporter
    from backend.services.snapshot_v2.object_store import ManifestStore

    body = b"payload"
    digest = hashlib.sha256(body).hexdigest()
    (v2_paths.objects / digest).write_bytes(body)

    # A pre-existing corrupt *visible* manifest — the state
    # list_committed_with_diagnostics exists to render, and the state that
    # makes reconcile() raise IMPORT_RECONCILE_CORRUPT in the first place.
    _publish(v2_paths, "pre-existing", digest, len(body))
    (v2_paths.integrity / "pre-existing.sig").write_text("00" * 32)

    _pending(v2_paths, "imported-1", digest, len(body))

    SnapshotImporter("s1", v2_paths)._rollback_pending_publish("tx", ["imported-1"])

    stems = ManifestStore(v2_paths)._iter_committed_json_stems()
    assert "imported-1" not in stems, (
        "the rollback cleared the pending-import marker although the delete "
        "had raised, publishing snapshots from an import the API reported as "
        "failed"
    )


def test_import_rollback_keeps_the_marker_when_the_delete_raises(v2_paths, monkeypatch):
    from backend.services.snapshot_v2.deletion import DeletionService
    from backend.services.snapshot_v2.importer import SnapshotImporter
    from backend.services.snapshot_v2.object_store import ManifestStore

    body = b"payload"
    digest = hashlib.sha256(body).hexdigest()
    (v2_paths.objects / digest).write_bytes(body)
    _pending(v2_paths, "imported-1", digest, len(body))

    def boom(self, snapshot_ids):
        raise OSError("read-only volume")

    monkeypatch.setattr(
        DeletionService, "delete_pending_import_manifests", boom,
    )

    SnapshotImporter("s1", v2_paths)._rollback_pending_publish("tx", ["imported-1"])

    assert ManifestStore._pending_marker_path(
        v2_paths.committed, "imported-1",
    ).exists(), (
        "clearing the marker after a failed delete makes the manifest visible; "
        "invisible-and-retryable is the only safe direction to fail"
    )


# ── F20. Import admits itself against the quota and reserve it is given ──────


def _one_object_archive(tmp_path, size=4096):
    from backend.tests.test_snapshot_v2_import import (
        _build_archive, _build_manifest, _make_ref,
    )

    data = b"y" * size
    digest = hashlib.sha256(data).hexdigest()
    manifest = _build_manifest(
        "src-session", "snap-a", objects={digest: _make_ref(digest, size=size)},
    )
    return _build_archive(
        tmp_path / "quota.tar.gz", [manifest], {digest: data}, "src-session",
    )


def test_import_refuses_to_exceed_the_session_quota(tmp_path, v2_paths):
    from backend.services.snapshot_v2.importer import SnapshotImporter
    from backend.services.snapshot_v2.object_store import SnapshotImportLimits

    archive = _one_object_archive(tmp_path)

    result = SnapshotImporter(
        "dest", v2_paths, limits=SnapshotImportLimits(session_quota_bytes=1),
    ).import_bundle(archive)

    assert not result.success
    assert any("snapshot_quota_exceeded" in e for e in result.errors), result.errors
    assert list(Path(v2_paths.objects).iterdir()) == [], (
        "the quota was checked only after pass 2 had already allocated, so an "
        "operator's session ceiling could not bound the import at all"
    )


def test_import_refuses_to_breach_the_free_space_reserve(tmp_path, v2_paths):
    from backend.services.snapshot_v2.importer import SnapshotImporter
    from backend.services.snapshot_v2.object_store import SnapshotImportLimits

    archive = _one_object_archive(tmp_path)

    result = SnapshotImporter(
        "dest", v2_paths,
        limits=SnapshotImportLimits(free_space_reserve_bytes=10 ** 18),
    ).import_bundle(archive)

    assert not result.success
    assert any("snapshot_quota_exceeded" in e for e in result.errors), result.errors
    assert list(Path(v2_paths.objects).iterdir()) == [], (
        "import wrote the data volume to zero free space, the condition the "
        "reserve exists to prevent for every other session on the host"
    )


def test_import_admits_a_bundle_that_fits(tmp_path, v2_paths):
    from backend.services.snapshot_v2.importer import SnapshotImporter
    from backend.services.snapshot_v2.object_store import SnapshotImportLimits

    archive = _one_object_archive(tmp_path)

    result = SnapshotImporter(
        "dest", v2_paths, limits=SnapshotImportLimits(),
    ).import_bundle(archive)

    assert result.success, result.errors


# ── F21. import-v2 stops reading at the limit it is configured with ──────────


def test_import_v2_rejects_a_body_past_the_compressed_archive_limit(monkeypatch):
    from fastapi import HTTPException

    from backend.config._config import default_config
    from backend.routes.export_import import import_v2
    from backend.routes.export_import_parts import import_routes

    monkeypatch.setattr(
        default_config, "snapshot_v2_import_compressed_bytes", 1024, raising=False,
    )

    async def fake_owner(session_id, user_id):
        return None

    monkeypatch.setattr(import_routes, "require_session_ownership", fake_owner)

    consumed = []

    class _Stream:
        async def __aiter__(self):
            for i in range(64):
                consumed.append(i)
                yield b"z" * 512

    class _Request:
        state = types.SimpleNamespace(user_id="u")
        headers: dict = {}

        def stream(self):
            return _Stream().__aiter__()

    with pytest.raises(HTTPException) as caught:
        asyncio.run(import_v2("sess", _Request()))

    assert caught.value.status_code == 413
    assert len(consumed) < 64, (
        "the whole body was written to the data volume before "
        "max_compressed_bytes could reject it"
    )


# ── F22. An export protection journal names the process that wrote it ────────


def test_export_protection_journal_records_its_writing_process(v2_paths):
    from backend.services.snapshot_v2.exporter import _write_export_protection
    from backend.services.snapshot_v2.journal_store import (
        JOURNAL_OWNER_ID, JournalStore,
    )

    _write_export_protection(v2_paths, "exp-crash", "s1", ["snap-a"])

    env = JournalStore(v2_paths).read("export-exp-crash")
    assert env.payload.get("owner") == JOURNAL_OWNER_ID, (
        "without an owner stamp recovery cannot tell a live export from crash "
        "debris, so an ACTIVE journal nothing reaps keeps every snapshot that "
        "existed at export time permanently undeletable"
    )


@pytest.mark.asyncio
async def test_recovery_reaps_an_export_protection_journal_from_a_dead_process(
    v2_paths,
):
    """An ACTIVE export journal whose writer is gone must not protect forever."""
    from backend.services.snapshot_v2.exporter import _write_export_protection
    from backend.services.snapshot_v2.journal_store import JournalStore
    from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

    _write_export_protection(v2_paths, "exp-dead", "sess-x", ["snap-a"])
    # Restamp with a foreign owner: this is what the journal of a process that
    # died mid-archive looks like to the next process that reads it.
    store = JournalStore(v2_paths)
    env = store.read("export-exp-dead")
    payload = dict(env.payload or {})
    payload["owner"] = "0" * 32
    from backend.services.snapshot_v2.journal_store import make_envelope

    store.write(make_envelope(
        kind="export_protection",
        transaction_id="export-exp-dead",
        session_id="sess-x",
        status="ACTIVE",
        payload=payload,
    ))

    result = await recover_snapshot_transactions(
        session_id="sess-x",
        paths=v2_paths,
        session_dir=str(v2_paths.root),
        workspace_root=str(v2_paths.root / "ws"),
        states_root=str(v2_paths.root / "states"),
    )

    assert not (Path(str(v2_paths.journals)) / "export-exp-dead.json").exists(), (
        "the ACTIVE export journal of a dead process was left in place, so "
        "compute_protected_snapshot_ids keeps answering 409 PROTECTED for "
        "every snapshot it names and GC can never reclaim their bytes"
    )
    assert "export-exp-dead" in result.recovered_transactions


@pytest.mark.asyncio
async def test_recovery_keeps_an_export_protection_journal_of_a_live_export(
    v2_paths,
):
    """Recovery runs on the event loop; an export runs in a worker thread."""
    from backend.services.snapshot_v2.exporter import _write_export_protection
    from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

    _write_export_protection(v2_paths, "exp-live", "sess-x", ["snap-a"])

    await recover_snapshot_transactions(
        session_id="sess-x",
        paths=v2_paths,
        session_dir=str(v2_paths.root),
        workspace_root=str(v2_paths.root / "ws"),
        states_root=str(v2_paths.root / "states"),
    )

    assert (Path(str(v2_paths.journals)) / "export-exp-live.json").exists(), (
        "an export owned by this process was reaped mid-run — recovery can "
        "land while the archive build is still in its to_thread worker, and "
        "removing the journal unprotects snapshots the export is still reading"
    )
