"""Regression tests for the 0.36.12 Snapshot V2 edge-case hardening pass.

One test per defect, each named for the failure it prevents rather than for the
function it calls. Every one of these was verified to fail against 0.36.11.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── shared minimal paths object ──────────────────────────────────────────────


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


# ── 1. HITL wait gates hand the safe point over to a waiting restore ─────────
#
# Every HITL gate blocks *inside* a graph node, so the run task stays in flight
# and the graph never reaches another node entry — the only place the
# cooperative safe point was otherwise acknowledged. A restore requested while
# the user sat at a gate therefore had no acknowledger at all and could only
# time out.


@pytest.mark.asyncio
async def test_hitl_wait_hands_the_safe_point_to_a_pending_restore():
    from backend.workflow.helpers import wait_with_skip_time

    event = asyncio.Event()
    calls = []

    async def hook():
        calls.append(True)
        # Stand-in for the restore committing and releasing the user.
        event.set()

    signaled = await wait_with_skip_time(
        event, 5, "sess-hitl", "round_wait", safe_point_hook=hook,
    )

    assert signaled is True
    assert calls, (
        "the gate never invoked the safe-point hook, so a restore requested "
        "while the user sits at the gate has nowhere to be acknowledged"
    )


@pytest.mark.asyncio
async def test_hitl_wait_without_a_hook_keeps_the_original_contract():
    from backend.workflow.helpers import wait_with_skip_time

    event = asyncio.Event()
    assert await wait_with_skip_time(event, 0, "s", "g") is False
    event.set()
    assert await wait_with_skip_time(event, -1, "s", "g") is True


@pytest.mark.asyncio
async def test_hitl_wait_hook_freeze_does_not_consume_the_countdown():
    """A restore hold must not silently trip the gate's auto-skip."""
    from backend.workflow import helpers

    event = asyncio.Event()
    frozen = {"done": False}

    async def hook():
        if not frozen["done"]:
            frozen["done"] = True
            await asyncio.sleep(0.35)
            event.set()

    # Countdown shorter than the freeze: without deadline extension the wait
    # would return False (auto-skip) instead of seeing the user's signal.
    orig = helpers.HITL_SAFE_POINT_POLL_SECONDS
    helpers.HITL_SAFE_POINT_POLL_SECONDS = 0.05
    try:
        signaled = await helpers.wait_with_skip_time(
            event, -1, "sess", "round_wait", safe_point_hook=hook,
        )
    finally:
        helpers.HITL_SAFE_POINT_POLL_SECONDS = orig
    assert signaled is True


def test_every_hitl_gate_passes_the_safe_point_hook():
    """A gate added without the hook silently reintroduces the deadlock."""
    import inspect

    from backend.workflow import annotation, callbacks, planning, synthesis

    for module in (synthesis, annotation, planning, callbacks):
        source = inspect.getsource(module)
        calls = source.count("await wait_with_skip_time(")
        hooks = source.count("safe_point_hook=getattr(self, \"await_restore_safe_point\"")
        assert calls == hooks, (
            f"{module.__name__} has {calls} HITL wait(s) but only {hooks} pass "
            f"safe_point_hook; a restore requested at that gate cannot be "
            f"acknowledged and will time out"
        )


# ── 2. A parked session freezes its gate, not Idle ───────────────────────────


def test_inactive_descriptor_still_records_the_parked_gate():
    from backend.models.schemas import TaskStatus
    from backend.services.snapshot_v2.continuation import ContinuationRegistry

    class _Store:
        def put_bytes(self, data, media_type):
            from backend.services.snapshot_v2.models import ObjectRef

            class _R:
                ref = ObjectRef(
                    sha256=hashlib.sha256(data).hexdigest(),
                    size=len(data),
                    media_type=media_type,
                )
            return _R()

    parked = MagicMock()
    parked.status = TaskStatus.WAITING_ANNOTATION
    parked.wait_input_reason = None

    # Exactly the descriptor RestoreJobManager._acknowledge_if_quiescent writes.
    capsule = ContinuationRegistry(_Store()).freeze_dynamic(
        {
            "kernel_present": False,
            "current_round": 3,
            "inactive": True,
            "status": "waiting_annotation",
        },
        analysis_state=parked,
        workflow_state={"current_round": 3},
    )

    assert capsule.kind == "semantic_gate", (
        "the quiescent-coordinator descriptor hard-codes inactive=True, so "
        "pre_restore — the snapshot a user reaches for to undo a restore — "
        f"froze as {capsule.kind!r} and lost the gate it was parked at"
    )
    assert capsule.gate == "annotation_confirmation"


def test_inactive_descriptor_still_idles_when_nothing_is_parked():
    from backend.models.schemas import TaskStatus
    from backend.services.snapshot_v2.continuation import ContinuationRegistry

    idle = MagicMock()
    idle.status = TaskStatus.PAUSED
    idle.wait_input_reason = None

    capsule = ContinuationRegistry(MagicMock()).freeze_dynamic(
        {"kernel_present": False, "current_round": 1, "inactive": True},
        analysis_state=idle,
        workflow_state={},
    )
    assert capsule.kind == "idle"


# ── 3. A restored gate is parked on the status its own endpoint accepts ──────


@pytest.mark.parametrize("gate,expected", [
    ("annotation_confirmation", "waiting_annotation"),
    ("plan_review", "waiting_plan_review"),
    ("round_wait", "waiting_input"),
    ("direction_selection", "waiting_input"),
    ("integration_selection", "waiting_input"),
])
def test_restored_gate_status_matches_its_live_counterpart(gate, expected):
    from backend.services.snapshot_v2.dispatcher import (
        ContinuationDispatcher, StagedCheckpoint,
    )

    staged = StagedCheckpoint(
        kind="semantic_gate",
        gate=gate,
        payload={"gate": gate, "current_round": 2},
        branch_id="restore-tx",
        branch_label="branch-abc",
        parent_snapshot_id="snap-1",
        transaction_id="tx-1",
    )
    state = MagicMock()
    state.sandbox_state = {}
    ContinuationDispatcher(MagicMock()).activate(state, staged)

    assert getattr(state.status, "value", state.status) == expected, (
        f"gate {gate!r} was restored onto {state.status!r}; its HITL endpoint "
        f"validates {expected!r} and rejects everything else, so the restored "
        f"session has no reachable action"
    )


# ── 4. The post-write disk gate compares against the reserve, not 2x payload ─


@pytest.mark.asyncio
async def test_post_write_disk_gate_does_not_double_count_the_payload(monkeypatch):
    import collections

    from backend.services.snapshot_v2.service_parts import capture_support

    svc = capture_support.CaptureSupportMixin.__new__(
        capture_support.CaptureSupportMixin,
    )
    svc.session_id = "s"
    svc.paths = _Paths(Path("/tmp/does-not-matter"))
    svc.session_quota_bytes = 10 ** 15
    svc.pre_restore_quota_grace_bytes = 0
    svc.free_space_reserve_bytes = 5 * 1024 ** 3
    svc._physical_object_store_bytes = lambda: 0

    manifest = MagicMock()
    manifest.physical_added_bytes = 20 * 1024 ** 3

    usage = collections.namedtuple("usage", "total used free")
    # 10 GiB free *after* the 20 GiB write — comfortably above the 5 GiB
    # reserve. The old arithmetic computed 10 - 20 = -10 and refused.
    monkeypatch.setattr(
        capture_support.shutil, "disk_usage",
        lambda _p: usage(total=0, used=0, free=10 * 1024 ** 3),
    )

    await svc._check_quota(manifest)


# ── 5. put_file cleans up its staging temp on failure ────────────────────────


def test_put_file_does_not_leak_its_staging_temp_on_error(v2_paths, monkeypatch):
    from backend.services.snapshot_v2.object_store import ObjectStore

    src = v2_paths.root.parent / "big.bin"
    src.write_bytes(b"x" * 4096)

    store = ObjectStore(v2_paths)

    real_replace = __import__("os").replace

    def boom(a, b):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("backend.services.snapshot_v2.object_store.os.replace", boom)
    with pytest.raises(OSError):
        store.put_file(src, "application/octet-stream")
    monkeypatch.setattr("backend.services.snapshot_v2.object_store.os.replace", real_replace)

    leaked = [p.name for p in v2_paths.staging.iterdir() if p.is_file()]
    assert leaked == [], (
        f"put_file leaked {leaked} in staging/; nothing reclaims dot-prefixed "
        f"temps, so every HITL Retry of a failing capture leaves another copy"
    )


def test_reconcile_leaves_object_staging_temps_alone(v2_paths):
    from backend.services.snapshot_v2.object_store import ManifestStore

    temp = v2_paths.staging / f".{'a' * 64}.tmp1234"
    temp.write_bytes(b"in flight")
    manifest_staging = v2_paths.staging / "tx1_snap-1.json"
    manifest_staging.write_text("{}")

    ManifestStore(v2_paths).reconcile()

    assert temp.exists(), "reconcile unlinked an in-flight object staging temp"
    assert not manifest_staging.exists()


# ── 6. Undeclared WorkflowState keys never cross a node boundary ─────────────
#
# Pins the LangGraph semantics the restore routing relies on. `restore_entry`
# carries `_restore_entry_plan` to its own conditional edge (passthrough, works)
# but `_preserve_round_on_initialize` / `_restore_resume_node` to a *later* node
# (dropped). If a future upgrade started persisting undeclared keys,
# `_should_continue_after_init` would begin re-routing every round to the
# restore target — so the assumption is worth a guard rather than a comment.


@pytest.mark.asyncio
async def test_undeclared_workflow_state_keys_do_not_reach_the_next_node():
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    class _S(TypedDict, total=False):
        declared: int

    seen: dict = {}

    async def first(state):
        return {**state, "declared": 1, "_undeclared": "x"}

    async def second(state):
        seen["second_in"] = dict(state)
        return {**state, "declared": 2}

    def route(state):
        seen["router_in"] = dict(state)
        return "to_second"

    g = StateGraph(_S)
    g.add_node("first", first)
    g.add_node("second", second)
    g.set_entry_point("first")
    g.add_conditional_edges("first", route, {"to_second": "second"})
    g.add_edge("second", END)

    await g.compile().ainvoke({"declared": 0})

    assert "_undeclared" in seen["router_in"], (
        "restore_entry hands _restore_entry_plan to its own conditional edge "
        "through this passthrough"
    )
    assert "_undeclared" not in seen["second_in"], (
        "an undeclared key reaching a later node would make "
        "_should_continue_after_init re-route every round to the restore target"
    )


# ── 7. Not-found is decided by the code, not the producer class ──────────────


def test_missing_snapshot_delete_is_404_not_507():
    from backend.services.snapshot_v2.errors import (
        SnapshotStorageError, snapshot_error_detail,
    )

    detail = snapshot_error_detail(
        SnapshotStorageError("SNAPSHOT_NOT_FOUND", "Snapshot snap-x not found"),
    )
    assert detail["status_code"] == 404, (
        "a DELETE of an already-deleted snapshot answered 507 Insufficient "
        "Storage, telling an idempotently-retrying client the store is full"
    )


def test_real_storage_failures_still_map_to_507():
    from backend.services.snapshot_v2.errors import (
        SnapshotStorageError, snapshot_error_detail,
    )

    detail = snapshot_error_detail(
        SnapshotStorageError("snapshot_quota_exceeded", "over quota"),
    )
    assert detail["status_code"] == 507


# ── 8. A crashed restore job stops protecting its target forever ─────────────


def test_abandoned_restore_job_journal_is_marked_failed(v2_paths):
    from backend.services.snapshot_v2.journal_store import JournalStore, make_envelope
    from backend.services.snapshot_v2.recovery import _recover_restore_job_envelope

    store = JournalStore(v2_paths)
    env = make_envelope(
        kind="restore_job",
        transaction_id="restore-job-abandoned",
        session_id="sess-1",
        status="waiting_safe_point",
        payload={"snapshot_id": "snap-target", "recovery_mode": "rollback"},
    )
    store.write(env)

    recovered: list = []
    _recover_restore_job_envelope(
        env, paths=v2_paths, session_id="sess-1", recovered_txns=recovered,
    )

    reloaded = store.read("restore-job-abandoned")
    assert reloaded.status == "failed", (
        "a restore_job journal left non-terminal by a crash kept its target "
        "snapshot protected from deletion and pinned status polls at "
        "waiting_safe_point forever, with no API able to clear it"
    )
    assert recovered == ["restore-job-abandoned"]


def test_live_restore_job_journal_is_left_alone(v2_paths, monkeypatch):
    from backend.services.snapshot_v2 import recovery as recovery_mod
    from backend.services.snapshot_v2.journal_store import JournalStore, make_envelope

    store = JournalStore(v2_paths)
    env = make_envelope(
        kind="restore_job",
        transaction_id="restore-job-live",
        session_id="sess-1",
        status="waiting_safe_point",
        payload={"snapshot_id": "snap-target"},
    )
    store.write(env)

    from backend.state import snapshot_restore_jobs

    monkeypatch.setitem(snapshot_restore_jobs, "restore-job-live", MagicMock())

    recovered: list = []
    recovery_mod._recover_restore_job_envelope(
        env, paths=v2_paths, session_id="sess-1", recovered_txns=recovered,
    )

    assert store.read("restore-job-live").status == "waiting_safe_point"
    assert recovered == []


# ── 9. Rollback does not leave the target's residual pkl installed ───────────


def test_rollback_reports_that_it_rewrote_session_json():
    """The in-memory cache still holds the target state after a rollback."""
    import inspect

    from backend.services.snapshot_v2 import recovery
    from backend.services import session_store

    assert "session_state_rewritten" in inspect.getsource(recovery), (
        "recovery must report when it rewrote session.json"
    )
    assert "session_state_rewritten" in inspect.getsource(session_store), (
        "ensure_session_loaded must evict the cached target state after a "
        "rollback, or it writes that state straight back over session.json"
    )


def test_rollback_quarantines_a_residual_the_pre_restore_never_had():
    import inspect

    from backend.services.snapshot_v2 import recovery

    source = inspect.getsource(recovery)
    assert "quarantine_live_runtime_pkl" in source, (
        "rollback must mirror apply()'s else-branch: with no pre_restore "
        "residual the TARGET's pkl stays installed with a valid signature"
    )


# ── 10. /human-input cannot release a fail-closed snapshot gate ──────────────


def test_human_input_refuses_while_the_snapshot_gate_is_armed():
    import inspect

    from backend.routes import interaction

    source = inspect.getsource(interaction.send_human_input)
    assert "_refuse_if_snapshot_blocked" in source, (
        "/human-input signals the same _continue_event the round gate waits "
        "on, so without this guard it releases the snapshot gate too"
    )
    assert "is_snapshot_failure_blocked" in inspect.getsource(
        interaction._refuse_if_snapshot_blocked
    )


# ── 11. The live pkl wins over a stale h5ad sidecar ──────────────────────────


def test_restore_template_prefers_the_pkl_over_a_stale_sidecar():
    import inspect

    from backend.utils import sandbox

    source = inspect.getsource(sandbox)
    assert "if _name in _restored_names:\n                continue" in source, (
        "the sidecar set is never retired, so an unconditional rebind rolled "
        "the namespace back to the restored snapshot on every later rehydrate"
    )


# ── 12. apply() replaces the whole WAL-mode checkpoint triple ────────────────


def test_apply_removes_stale_checkpoint_wal_companions():
    import inspect

    from backend.services.snapshot_v2.checkpoint_paths import SQLITE_COMPANION_SUFFIXES
    from backend.services.snapshot_v2.restore_parts import coordinator

    assert SQLITE_COMPANION_SUFFIXES == ("-wal", "-shm")
    source = inspect.getsource(coordinator.RestoreCoordinator.apply)
    assert "SQLITE_COMPANION_SUFFIXES" in source, (
        "orphan -wal/-shm left by an unclean shutdown are replayed over the "
        "restored checkpoint when no workflow object exists to close them"
    )


# ── 13. GC protects the audit bundle it was told to retain ───────────────────


def test_gc_marks_audit_bundle_objects(v2_paths):
    from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
    from backend.tests.snapshot_v2_fixtures import (
        make_snapshot_v2_manifest, write_manifest_to_store,
    )
    from backend.services.snapshot_v2.models import ObjectRef

    digest = hashlib.sha256(b"discarded-state").hexdigest()
    (v2_paths.objects / digest).write_bytes(b"discarded-state")

    manifest = make_snapshot_v2_manifest(
        session_id="s1",
        snapshot_id="audit-deadbeefdeadbeef",
        objects={
            digest: ObjectRef(
                sha256=digest, size=15, media_type="application/octet-stream",
            )
        },
    )
    write_manifest_to_store(manifest, v2_paths.committed, v2_paths.integrity)

    gc = SnapshotGarbageCollector("s1", v2_paths)
    gc.sweep(gc.mark())

    assert (v2_paths.objects / digest).exists(), (
        "the audit bundle is the only record of the state a discard_current "
        "restore threw away, and GC swept its objects on the next request"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Second hardening batch
# ═══════════════════════════════════════════════════════════════════════════


# ── 14. A runtime-omitted export describes what it actually packed ───────────


def test_export_without_runtime_emits_a_self_consistent_manifest(v2_paths, monkeypatch):
    import hashlib as _h

    from backend.services.snapshot_v2.exporter import _without_omitted_runtime
    from backend.services.snapshot_v2.models import ObjectRef, RuntimeManifest
    from backend.services.snapshot_v2.object_store import canonical_json
    from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

    base_h = _h.sha256(b"base").hexdigest()
    rt_h = _h.sha256(b"runtime").hexdigest()
    base_ref = ObjectRef(sha256=base_h, size=4, media_type="application/json")
    rt_ref = ObjectRef(sha256=rt_h, size=7, media_type="application/x-hdf5")

    manifest = make_snapshot_v2_manifest(
        session_id="s1",
        snapshot_id="snap-rt",
        objects={base_h: base_ref, rt_h: rt_ref},
        runtime=RuntimeManifest(
            status="complete",
            dependencies={"required": {}, "incidental_skips": []},
            adata_refs={"adata": rt_ref},
        ),
    )

    emitted = _without_omitted_runtime(manifest, {base_h})

    assert rt_h not in emitted.objects, (
        "the emitted manifest still declares a hash the archive does not carry, "
        "so the importer rejects it with MISSING_OBJECT"
    )
    assert base_h in emitted.objects
    assert emitted.runtime.status == "missing"
    assert emitted.runtime.adata_refs == {}
    # The importer re-derives root_hash and refuses a mismatch.
    assert emitted.root_hash == _h.sha256(canonical_json(emitted)).hexdigest()
    # Untouched when nothing was omitted.
    assert _without_omitted_runtime(manifest, {base_h, rt_h}) is manifest


# ── 15. Import limits run before the allocation they bound ───────────────────


def test_import_bounds_the_members_it_materialises(tmp_path, v2_paths):
    import io
    import tarfile

    from backend.services.snapshot_v2.errors import SnapshotStorageError
    from backend.services.snapshot_v2.importer import SnapshotImporter
    from backend.services.snapshot_v2.object_store import SnapshotImportLimits

    archive = tmp_path / "bomb.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for i in range(40):
            info = tarfile.TarInfo(name=f"objects/{i:064x}")
            info.size = 0
            tf.addfile(info, io.BytesIO(b""))

    importer = SnapshotImporter(
        "dest", v2_paths, limits=SnapshotImportLimits(max_member_count=5),
    )
    result = importer.import_bundle(archive)
    assert not result.success
    assert any("TOO_MANY_MEMBERS" in e for e in result.errors), result.errors


def test_import_caps_the_members_it_reads_whole(tmp_path, v2_paths):
    import io
    import json as _json
    import tarfile

    from backend.services.snapshot_v2.importer import SnapshotImporter
    from backend.services.snapshot_v2.object_store import SnapshotImportLimits

    # Metadata is read whole into memory; before the cap it was bounded only by
    # max_single_member_bytes (80 GB) and excluded from the uncompressed budget.
    payload = _json.dumps({"format_version": 2, "pad": "x" * 200_000}).encode()
    archive = tmp_path / "fat-meta.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo(name="snapshot-v2-export.json")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    importer = SnapshotImporter(
        "dest", v2_paths, limits=SnapshotImportLimits(max_metadata_bytes=1024),
    )
    result = importer.import_bundle(archive)
    assert not result.success
    assert any("MEMBER_TOO_LARGE" in e for e in result.errors), result.errors


# ── 16. A legacy import releases the session id it claimed ───────────────────


def test_failed_zip_import_releases_the_session_directory(monkeypatch, tmp_path):
    import io
    import zipfile

    from fastapi import HTTPException

    from backend.routes.export_import_parts import import_impl

    session_dirs: dict[str, Path] = {}

    def fake_allocate(archive_session_id: str, _user_id: str) -> str:
        d = tmp_path / "sessions" / archive_session_id
        d.mkdir(parents=True, exist_ok=False)
        session_dirs[archive_session_id] = d
        return archive_session_id

    cleaned: list[str] = []

    def fake_cleanup(session_id):
        cleaned.append(session_id)

    monkeypatch.setattr(import_impl, "_allocate_import_session_id", fake_allocate)
    monkeypatch.setattr(import_impl, "_cleanup_partial_session", fake_cleanup)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # A top-level dir but no state.json — the truncated-download case.
        zf.writestr("sess-a/notebook.ipynb", "{}")

    with pytest.raises(HTTPException):
        import_impl._import_zip(buf.getvalue(), "user-1")

    assert cleaned == ["sess-a"], (
        "the claimed session directory was never released, so it holds the "
        "archive's own id forever and a corrected re-upload lands elsewhere"
    )


# ── 17. A terminal restore keeps the reports it is restoring ─────────────────


def test_terminal_restore_keeps_report_paths():
    from backend.services.snapshot_v2.dispatcher import (
        ContinuationDispatcher, StagedCheckpoint,
    )

    staged = StagedCheckpoint(
        kind="terminal",
        terminal_status="completed",
        branch_id="restore-tx",
        branch_label="branch-abc",
        parent_snapshot_id="snap-1",
        transaction_id="tx-1",
    )
    state = MagicMock()
    state.sandbox_state = {}
    state.detailed_report_html_path = "/reports/detailed.html"
    state.exploration_report_html_path = "/reports/exploration.html"

    ContinuationDispatcher(MagicMock()).activate(state, staged)

    assert state.detailed_report_html_path == "/reports/detailed.html", (
        "a terminal restore cannot regenerate reports — run() and resume() both "
        "short-circuit on restore_resume_kind == 'terminal' — so nulling the "
        "paths left every report view answering 404 with no way back"
    )
    assert state.exploration_report_html_path == "/reports/exploration.html"


def test_non_terminal_restore_still_invalidates_report_paths():
    from backend.services.snapshot_v2.dispatcher import (
        ContinuationDispatcher, StagedCheckpoint,
    )

    staged = StagedCheckpoint(
        kind="semantic_gate",
        gate="round_wait",
        payload={"gate": "round_wait", "current_round": 1},
        branch_id="restore-tx",
        branch_label="branch-abc",
        parent_snapshot_id="snap-1",
        transaction_id="tx-1",
    )
    state = MagicMock()
    state.sandbox_state = {}
    ContinuationDispatcher(MagicMock()).activate(state, staged)
    assert state.detailed_report_html_path is None


# ── 18. A capture never spans two kernel generations ─────────────────────────


@pytest.mark.asyncio
async def test_capture_refuses_a_kernel_that_restarted_underneath_it():
    from backend.services.snapshot_v2.errors import SnapshotCaptureError
    from backend.services.snapshot_v2.runtime import RuntimeCapture

    capture = RuntimeCapture(MagicMock(), MagicMock())

    class _Sandbox:
        kernel_epoch = 3

        async def execute(self, *a, **k):
            # Stand-in for the pre-execute liveness guard restarting the kernel
            # and rehydrating it from the last saved pkl.
            self.kernel_epoch += 1
            return MagicMock(success=True, output="", error=None)

    with pytest.raises(SnapshotCaptureError) as exc:
        await capture._run_kernel(_Sandbox(), "print(1)")
    assert exc.value.code == "required_runtime_unpersistable"


@pytest.mark.asyncio
async def test_capture_accepts_a_stable_kernel():
    from backend.services.snapshot_v2.runtime import RuntimeCapture

    capture = RuntimeCapture(MagicMock(), MagicMock())

    class _Sandbox:
        kernel_epoch = 3

        async def execute(self, *a, **k):
            return MagicMock(success=True, output="ok", error=None)

    result = await capture._run_kernel(_Sandbox(), "print(1)")
    assert result.output == "ok"


# ── 19. The pre-write estimate accounts for the runtime ──────────────────────


def test_pre_write_estimate_counts_the_projected_runtime(v2_paths):
    import hashlib as _h

    from backend.services.snapshot_v2.models import ObjectRef, RuntimeManifest
    from backend.services.snapshot_v2.service_parts import capture_support
    from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

    svc = capture_support.CaptureSupportMixin.__new__(
        capture_support.CaptureSupportMixin,
    )
    svc.session_id = "s"
    svc.paths = v2_paths

    rt_h = _h.sha256(b"big-adata").hexdigest()
    previous = make_snapshot_v2_manifest(
        session_id="s",
        snapshot_id="snap-prev",
        runtime=RuntimeManifest(
            status="complete",
            dependencies={"required": {}, "incidental_skips": []},
            adata_refs={
                "adata": ObjectRef(
                    sha256=rt_h, size=8 * 1024 ** 3, media_type="application/x-hdf5",
                )
            },
        ),
    )
    svc._resolve_previous_manifest = lambda _state: previous

    request = MagicMock()
    request.runtime = None
    sandbox = MagicMock()
    sandbox.km = object()

    projected = svc._estimate_runtime_addition(
        request, MagicMock(), sandbox=sandbox,
    )
    assert projected == 8 * 1024 ** 3, (
        "the pre-write ceiling counted only state/notebook/checkpoint/workspace "
        "— a few MB against the tens of GB a runtime capture writes — so it "
        "could never fire before the kernel had been driven and the bytes written"
    )

    # A hash already held in objects/ does NOT zero the projection.
    # The head's refs model the bytes this capture is about to write; they are
    # not those bytes. GC marks every object of a committed lineage head
    # reachable, so the head's own runtime objects are in objects/ by
    # construction — deduping against that set returned 0 for exactly the
    # steady state the pre-I/O ceiling exists for. Genuine reuse is
    # short-circuited upstream instead, by `request.runtime is not None`.
    (v2_paths.objects / rt_h).write_bytes(b"big-adata")
    assert svc._estimate_runtime_addition(
        request, MagicMock(), sandbox=sandbox,
    ) == 8 * 1024 ** 3
    # No kernel → nothing to capture.
    assert svc._estimate_runtime_addition(
        request, MagicMock(), sandbox=None,
    ) == 0


# ── 20. A present canonical artifact does not vanish from a "complete" run ───


def test_present_canonical_artifact_failing_dill_is_fail_closed():
    import inspect

    from backend.services.snapshot_v2.runtime_parts import capture

    source = inspect.getsource(capture)
    assert "strict_names = mandatory_names |" in source, (
        "checkpoint_dependencies has no production writer, so the fail-closed "
        "tier was empty and a live canonical artifact that dill failed on was "
        "silently dropped from a runtime published as complete"
    )


def test_checkpoint_dependencies_are_read_from_the_safe_point_descriptor():
    import inspect

    from backend.services.snapshot_v2.service_parts import capture_steps

    source = inspect.getsource(
        capture_steps.CaptureStepsMixin._collect_checkpoint_dependencies,
    )
    assert "self.safe_point.get_descriptor" in source, (
        "the descriptor is the channel a boundary producer actually has; "
        "omitting it was half of why the tier had no writer"
    )


# ── 21. The discard-current escape hatch is reachable ────────────────────────


def test_pre_restore_failure_reports_the_code_the_client_branches_on():
    import inspect

    from backend.services.snapshot_v2 import service

    source = inspect.getsource(service.SnapshotService.restore_snapshot)
    assert source.count('"code": "PRE_RESTORE_UNAVAILABLE"') == 3, (
        "every way the pre-restore safety capture can fail must report the one "
        "code the client branches on to offer discard_current; without it that "
        "path — and the whole audit-bundle branch behind it — was dead code"
    )


# ── 22. A post-publish session-save failure is reconciled ────────────────────


def test_session_reconciliation_pending_is_consumed():
    import inspect

    from backend.services.snapshot_v2 import service

    source = inspect.getsource(service.SnapshotService.capture_required_snapshot)
    assert "record.session_reconciliation_pending" in source, (
        "_capture_impl deliberately does not raise when the post-publish "
        "session save fails and signals it with this flag instead; nothing "
        "read it, so the escape hatch had no other end"
    )


# ── 23. A boundary capture that loses the safe point retries once ────────────


def test_boundary_capture_retries_once_after_losing_the_safe_point():
    import inspect

    from backend.workflow import core

    source = inspect.getsource(core.SingleCellAnalysisWorkflow._create_auto_snapshot)
    assert 'getattr(exc, "code", "") == "SAFE_POINT_BUSY"' in source, (
        "the pending-request yield is a check-then-act; a restore reserving the "
        "safe point in the window turned the required capture into a gate that "
        "unwound the graph, leaving the restore with nobody left to acknowledge it"
    )
