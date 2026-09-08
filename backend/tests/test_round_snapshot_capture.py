"""Capturing one decision snapshot. Part B spec 4.5.

The writer copies bytes an ordinary save already committed. What these tests
are about is the ways that can go wrong without anyone noticing: a bundle that
declares a file it does not hold, a dump replaced by the next save while the
capture was reading it, a hung target that stops the analysis instead of
stopping itself, and a workspace entry that is not a file at all.

Imports are guarded so the module still collects on a tree without the feature:
a module that raises during collection publishes no evidence, which is exactly
the state a red-tree run has to read.
"""
import asyncio
import hashlib
import json
import os
import time

import pytest

from backend.constants import (
    GATE_ANNOTATION_CONFIRMATION,
    GATE_DIRECTION_SELECTION,
    GATE_ROUND_WAIT,
)

try:
    from backend.config import _config
    from backend.services.round_snapshots import layout
    from backend.services.round_snapshots.layout import (
        MANIFEST_NAME,
        PAIR_EXECUTOR_RELPATH,
        RUNTIME_PKL_RELPATH,
    )
    from backend.services.round_snapshots.listing import read_session_listing
    from backend.services.round_snapshots.manifest import (
        read_snapshot_manifest,
        validate_continuation,
    )
    from backend.services.round_snapshots import writer as writer_module
    from backend.services.round_snapshots.writer import (
        PIN_DIRNAME,
        capture_decision_snapshot,
        reclaim_abandoned_capture_work,
    )
except ImportError as exc:  # the module under test does not exist yet
    _IMPORT_ERROR = exc
    MANIFEST_NAME = "manifest.json"
    RUNTIME_PKL_RELPATH = "runtime/session.pkl"
    PAIR_EXECUTOR_RELPATH = "runtime/pair/executor_checkpoint.json"
    PIN_DIRNAME = "snapshot-pins"
else:
    _IMPORT_ERROR = None


@pytest.fixture(autouse=True)
def _require_module_under_test():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"module under test is missing: {_IMPORT_ERROR}")


# ------------------------------------------------------------------ harness


class _Notebook:
    def __init__(self, session_id: str, revision: int, digest: str, current: int):
        from backend.config import get_session_states, get_session_workspace

        self.session_id = session_id
        self.state_dir = get_session_states(session_id)
        self.working_dir = get_session_workspace(session_id)
        self.session_path = self.state_dir / f"{session_id}_session.pkl"
        self.runtime_persistence = {
            "committed_revision": revision,
            "current_revision": current,
            "saved_revision": revision,
            "committed_source": "regular_save",
            "committed_digest": digest,
            "last_attempt_status": "succeeded",
            "status": "current",
        }

    def has_session_dump(self) -> bool:
        return self.session_path.is_file()


class _Sandbox:
    """Only what the writer touches: the lock, the record, and a save.

    The save is the real contract in miniature -- it advances the committed
    revision to the current one and lays down the pair beside it -- because the
    writer's freshness rule is stated in terms of exactly that.
    """

    def __init__(
        self, session_id: str, revision: int, digest: str, *,
        current: int | None = None, round_number: int = 3,
    ):
        self.session_id = session_id
        self._exec_lock = asyncio.Lock()
        self.km = object()
        self.notebook_state = _Notebook(
            session_id, revision, digest, revision if current is None else current,
        )
        self.round_number = round_number
        self.saves = 0
        self.save_result = True
        self.synced = []

    async def _save_session_locked(self):
        self.saves += 1
        if self.save_result is not True:
            return self.save_result
        record = self.notebook_state.runtime_persistence
        revision = int(record["current_revision"])
        record["committed_revision"] = revision
        record["saved_revision"] = revision
        record["last_attempt_status"] = "succeeded"
        _write_pair(self.session_id, revision, round_number=self.round_number)
        _rewrite_runtime_manifest(self.session_id, revision)
        return True

    def sync_runtime_persistence_state(self, state):
        self.synced.append(state)
        return dict(self.notebook_state.runtime_persistence)


def _digest_of(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint_db(path, session_id: str, round_number: int) -> None:
    """A real graph checkpoint, because the restore reads one back.

    Placeholder bytes were enough while only the capture was under test -- it
    copies the file and hashes it -- but a restore opens it through LangGraph's
    own serializer and pulls the root position out, so the harness has to
    produce something that actually is one.
    """
    import sqlite3

    from backend.models.schemas import AnalysisState
    from backend.utils.checkpoint import DurableJsonPlusSerializer

    serde = DurableJsonPlusSerializer(allowed_json_modules=[
        ("backend.models.schemas", name)
        for name in ("AgentType", "AnalysisState", "AnalysisPhase",
                     "RoundResult", "TaskStatus", "MessageRole")
    ])
    kind, blob = serde.dumps_typed({
        "v": 1,
        "id": "checkpoint-1",
        "channel_values": {
            "analysis_state": AnalysisState(
                id=session_id, name="Liver atlas", user_id="owner-1",
                current_round=round_number,
            ),
            "current_round": round_number,
            "planning_stage": "full",
            "plan_approved": True,
            "should_continue": True,
        },
        "channel_versions": {},
        "versions_seen": {},
    })
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints "
            "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
            "type TEXT, checkpoint BLOB)"
        )
        connection.execute("DELETE FROM checkpoints")
        connection.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
            (session_id, "", "checkpoint-1", kind, blob),
        )


def _write_pair(session_id: str, revision: int, *, round_number: int = 3) -> None:
    from backend.config import get_session_states
    from backend.utils.sandbox_parts.regular_save import GENERATION_PAIR_DIRNAME

    pair = get_session_states(session_id) / GENERATION_PAIR_DIRNAME / str(revision)
    pair.mkdir(parents=True, exist_ok=True)
    (pair / "state.json").write_text(json.dumps({
        "session_id": session_id,
        "lifecycle_generation": 1,
        # The notebook metadata a real commit captures beside the dump. A
        # restore projects the new session's persistence record from this, and
        # coherence is decided on the result, so a placeholder here would make
        # every restored session read as having no runtime at all.
        "runtime_persistence": {
            "status": "current",
            "current_revision": revision,
            "saved_revision": revision,
            "committed_revision": revision,
            "committed_source": "regular_save",
            "committed_digest": _digest_of(
                get_session_states(session_id) / f"{session_id}_session.pkl"
            ),
            "committed_pairing": {"state_version": 1189, "checkpoint": "d" * 64},
            "last_attempt_status": "succeeded",
            # Omitting this reads as "committed a revision nobody attempted",
            # which coherence calls inconsistent -- a real save always records it.
            "last_attempt_revision": revision,
            "last_saved_at": "2026-09-08T10:00:00+00:00",
            "lifecycle_generation": 1,
        },
    }, indent=2))
    _write_checkpoint_db(pair / "checkpoints.sqlite", session_id, round_number)


def _rewrite_runtime_manifest(session_id: str, revision: int) -> None:
    from backend.config import get_session_states
    from backend.utils.runtime_manifest import (
        build_runtime_manifest,
        runtime_manifest_path,
        write_runtime_manifest,
    )

    states = get_session_states(session_id)
    pkl = states / f"{session_id}_session.pkl"
    write_runtime_manifest(
        runtime_manifest_path(session_id, states_dir=states),
        build_runtime_manifest(
            revision=revision,
            state_revision=revision,
            pkl_digest=_digest_of(pkl),
            # `load_loom` is here because a carried definition has to be a
            # name the dump actually holds -- the generation manifest
            # refuses a record that claims otherwise.
            saved_names=["adata", "markers", "load_loom"],
            skipped_names=[
                {"name": "ss", "type": "SingleSampleGSEA", "reason": "serialization_failed"},
            ],
            serializer_version="dill-zstd-1",
            carried_definitions=[{"name": "load_loom", "kind": "function"}],
        ),
    )


def _seed_session(
    session_id: str, *, revision: int = 57, dump: bytes = b"a dump",
    round_number: int = 3,
) -> str:
    """A session on disk with one committed generation and its pair."""
    from backend.config import (
        ensure_session_dirs,
        get_session_dir,
        get_session_states,
        get_session_uploads,
        get_session_workspace,
    )
    from backend.utils.sandbox import _integrity_dir, _write_pkl_signature

    ensure_session_dirs(session_id)
    states = get_session_states(session_id)
    pkl = states / f"{session_id}_session.pkl"
    pkl.write_bytes(dump)
    _write_pkl_signature(pkl, _integrity_dir(session_id))
    _rewrite_runtime_manifest(session_id, revision)
    _write_pair(session_id, revision, round_number=round_number)
    (states / f"{session_id}.ipynb").write_text(json.dumps({"cells": []}))

    workspace = get_session_workspace(session_id)
    (workspace / "figures").mkdir(parents=True, exist_ok=True)
    (workspace / "figures" / "umap.png").write_bytes(b"PNG bytes")
    (workspace / "table.csv").write_text("gene,logfc\nCD3E,2.1\n")

    uploads = get_session_uploads(session_id)
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "matrix.h5ad").write_bytes(b"an input matrix")

    (get_session_dir(session_id) / "blobs").mkdir(parents=True, exist_ok=True)
    (get_session_dir(session_id) / "blobs" / ("c" * 64 + ".b64")).write_text("blob")
    return _digest_of(pkl)


def _state(session_id: str, *, round_number: int = 3, name: str = "Liver atlas"):
    from backend.models.schemas import AnalysisState

    return AnalysisState(
        id=session_id,
        name=name,
        user_id="owner-1",
        current_round=round_number,
        state_version=1189,
    )


class _Workflow:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.session_id = sandbox.session_id


@pytest.fixture
def capture(monkeypatch, tmp_path):
    """A seeded session, a fake sandbox, and a snapshot root to write into."""
    import uuid

    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path / "snapshots")

    def _build(*, revision=57, current=None, round_number=3, dump=b"a dump"):
        session_id = "snap-" + uuid.uuid4().hex[:12]
        digest = _seed_session(
            session_id, revision=revision, dump=dump, round_number=round_number,
        )
        sandbox = _Sandbox(
            session_id, revision, digest, current=current, round_number=round_number,
        )
        return _Workflow(sandbox), _state(session_id, round_number=round_number)

    return _build


def _bundle(session_id: str, snapshot: str):
    return layout.snapshot_dir(session_id, snapshot)


# ------------------------------------------------------------------ happy path


@pytest.mark.asyncio
async def test_a_committed_generation_is_captured_whole(capture):
    workflow, state = capture()
    snapshot = await capture_decision_snapshot(
        workflow, state, gate=GATE_ROUND_WAIT, workflow_state={"current_round": 3},
    )

    assert snapshot == "r003-round-g57"
    bundle = _bundle(state.id, snapshot)
    manifest = read_snapshot_manifest(bundle / MANIFEST_NAME)
    assert manifest["round"] == 3
    assert manifest["gate"] == GATE_ROUND_WAIT
    assert manifest["runtime_revision"] == 57
    assert manifest["retention_rank"] == 4
    assert manifest["source_session_name"] == "Liver atlas"

    # Every declared file is there, and is the file the manifest says it is.
    for relpath, entry in manifest["files"].items():
        target = bundle / relpath
        assert target.is_file(), relpath
        assert _digest_of(target) == entry["sha256"], relpath
        assert target.stat().st_size == entry["size"], relpath


@pytest.mark.asyncio
async def test_the_bundle_carries_the_dump_the_generation_manifest_names(capture):
    from backend.config import get_session_states
    from backend.utils.runtime_manifest import read_runtime_manifest, runtime_manifest_path

    workflow, state = capture()
    snapshot = await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)

    committed = read_runtime_manifest(
        runtime_manifest_path(state.id, states_dir=get_session_states(state.id))
    )
    manifest = read_snapshot_manifest(_bundle(state.id, snapshot) / MANIFEST_NAME)
    assert manifest["pkl_sha256"] == committed["pkl_sha256"]
    assert _digest_of(_bundle(state.id, snapshot) / RUNTIME_PKL_RELPATH) == committed["pkl_sha256"]


@pytest.mark.asyncio
async def test_the_bundle_is_self_sufficient(capture):
    """Notebook, inputs, workspace, blobs and the paired positions, all of it.

    A restore reads only what is under the bundle and the session's shared
    uploads; anything the capture leaves behind is something the new session
    would have to borrow from a source that may be gone.
    """
    workflow, state = capture()
    snapshot = await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)
    manifest = read_snapshot_manifest(_bundle(state.id, snapshot) / MANIFEST_NAME)

    declared = set(manifest["files"])
    assert "session.json" in declared
    assert "continuation.json" in declared
    assert f"notebooks/{state.id}.ipynb" in declared
    assert "workspace/table.csv" in declared
    assert "workspace/figures/umap.png" in declared
    assert any(name.startswith("blobs/") for name in declared)
    assert "runtime/pair/state.json" in declared
    assert "runtime/pair/checkpoints.sqlite" in declared
    assert manifest["uploads"]["matrix.h5ad"]["size"] == len(b"an input matrix")
    # No executor position at a round boundary: the executor has finished.
    assert manifest["executor_checkpoint_present"] is False


@pytest.mark.asyncio
async def test_the_capsule_arms_the_decision_the_snapshot_precedes(capture):
    workflow, state = capture()
    snapshot = await capture_decision_snapshot(
        workflow, state, gate=GATE_DIRECTION_SELECTION,
        workflow_state={"current_round": 3, "planning_stage": "stage2", "_private": 1},
    )
    capsule = json.loads(
        (_bundle(state.id, snapshot) / "continuation.json").read_text(encoding="utf-8")
    )
    assert validate_continuation(capsule) == capsule
    assert capsule["gate"] == GATE_DIRECTION_SELECTION
    assert capsule["next_node"] == "check_convergence"
    assert capsule["restored_status"] == "waiting_input"
    assert capsule["state_version"] == 1189
    # Graph-internal channels are not the capsule's to carry.
    assert capsule["workflow_state_keys"] == ["current_round", "planning_stage"]


@pytest.mark.asyncio
async def test_an_annotation_capsule_opens_on_the_annotation_dialog(capture):
    workflow, state = capture(round_number=1)
    snapshot = await capture_decision_snapshot(
        workflow, state, gate=GATE_ANNOTATION_CONFIRMATION,
    )
    assert snapshot == "r001-annotation-g57"
    capsule = json.loads(
        (_bundle(state.id, snapshot) / "continuation.json").read_text(encoding="utf-8")
    )
    assert capsule["restored_status"] == "waiting_annotation"
    assert capsule["next_node"] == "bioinformatics_analysis"
    manifest = read_snapshot_manifest(_bundle(state.id, snapshot) / MANIFEST_NAME)
    assert manifest["retention_rank"] == 1


@pytest.mark.asyncio
async def test_the_coverage_record_travels_with_the_bundle(capture):
    """A restore that comes up missing a name has to be able to say why."""
    workflow, state = capture()
    snapshot = await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)
    coverage = read_snapshot_manifest(
        _bundle(state.id, snapshot) / MANIFEST_NAME
    )["coverage"]
    assert coverage["saved_names"] == 3
    assert coverage["skipped_names"] == [
        {"name": "ss", "type": "SingleSampleGSEA", "reason": "serialization_failed"},
    ]
    assert coverage["carried_definitions"] == [{"name": "load_loom", "kind": "function"}]


@pytest.mark.asyncio
async def test_snapshots_are_off_without_a_configured_directory(capture, monkeypatch):
    workflow, state = capture()
    monkeypatch.setattr(_config, "SNAPSHOT_DIR", None)
    assert await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT) is None
    assert workflow.sandbox.saves == 0


# ------------------------------------------------------------------ freshness


@pytest.mark.asyncio
async def test_a_boundary_that_follows_a_save_writes_no_second_dump(capture):
    """The round-boundary save has just committed; re-dumping costs minutes.

    Three of the four decisions arrive this way, and a capture that always
    saved would double the pause at every one of them for no change at all.
    """
    workflow, state = capture()
    await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)
    assert workflow.sandbox.saves == 0


@pytest.mark.asyncio
async def test_a_boundary_reached_after_a_cell_commits_first(capture):
    """The annotation decision always, and a round boundary a chat cell raced.

    Without this the bundle would pair a workspace written at revision 60 with
    a dump taken at 57 -- half of one decision and half of the next.
    """
    workflow, state = capture(revision=57, current=60)
    snapshot = await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)
    assert workflow.sandbox.saves == 1
    assert snapshot == "r003-round-g60"
    assert read_snapshot_manifest(
        _bundle(state.id, snapshot) / MANIFEST_NAME
    )["runtime_revision"] == 60


# ------------------------------------------------------------------ preconditions


def _failed(session_id: str):
    return read_session_listing(session_id)["failed"]


@pytest.mark.asyncio
async def test_a_save_that_does_not_commit_leaves_a_recorded_gap(capture):
    workflow, state = capture(revision=57, current=60)
    workflow.sandbox.save_result = False

    assert await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT) is None
    listing = read_session_listing(state.id)
    assert listing["snapshots"] == []
    assert [row["reason"] for row in listing["failed"]] == ["save_failed"]


@pytest.mark.asyncio
async def test_a_generation_without_its_pair_is_not_captured(capture):
    """Half a restore point is not a restore point.

    A state without its graph position would resume a position that no longer
    exists, which is why the save commits the three of them together.
    """
    from backend.config import get_session_states
    from backend.utils.sandbox_parts.regular_save import GENERATION_PAIR_DIRNAME

    workflow, state = capture()
    pair = get_session_states(state.id) / GENERATION_PAIR_DIRNAME / "57"
    (pair / "checkpoints.sqlite").unlink()

    assert await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT) is None
    assert [row["reason"] for row in _failed(state.id)] == ["no_pair"]


@pytest.mark.asyncio
async def test_a_dump_whose_signature_does_not_verify_is_not_captured(capture):
    """Shipping it would put a generation nobody can trust into a new session."""
    from backend.config import get_session_states

    workflow, state = capture()
    pkl = get_session_states(state.id) / f"{state.id}_session.pkl"
    pkl.write_bytes(b"different bytes entirely")

    assert await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT) is None
    assert [row["reason"] for row in _failed(state.id)] == ["dump_signature_invalid"]


@pytest.mark.asyncio
async def test_a_recorded_gap_disappears_when_the_decision_is_captured(capture):
    """A decision that failed once and succeeded later is not a gap."""
    workflow, state = capture()
    workflow.sandbox.notebook_state.runtime_persistence["last_attempt_status"] = "failed"
    assert await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT) is None
    assert len(_failed(state.id)) == 1

    workflow.sandbox.notebook_state.runtime_persistence["last_attempt_status"] = "succeeded"
    assert await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)
    listing = read_session_listing(state.id)
    assert listing["failed"] == []
    assert len(listing["snapshots"]) == 1


@pytest.mark.asyncio
async def test_an_unexpected_error_is_recorded_rather_than_raised(capture, monkeypatch):
    """The writer's contract, at the writer's own boundary.

    Invariant 3: a snapshot failure leaves the analysis exactly as it would
    have been. The gate calls this without a guard of its own, deliberately --
    a try/except at the call site would make the contract someone else's
    problem, and only the writer knows which failures are recordable.
    """
    workflow, state = capture()

    def _explode(**kwargs):
        raise RuntimeError("the snapshot volume went away")

    monkeypatch.setattr(writer_module, "_stage_session_files", _explode)

    assert await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT) is None
    assert not workflow.sandbox._exec_lock.locked()
    assert [row["reason"] for row in _failed(state.id)] == ["error"]


# ------------------------------------------------------------------ the pin


@pytest.mark.asyncio
async def test_the_pin_holds_the_generation_the_next_save_replaces(capture, monkeypatch):
    """The whole reason the dump is not copied under the lock.

    The generation copy runs after the locks are released, so by the time it
    reads the dump a later save may have replaced the file and the pair pruner
    may have removed the pair. The hardlink keeps the inode; this replaces both
    at exactly that moment and checks the bundle still holds the first bytes.

    The replacement below is a rename onto a fresh inode, because that is what
    ``_commit_pkl_transaction`` does. The distinction is the whole basis of the
    pin: an in-place truncation would be followed through the hardlink and the
    capture would read the new generation. Production never writes the live
    dump in place, and ``test_the_commit_publishes_the_dump_by_rename`` is what
    keeps that true.
    """
    from backend.config import get_session_states
    from backend.utils.sandbox_parts.regular_save import GENERATION_PAIR_DIRNAME

    workflow, state = capture(dump=b"generation fifty-seven")
    original = _stage_generation_with_interference(monkeypatch, state.id)

    snapshot = await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)
    assert original["ran"] is True

    bundle = _bundle(state.id, snapshot)
    assert (bundle / RUNTIME_PKL_RELPATH).read_bytes() == b"generation fifty-seven"
    assert (get_session_states(state.id) / f"{state.id}_session.pkl").read_bytes() != (
        b"generation fifty-seven"
    )
    assert not (
        get_session_states(state.id) / GENERATION_PAIR_DIRNAME / "57"
    ).exists()


def _stage_generation_with_interference(monkeypatch, session_id: str) -> dict:
    """Replace the live generation just before the off-lock copy reads it."""
    import shutil

    from backend.config import get_session_states
    from backend.utils.sandbox_parts.regular_save import GENERATION_PAIR_DIRNAME

    marker = {"ran": False}
    original = writer_module._stage_generation

    def _interfere(**kwargs):
        marker["ran"] = True
        states = get_session_states(session_id)
        replacement = states / "next-generation.tmp"
        replacement.write_bytes(b"a later generation")
        os.replace(replacement, states / f"{session_id}_session.pkl")
        shutil.rmtree(states / GENERATION_PAIR_DIRNAME / "57", ignore_errors=True)
        return original(**kwargs)

    monkeypatch.setattr(writer_module, "_stage_generation", _interfere)
    return marker


@pytest.mark.asyncio
async def test_two_decisions_over_one_dump_share_a_single_inode(capture):
    """A decision reached without the kernel executing anything in between.

    Storing the dump twice would double the cost of the feature for two
    bundles that are byte-identical, and the filesystem reference counts the
    sharing so deleting either leaves the other readable.
    """
    import shutil

    workflow, state = capture(round_number=3)
    first = await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)
    state.current_round = 4
    second = await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)

    one = _bundle(state.id, first) / RUNTIME_PKL_RELPATH
    two = _bundle(state.id, second) / RUNTIME_PKL_RELPATH
    assert one.stat().st_ino == two.stat().st_ino

    shutil.rmtree(_bundle(state.id, first))
    assert two.read_bytes() == b"a dump"


def test_the_commit_publishes_the_dump_by_rename():
    """The property that makes a hardlink a valid pin.

    A capture holds the committed generation by hardlink and reads it minutes
    later, off the execution lock. That is safe only while every writer of
    those files publishes onto a fresh inode: an ``open(path, "wb")`` anywhere
    in the commit would truncate the inode the pin shares and the capture would
    silently read the *next* generation instead of the one it verified.

    Asserted on the source because the alternative is a real multi-gigabyte
    save, and because the failure it guards against is a one-line change in
    another module that no snapshot test would otherwise notice.
    """
    import inspect

    from backend.utils.sandbox_parts import regular_save

    source = inspect.getsource(regular_save._commit_pkl_transaction)
    assert "os.replace(staged_pkl, live_pkl)" in source
    assert "open(live_pkl" not in source


# ------------------------------------------------------------------ the watchdog


@pytest.mark.asyncio
async def test_a_capture_that_stops_making_progress_is_abandoned(capture, monkeypatch):
    """A hung mount must cost the analysis a bound, not the analysis.

    The worker is deliberately not cancelled -- a thread in an uninterruptible
    write cannot be -- so what is asserted here is the part that can be: the
    lock comes back, the decision is recorded, and nothing is published.
    """
    monkeypatch.setattr(_config, "SNAPSHOT_STALL_SECONDS", 1)
    monkeypatch.setattr(_config, "SNAPSHOT_MAX_SECONDS", 30)

    workflow, state = capture()

    def _hang(**kwargs):
        time.sleep(3)
        return {}

    monkeypatch.setattr(writer_module, "_stage_session_files", _hang)

    assert await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT) is None
    assert not workflow.sandbox._exec_lock.locked()
    assert [row["reason"] for row in _failed(state.id)] == ["stall"]
    assert read_session_listing(state.id)["snapshots"] == []


@pytest.mark.asyncio
async def test_an_abandoned_worker_publishes_nothing(capture, monkeypatch):
    """Publication belongs to the caller, and the caller has given up."""
    monkeypatch.setattr(_config, "SNAPSHOT_STALL_SECONDS", 1)
    monkeypatch.setattr(_config, "SNAPSHOT_MAX_SECONDS", 30)

    workflow, state = capture()
    finished = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _slow(**kwargs):
        time.sleep(2.5)
        loop.call_soon_threadsafe(finished.set)
        return {}

    monkeypatch.setattr(writer_module, "_stage_session_files", _slow)
    assert await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT) is None
    await asyncio.wait_for(finished.wait(), timeout=10)
    await asyncio.sleep(0.2)
    assert read_session_listing(state.id)["snapshots"] == []


# ------------------------------------------------------------------ workspace rules


@pytest.mark.asyncio
async def test_a_workspace_entry_that_is_not_a_file_is_refused_and_named(capture):
    """Refusing is right; refusing silently is the coverage gap in a new place."""
    from backend.config import get_session_workspace

    workflow, state = capture()
    workspace = get_session_workspace(state.id)
    os.symlink("/etc/passwd", workspace / "escape.txt")
    os.mkfifo(workspace / "pipe")

    snapshot = await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)
    manifest = read_snapshot_manifest(_bundle(state.id, snapshot) / MANIFEST_NAME)

    refused = {entry["path"]: entry["reason"] for entry in manifest["skipped_entries"]}
    assert refused == {
        "workspace/escape.txt": "symlink",
        "workspace/pipe": "not_a_regular_file",
    }
    # The ordinary files still travelled.
    assert "workspace/table.csv" in manifest["files"]
    assert not (_bundle(state.id, snapshot) / "workspace" / "escape.txt").exists()


@pytest.mark.asyncio
async def test_a_symlinked_directory_is_not_walked_into(capture):
    """Refusing symlinks at every level is what makes containment hold."""
    from backend.config import get_session_dir, get_session_workspace

    workflow, state = capture()
    outside = get_session_dir(state.id) / "not-the-workspace"
    outside.mkdir()
    (outside / "secret.txt").write_text("should not travel")
    os.symlink(outside, get_session_workspace(state.id) / "linked")

    snapshot = await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)
    manifest = read_snapshot_manifest(_bundle(state.id, snapshot) / MANIFEST_NAME)
    assert not any("secret.txt" in name for name in manifest["files"])
    assert {"path": "workspace/linked", "reason": "symlink"} in manifest["skipped_entries"]


# ------------------------------------------------------------------ reclamation


def test_an_abandoned_pin_is_reclaimed_and_a_live_one_is_not(monkeypatch, tmp_path):
    """A pin is a hardlink to a whole dump, so leaking one leaks gigabytes."""
    from backend.config import ensure_session_dirs, get_session_dir

    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path / "snapshots")
    session_id = "snap-reclaim"
    ensure_session_dirs(session_id)
    pins = get_session_dir(session_id) / PIN_DIRNAME
    (pins / "abandoned").mkdir(parents=True)
    (pins / "abandoned" / "held.pkl").write_bytes(b"x")
    (pins / "inflight").mkdir(parents=True)

    monkeypatch.setattr(writer_module, "_LIVE_ATTEMPTS", {"inflight"})
    reclaim_abandoned_capture_work()

    assert not (pins / "abandoned").exists()
    assert (pins / "inflight").is_dir()


@pytest.mark.asyncio
async def test_a_published_capture_leaves_no_pin_and_no_staging(capture):
    from backend.config import get_session_dir

    workflow, state = capture()
    await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)

    pins = get_session_dir(state.id) / PIN_DIRNAME
    assert not pins.exists() or not any(pins.iterdir())
    staging = layout.staging_root(state.id)
    assert not staging.exists() or not any(staging.iterdir())


# ------------------------------------------------------------------ deletion


@pytest.mark.asyncio
async def test_deleting_a_session_deletes_its_snapshots(capture):
    """They live outside the session tree, so nothing else reaches them."""
    from backend.services.session_deletion import delete_session_generation

    workflow, state = capture()
    await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)
    root = layout.session_root(state.id)
    assert root.is_dir()

    await delete_session_generation(state.id, state, state.user_id, force_stop=True)
    assert not root.exists()


@pytest.mark.asyncio
async def test_with_snapshots_disabled_the_stored_tree_is_left_alone(capture, monkeypatch):
    """The operator can still reach it; deleting what we cannot name is worse."""
    from backend.services.session_deletion import delete_session_generation

    workflow, state = capture()
    await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT)
    root = layout.session_root(state.id)

    monkeypatch.setattr(_config, "SNAPSHOT_DIR", None)
    await delete_session_generation(state.id, state, state.user_id, force_stop=True)
    assert root.is_dir()


# ------------------------------------------------------------------ listing


@pytest.mark.asyncio
async def test_a_replayed_decision_supersedes_the_earlier_capture(capture):
    """A supplementary analysis re-presents the same dialog.

    Both stay restorable: the earlier one holds the namespace before the
    supplementary work, which is often the point of going back.
    """
    workflow, state = capture(round_number=1)
    first = await capture_decision_snapshot(
        workflow, state, gate=GATE_ANNOTATION_CONFIRMATION,
    )
    workflow.sandbox.notebook_state.runtime_persistence["current_revision"] = 61
    second = await capture_decision_snapshot(
        workflow, state, gate=GATE_ANNOTATION_CONFIRMATION,
    )
    assert first == "r001-annotation-g57" and second == "r001-annotation-g61"

    rows = {row["snapshot_id"]: row for row in read_session_listing(state.id)["snapshots"]}
    assert rows[first]["superseded"] is True
    assert rows[second]["superseded"] is False
    # Later goes first within a rank, so the superseded capture -- the one taken
    # before the supplementary work, and often the one worth going back to --
    # is the one kept longer.
    assert rows[second]["deletion_order"] < rows[first]["deletion_order"]
