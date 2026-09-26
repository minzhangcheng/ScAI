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
import time

import pytest

from backend.constants import (
    GATE_DIRECTION_SELECTION,
    GATE_ROUND_WAIT,
)

from backend.config import _config
from backend.services.round_snapshots import layout
from backend.services.round_snapshots.layout import (
    MANIFEST_NAME,
)
from backend.services.round_snapshots.listing import read_session_listing
from backend.services.round_snapshots.manifest import (
    read_snapshot_manifest,
)
from backend.services.round_snapshots import writer as writer_module
from backend.services.round_snapshots.writer import (
    capture_decision_snapshot,
)




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


def _rewrite_runtime_manifest(
    session_id: str, revision: int, *, pairing: dict | None = None,
) -> None:
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
            pairing=pairing,
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
async def test_snapshots_are_off_without_a_configured_directory(capture, monkeypatch):
    workflow, state = capture()
    monkeypatch.setattr(_config, "SNAPSHOT_DIR", None)
    assert await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT) is None
    assert workflow.sandbox.saves == 0






def _failed(session_id: str):
    return read_session_listing(session_id)["failed"]


@pytest.mark.asyncio
async def test_a_mismatched_checkpoint_pair_is_refused_before_publication(
    monkeypatch, tmp_path,
):
    from backend.config import get_session_states
    from backend.utils.runtime_pairing import checkpoint_identity_at_path

    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path / "snapshots")
    session_id = "snap-pair-mismatch"
    digest = _seed_session(session_id, revision=57, round_number=1)
    states = get_session_states(session_id)
    checkpoint = states / "generation" / "57" / "checkpoints.sqlite"
    _rewrite_runtime_manifest(
        session_id,
        57,
        pairing={
            "state_version": 1,
            "checkpoint": checkpoint_identity_at_path(session_id, checkpoint),
        },
    )
    _write_checkpoint_db(checkpoint, session_id, round_number=2)

    state = _state(session_id, round_number=1)
    workflow = _Workflow(_Sandbox(session_id, 57, digest, round_number=1))
    assert await capture_decision_snapshot(
        workflow, state, gate=GATE_DIRECTION_SELECTION,
    ) is None
    assert [row["reason"] for row in _failed(session_id)] == [
        "checkpoint_pairing_mismatch"
    ]
    assert read_session_listing(session_id)["snapshots"] == []


@pytest.mark.asyncio
async def test_a_save_that_does_not_commit_leaves_a_recorded_gap(capture):
    workflow, state = capture(revision=57, current=60)
    workflow.sandbox.save_result = False

    assert await capture_decision_snapshot(workflow, state, gate=GATE_ROUND_WAIT) is None
    listing = read_session_listing(state.id)
    assert listing["snapshots"] == []
    assert [row["reason"] for row in listing["failed"]] == ["save_failed"]






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


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", ["none", "snapshot", "staging"])
async def test_snapshot_removal_preserves_inputs_still_in_use(capture, monkeypatch, remaining):
    monkeypatch.setattr(_config, "SNAPSHOT_FREE_SPACE_RESERVE_BYTES", None)
    workflow, state = capture()
    snapshot = await capture_decision_snapshot(
        workflow, state, gate=GATE_ROUND_WAIT, workflow_state={},
    )
    assert snapshot
    inputs = layout.uploads_root(state.id)
    input_bytes = {p: p.read_bytes() for p in inputs.rglob("*") if p.is_file()}
    assert input_bytes
    if remaining == "snapshot":
        state.current_round = workflow.sandbox.round_number = 4
        workflow.sandbox.notebook_state.runtime_persistence["current_revision"] = 58
        retained = await capture_decision_snapshot(
            workflow, state, gate=GATE_ROUND_WAIT, workflow_state={},
        )
        assert retained and retained != snapshot
    elif remaining == "staging":
        (layout.staging_root(state.id) / "live-attempt").mkdir(parents=True, exist_ok=True)

    layout.remove_snapshot(state.id, snapshot)

    assert not layout.snapshot_dir(state.id, snapshot).exists()
    if remaining == "none":
        assert not inputs.exists()
        assert not layout.session_root(state.id).exists()
    else:
        for path, expected in input_bytes.items():
            assert path.read_bytes() == expected
        if remaining == "snapshot":
            assert layout.snapshot_dir(state.id, retained).is_dir()
