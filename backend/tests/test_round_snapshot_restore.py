"""Restoring one decision snapshot into a new session. Part B spec 4.6.

Restoring forks. The properties that matter are that the fork is complete --
it opens, its runtime verifies, and it depends on nothing that can be taken
away -- and that the source is untouched. Everything else is a refusal, and
each refusal has to leave nothing behind: a caller sees a session or it sees
none, never a directory that half exists.
"""
import json
import shutil

import pytest

from backend.constants import (
    GATE_ROUND_WAIT,
)

from backend.config import _config
from backend.services.round_snapshots import layout
from backend.services.round_snapshots.layout import (
    MANIFEST_NAME,
)
from backend.services.round_snapshots.restore import (
    SnapshotRestoreError,
    restore_snapshot_into_new_session,
)
from backend.services.round_snapshots.writer import capture_decision_snapshot

from backend.tests.test_round_snapshot_capture import (  # noqa: E402
    _Sandbox,
    _Workflow,
    _seed_session,
    _state,
)


OWNER = "owner-1"


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """A source session with one published snapshot, ready to restore."""
    import uuid

    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path / "snapshots")

    async def _build(
        *, gate=GATE_ROUND_WAIT, round_number=3, revision=57, prepare_state=None,
    ):
        session_id = "src-" + uuid.uuid4().hex[:12]
        digest = _seed_session(session_id, revision=revision, round_number=round_number)
        workflow = _Workflow(
            _Sandbox(session_id, revision, digest, round_number=round_number)
        )
        state = _state(session_id, round_number=round_number)
        if prepare_state is not None:
            prepare_state(state)
        snapshot = await capture_decision_snapshot(
            workflow, state, gate=gate, workflow_state={"current_round": round_number},
        )
        assert snapshot is not None
        return session_id, snapshot

    return _build






@pytest.mark.asyncio
async def test_the_fork_survives_losing_the_source_and_the_snapshot(captured):
    """Self-sufficiency, which is what makes a snapshot a restore point.

    A new session that still read from the source directory or from snapshot
    storage would be a view of them, not a fork -- and the operator who deletes
    either would find out only when the analysis failed.
    """
    from backend.config import get_session_dir, get_session_states
    from backend.services.session_store import load_session_from_disk
    from backend.utils.sandbox import _integrity_dir, _verify_pkl_integrity

    source, snapshot = await captured()
    new_id = await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )

    shutil.rmtree(get_session_dir(source))
    shutil.rmtree(layout.snapshot_storage_root())

    restored = load_session_from_disk(new_id)
    assert restored is not None and restored.id == new_id
    states = get_session_states(new_id)
    assert _verify_pkl_integrity(states / f"{new_id}_session.pkl", _integrity_dir(new_id))
    assert (get_session_dir(new_id) / "workspace" / "table.csv").is_file()
    assert (get_session_dir(new_id) / "uploads" / "matrix.h5ad").is_file()




def _sessions_on_disk() -> set[str]:
    from backend.services.session_store import list_saved_sessions

    return {summary["id"] for summary in list_saved_sessions()}


async def _expect_refusal(source, snapshot, reason):
    before = _sessions_on_disk()
    with pytest.raises(SnapshotRestoreError) as raised:
        await restore_snapshot_into_new_session(
            source_session_id=source, snapshot=snapshot, user_id=OWNER,
        )
    assert raised.value.reason == reason
    # Nothing half-built survives a refusal.
    assert _sessions_on_disk() == before


@pytest.mark.asyncio
async def test_a_tampered_manifest_is_refused(captured):
    source, snapshot = await captured()
    bundle = layout.snapshot_dir(source, snapshot)
    payload = json.loads((bundle / MANIFEST_NAME).read_text(encoding="utf-8"))
    payload["round"] = 9
    (bundle / MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")
    await _expect_refusal(source, snapshot, "snapshot_unverified")


@pytest.mark.asyncio
async def test_a_file_that_does_not_match_its_digest_is_refused(captured):
    """Verified while copying, so a corrupt file never becomes briefly usable."""
    source, snapshot = await captured()
    bundle = layout.snapshot_dir(source, snapshot)
    (bundle / "workspace" / "table.csv").write_text("gene,logfc\nEDITED,9.9\n")
    await _expect_refusal(source, snapshot, "bundle_corrupt")


@pytest.mark.asyncio
async def test_an_input_that_changed_under_the_snapshot_is_refused(captured):
    """The inputs are what a restored analysis re-reads.

    A session whose data silently changed under it is worse than one that will
    not open, so this refuses rather than warning.
    """
    source, snapshot = await captured()
    (layout.uploads_root(source) / "matrix.h5ad").write_bytes(b"a different matrix")
    await _expect_refusal(source, snapshot, "bundle_corrupt")


@pytest.mark.asyncio
async def test_a_bundle_claiming_another_session_is_refused(captured, monkeypatch):
    """The manifest names its source; a bundle moved between sessions is not one."""
    source, snapshot = await captured()
    other, _ = await captured()
    bundle = layout.snapshot_dir(source, snapshot)
    shutil.copytree(bundle, layout.snapshot_dir(other, snapshot), dirs_exist_ok=True)
    await _expect_refusal(other, snapshot, "snapshot_unverified")



# Both of these were found by driving the real HTTP surface on a live
# deployment rather than by reading it. The restore job answers everything a
# caller cannot know until the bundle is opened; these are the two things the
# server knew before it started, and the one thing it should not have said.


def _request(user_id=OWNER):
    import types

    return types.SimpleNamespace(state=types.SimpleNamespace(user_id=user_id))


@pytest.mark.asyncio
async def test_a_refusal_does_not_tell_the_caller_where_the_bundle_lives(captured):
    """The reason travels; the server's storage layout does not.

    The underlying manifest error names an absolute path, and that detail was
    reaching the client verbatim. A caller can do nothing with it, and it
    describes a filesystem they cannot see.
    """
    source, snapshot = await captured()
    bundle = layout.snapshot_dir(source, snapshot)
    (bundle / MANIFEST_NAME).unlink()

    with pytest.raises(SnapshotRestoreError) as raised:
        await restore_snapshot_into_new_session(
            source_session_id=source, snapshot=snapshot, user_id=OWNER,
        )

    assert raised.value.reason == "snapshot_unverified"
    detail = str(raised.value)
    assert str(bundle) not in detail
    assert MANIFEST_NAME not in detail
    assert "/" not in detail
