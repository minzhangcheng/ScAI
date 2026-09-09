"""Restoring one decision snapshot into a new session. Part B spec 4.6.

Restoring forks. The properties that matter are that the fork is complete --
it opens, its runtime verifies, and it depends on nothing that can be taken
away -- and that the source is untouched. Everything else is a refusal, and
each refusal has to leave nothing behind: a caller sees a session or it sees
none, never a directory that half exists.
"""
import hashlib
import json
import os
import shutil

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
        RUNTIME_PKL_RELPATH,
    )
    from backend.services.round_snapshots.restore import (
        SnapshotRestoreError,
        restore_snapshot_into_new_session,
        restored_session_name,
    )
    from backend.services.round_snapshots.writer import capture_decision_snapshot
except ImportError as exc:  # the module under test does not exist yet
    _IMPORT_ERROR = exc
    MANIFEST_NAME = "manifest.json"
    RUNTIME_PKL_RELPATH = "runtime/session.pkl"
else:
    _IMPORT_ERROR = None

from backend.tests.test_round_snapshot_capture import (  # noqa: E402
    _Sandbox,
    _Workflow,
    _digest_of,
    _seed_session,
    _state,
)


@pytest.fixture(autouse=True)
def _require_module_under_test():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"module under test is missing: {_IMPORT_ERROR}")


OWNER = "owner-1"


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """A source session with one published snapshot, ready to restore."""
    import uuid

    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path / "snapshots")

    async def _build(*, gate=GATE_ROUND_WAIT, round_number=3, revision=57):
        session_id = "src-" + uuid.uuid4().hex[:12]
        digest = _seed_session(session_id, revision=revision, round_number=round_number)
        workflow = _Workflow(
            _Sandbox(session_id, revision, digest, round_number=round_number)
        )
        state = _state(session_id, round_number=round_number)
        snapshot = await capture_decision_snapshot(
            workflow, state, gate=gate, workflow_state={"current_round": round_number},
        )
        assert snapshot is not None
        return session_id, snapshot

    return _build


def _tree(root):
    """Every file under *root* with its digest, for a before/after comparison."""
    if not root.is_dir():
        return {}
    return {
        str(path.relative_to(root)): _digest_of(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ------------------------------------------------------------------ the fork


@pytest.mark.asyncio
async def test_a_restore_produces_a_session_parked_on_the_decision(captured):
    from backend.services.session_store import load_session_from_disk

    source, snapshot = await captured()
    new_id = await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )
    assert new_id != source

    restored = load_session_from_disk(new_id)
    assert restored is not None
    assert restored.id == new_id
    assert restored.user_id == OWNER
    assert restored.current_round == 3
    assert restored.status.value == "waiting_input"
    assert restored.restore_resume_kind == "semantic_gate"
    assert restored.restore_resume_gate == GATE_ROUND_WAIT
    assert restored.last_restored_snapshot_id == f"{source}/{snapshot}"
    assert restored.restore_branch_id


@pytest.mark.asyncio
async def test_an_annotation_restore_opens_on_the_annotation_dialog(captured):
    """The gate decides the status, so the UI shows the dialog it belongs to."""
    from backend.services.session_store import load_session_from_disk

    source, snapshot = await captured(gate=GATE_ANNOTATION_CONFIRMATION, round_number=1)
    new_id = await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )
    restored = load_session_from_disk(new_id)
    assert restored.status.value == "waiting_annotation"
    assert restored.restore_resume_gate == GATE_ANNOTATION_CONFIRMATION


@pytest.mark.asyncio
async def test_the_restored_session_is_named_for_what_it_rolls_back_to(captured):
    from backend.services.session_store import load_session_from_disk

    source, snapshot = await captured()
    new_id = await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )
    assert load_session_from_disk(new_id).name == "rollback Liver atlas round 3"


def test_a_round_with_two_decisions_names_both_restores_apart():
    """Round 1 holds the annotation confirmation and the direction selection.

    Two sessions called the same thing would be worse than a longer name.
    """
    assert restored_session_name(
        "Liver atlas", 1, GATE_ANNOTATION_CONFIRMATION, ambiguous=True,
    ) == "rollback Liver atlas round 1 annotation"
    assert restored_session_name(
        "Liver atlas", 1, GATE_DIRECTION_SELECTION, ambiguous=True,
    ) == "rollback Liver atlas round 1 direction"
    assert restored_session_name(
        "Liver atlas", 3, GATE_ROUND_WAIT, ambiguous=False,
    ) == "rollback Liver atlas round 3"


@pytest.mark.asyncio
async def test_the_restore_carries_the_runtime_the_snapshot_holds(captured):
    from backend.config import get_session_states
    from backend.utils.runtime_manifest import read_runtime_manifest, runtime_manifest_path
    from backend.utils.sandbox import _integrity_dir, _verify_pkl_integrity

    source, snapshot = await captured()
    new_id = await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )

    states = get_session_states(new_id)
    pkl = states / f"{new_id}_session.pkl"
    bundle = layout.snapshot_dir(source, snapshot)
    assert pkl.read_bytes() == (bundle / RUNTIME_PKL_RELPATH).read_bytes()
    # Signed under this installation's key, against this file's name.
    assert _verify_pkl_integrity(pkl, _integrity_dir(new_id))

    manifest = read_runtime_manifest(runtime_manifest_path(new_id, states_dir=states))
    assert manifest["pkl_sha256"] == _digest_of(pkl)
    # The graph position is deliberately not carried, so the pairing must not
    # claim one -- the same thing an import does for a reallocated id.
    assert manifest["pairing"]["checkpoint"] is None


@pytest.mark.asyncio
async def test_the_graph_checkpoint_is_read_and_not_installed(captured):
    """It is keyed by the source's thread id and cannot be renamed into place.

    A new session that kept it would resume a thread that does not exist here,
    which is the failure spec 3.6 records for an archive import.
    """
    from backend.config import get_session_states

    source, snapshot = await captured()
    new_id = await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )
    assert not (get_session_states(new_id) / "checkpoints.sqlite").exists()


@pytest.mark.asyncio
async def test_the_restore_brings_the_notebook_workspace_and_inputs(captured):
    from backend.config import (
        get_session_dir,
        get_session_states,
        get_session_uploads,
    )

    source, snapshot = await captured()
    new_id = await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )

    # Under the fork's own name -- see the test that follows this section.
    assert (get_session_states(new_id) / f"{new_id}.ipynb").is_file()
    assert (get_session_dir(new_id) / "workspace" / "table.csv").is_file()
    assert (get_session_dir(new_id) / "workspace" / "figures" / "umap.png").is_file()
    assert (get_session_uploads(new_id) / "matrix.h5ad").read_bytes() == b"an input matrix"
    assert any((get_session_dir(new_id) / "blobs").iterdir())
    # Reports and logs are not carried: this session has not produced them.
    assert not any((get_session_dir(new_id) / "reports").iterdir())
    assert not any((get_session_dir(new_id) / "logs").iterdir())


@pytest.mark.asyncio
async def test_no_report_or_log_path_survives_into_the_restored_state(captured):
    """A dangling path renders as a broken link rather than as an absence."""
    from backend.config import get_session_dir

    source, snapshot = await captured()
    new_id = await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )
    text = (get_session_dir(new_id) / "session.json").read_text(encoding="utf-8")
    assert f"/{new_id}/reports/" not in text
    assert f"/{new_id}/logs/" not in text
    # And no *path* still points into the session it was forked from. The
    # source's id does survive, twice and deliberately: last_restored_snapshot_id
    # and restore_branch_id are lineage, which is what lets the source list its
    # forks and the fork name its origin.
    assert str(get_session_dir(source)) not in text
    assert f'"{source}"' not in text


@pytest.mark.asyncio
async def test_the_fork_reads_its_history_from_a_notebook_named_after_itself(captured):
    """Found by exporting a fork: its notebook was named after the source.

    A session reads its history from ``states/<its own id>.ipynb``. The bundle
    stores that file under the name it had, so laying it down unrenamed gave
    the fork an empty notebook, left the real history in a file naming a
    session that does not exist on this installation, and carried that name
    into every export the fork later produced.
    """
    from backend.config import get_session_states

    source, snapshot = await captured()
    new_id = await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )

    states = get_session_states(new_id)
    carried = states / f"{new_id}.ipynb"
    assert carried.is_file(), "the fork has no notebook of its own"
    assert carried.read_bytes() == (
        get_session_states(source) / f"{source}.ipynb"
    ).read_bytes()
    assert not (states / f"{source}.ipynb").exists(), (
        "a file named after the source session was left in the fork"
    )


# ------------------------------------------------------------------ the source


@pytest.mark.asyncio
async def test_the_source_session_and_its_snapshots_are_untouched(captured):
    """The promise the whole design is built on: restoring never rewinds."""
    from backend.config import get_session_dir

    source, snapshot = await captured()
    before_session = _tree(get_session_dir(source))
    before_snapshots = _tree(layout.session_root(source))

    await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )

    assert _tree(get_session_dir(source)) == before_session
    assert _tree(layout.session_root(source)) == before_snapshots


@pytest.mark.asyncio
async def test_one_snapshot_can_be_restored_twice(captured):
    source, snapshot = await captured()
    first = await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )
    second = await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )
    assert first != second


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


# ------------------------------------------------------------------ refusals


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
async def test_a_missing_paired_checkpoint_is_refused(captured):
    from backend.services.round_snapshots.layout import PAIR_CHECKPOINT_RELPATH

    source, snapshot = await captured()
    (layout.snapshot_dir(source, snapshot) / PAIR_CHECKPOINT_RELPATH).unlink()
    await _expect_refusal(source, snapshot, "bundle_incomplete")


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
async def test_an_unknown_snapshot_is_refused(captured):
    source, _snapshot = await captured()
    await _expect_refusal(source, "r009-round-g999", "snapshot_unverified")


@pytest.mark.asyncio
async def test_an_identifier_that_is_not_one_is_refused(captured):
    source, _snapshot = await captured()
    await _expect_refusal(source, "../../etc", "unknown_snapshot")


@pytest.mark.asyncio
async def test_a_bundle_claiming_another_session_is_refused(captured, monkeypatch):
    """The manifest names its source; a bundle moved between sessions is not one."""
    source, snapshot = await captured()
    other, _ = await captured()
    bundle = layout.snapshot_dir(source, snapshot)
    shutil.copytree(bundle, layout.snapshot_dir(other, snapshot), dirs_exist_ok=True)
    await _expect_refusal(other, snapshot, "snapshot_unverified")


# ------------------------------------------------------ what the caller is told

# Both of these were found by driving the real HTTP surface on a live
# deployment rather than by reading it. The restore job answers everything a
# caller cannot know until the bundle is opened; these are the two things the
# server knew before it started, and the one thing it should not have said.


def _request(user_id=OWNER):
    import types

    return types.SimpleNamespace(state=types.SimpleNamespace(user_id=user_id))


async def _restore_over_http(source, snapshot):
    from backend.routes.round_snapshots import restore_session_snapshot

    return await restore_session_snapshot(source, snapshot, _request())


def _jobs():
    from backend.routes.round_snapshots import _import_jobs

    return dict(_import_jobs)


@pytest.mark.asyncio
async def test_an_id_that_names_no_snapshot_is_refused_before_a_job_starts(captured):
    """404, not an accepted job that fails later.

    Answering 200 and reporting the failure through the import-job record made
    a caller poll to be told something the server could have said immediately,
    and it spent a job slot on a request that was never going to do any work.
    """
    from fastapi import HTTPException

    source, _snapshot = await captured()
    before = _jobs()

    with pytest.raises(HTTPException) as raised:
        await _restore_over_http(source, "r009-round-g999")

    assert raised.value.status_code == 404
    assert _jobs() == before


@pytest.mark.asyncio
async def test_an_identifier_that_is_not_one_is_refused_as_a_bad_request(captured):
    """400 rather than 404: the request is malformed, not merely unsatisfiable."""
    from fastapi import HTTPException

    source, _snapshot = await captured()
    before = _jobs()

    with pytest.raises(HTTPException) as raised:
        await _restore_over_http(source, "../../etc")

    assert raised.value.status_code == 400
    # Compared against what was already there rather than against empty: the
    # registry is module state that the whole suite shares, so a full run
    # arrives here with other tests' jobs in it.
    assert _jobs() == before


@pytest.mark.asyncio
async def test_a_restore_that_can_run_is_still_accepted(captured):
    """The guard above must not answer for the bundle the job is meant to open."""
    source, snapshot = await captured()

    before = _jobs()

    answer = await _restore_over_http(source, snapshot)

    assert answer["job_id"]
    assert answer["job_id"] in _jobs()
    assert answer["job_id"] not in before


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


# ------------------------------------------------------------------ lineage


@pytest.mark.asyncio
async def test_the_source_lists_the_sessions_it_was_forked_into(captured):
    from backend.routes.round_snapshots import _restored_from

    source, snapshot = await captured()
    new_id = await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )
    assert _restored_from(source) == {snapshot: [new_id]}


@pytest.mark.asyncio
async def test_the_session_summary_carries_where_a_fork_came_from(captured):
    from backend.services.session_store import list_saved_sessions

    source, snapshot = await captured()
    new_id = await restore_snapshot_into_new_session(
        source_session_id=source, snapshot=snapshot, user_id=OWNER,
    )
    summaries = {row["id"]: row for row in list_saved_sessions()}
    assert summaries[new_id]["last_restored_snapshot_id"] == f"{source}/{snapshot}"
