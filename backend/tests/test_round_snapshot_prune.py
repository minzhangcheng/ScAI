"""Reclaiming the least valuable snapshots. Part B spec 4.7, B4.

Off by default, and that is the decision under test as much as the order is: a
snapshot is the only copy of the decision it precedes, so an installation that
was never told to reclaim must never reclaim.

When it is armed, what matters is that it gives up exactly what the operator
was told it would, in the order the listing already showed them, and that two
things are never in the set: the snapshot the capture just wrote, and the
earliest decision of a session somebody is still working on.

Note the shape of every test below: the snapshots are built with the reserve
*unset*, and the pressure is arranged afterwards. A capture prunes on its way
out, so arming the reserve first would have each capture reclaim the ones
before it -- which is correct behaviour and useless as a fixture.
"""
import pytest

from backend.constants import (
    GATE_ANNOTATION_CONFIRMATION,
    GATE_DIRECTION_SELECTION,
    GATE_ROUND_WAIT,
)

try:
    from backend.config import _config
    from backend.services.round_snapshots import layout, prune as prune_module
    from backend.services.round_snapshots.listing import read_session_listing
    from backend.services.round_snapshots.prune import prune_for_free_space
    from backend.services.round_snapshots.writer import capture_decision_snapshot
except ImportError as exc:  # the module under test does not exist yet
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from backend.tests.test_round_snapshot_capture import (  # noqa: E402
    _Sandbox,
    _Workflow,
    _seed_session,
    _state,
)


@pytest.fixture(autouse=True)
def _require_module_under_test():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"module under test is missing: {_IMPORT_ERROR}")


@pytest.fixture
def snapshots(monkeypatch, tmp_path):
    """Capture a series of decisions with the reserve unset, so nothing prunes."""
    import uuid

    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(_config, "SNAPSHOT_FREE_SPACE_RESERVE_BYTES", None)

    async def _build(decisions, *, dump=b"a dump"):
        """decisions: [(gate, round, revision)] -> (session_id, [snapshot ids])"""
        session_id = "prune-" + uuid.uuid4().hex[:12]
        digest = _seed_session(session_id, revision=50, dump=dump)
        sandbox = _Sandbox(session_id, 50, digest)
        state = _state(session_id)
        ids = []
        for gate, round_number, revision in decisions:
            sandbox.round_number = round_number
            state.current_round = round_number
            sandbox.notebook_state.runtime_persistence["current_revision"] = revision
            ids.append(await capture_decision_snapshot(
                _Workflow(sandbox), state, gate=gate, workflow_state={},
            ))
        assert all(ids), ids
        return session_id, ids

    return _build


def _arm(monkeypatch, *, reserve=1024, free=0):
    monkeypatch.setattr(_config, "SNAPSHOT_FREE_SPACE_RESERVE_BYTES", reserve)
    monkeypatch.setattr(prune_module, "free_space_bytes", lambda _path: free)


def _arm_until_one_goes(monkeypatch, *, reserve=1024):
    """Pressure that lifts as soon as anything is given up."""
    monkeypatch.setattr(_config, "SNAPSHOT_FREE_SPACE_RESERVE_BYTES", reserve)
    state = {"freed": False}

    def _free(_path):
        return 8192 if state["freed"] else 0

    monkeypatch.setattr(prune_module, "free_space_bytes", _free)
    original = prune_module._remove

    def _remove(session_id, snapshot):
        state["freed"] = True
        return original(session_id, snapshot)

    monkeypatch.setattr(prune_module, "_remove", _remove)


def _remaining(session_id):
    return [row["snapshot_id"] for row in read_session_listing(session_id)["snapshots"]]


# ------------------------------------------------------------------ the default


@pytest.mark.asyncio
async def test_without_a_reserve_nothing_is_ever_reclaimed(snapshots, monkeypatch):
    """The project owner's decision, and the one this module exists to honour."""
    session_id, ids = await snapshots([
        (GATE_ANNOTATION_CONFIRMATION, 1, 51),
        (GATE_ROUND_WAIT, 3, 52),
    ])
    monkeypatch.setattr(prune_module, "free_space_bytes", lambda _path: 0)

    assert prune_for_free_space() == []
    assert sorted(_remaining(session_id)) == sorted(ids)


@pytest.mark.asyncio
async def test_a_volume_above_its_reserve_gives_up_nothing(snapshots, monkeypatch):
    session_id, ids = await snapshots([(GATE_ROUND_WAIT, 3, 51)])
    _arm(monkeypatch, reserve=1024, free=8192)

    assert prune_for_free_space() == []
    assert _remaining(session_id) == ids


@pytest.mark.asyncio
async def test_an_unreadable_volume_is_not_treated_as_pressure(snapshots, monkeypatch):
    """Reporting zero for a volume nobody could stat would delete everything."""
    session_id, ids = await snapshots([(GATE_ROUND_WAIT, 3, 51)])
    _arm(monkeypatch, reserve=1024, free=-1)

    assert prune_for_free_space() == []
    assert _remaining(session_id) == ids


# ------------------------------------------------------------------ the order


@pytest.mark.asyncio
async def test_the_least_valuable_decision_goes_first(snapshots, monkeypatch):
    """The order the operator was shown, walked by the thing that acts on it.

    Round acceptance before the direction choice, and the round 1 annotation
    kept longest -- the same ranking the listing reports as deletion_order.
    """
    session_id, ids = await snapshots([
        (GATE_ANNOTATION_CONFIRMATION, 1, 51),
        (GATE_DIRECTION_SELECTION, 1, 52),
        (GATE_ROUND_WAIT, 3, 53),
    ])
    annotation, direction, acceptance = ids
    _arm_until_one_goes(monkeypatch)

    assert prune_for_free_space() == [acceptance]
    assert sorted(_remaining(session_id)) == sorted([annotation, direction])


@pytest.mark.asyncio
async def test_pressure_that_does_not_lift_walks_the_whole_order(snapshots, monkeypatch):
    session_id, ids = await snapshots([
        (GATE_ANNOTATION_CONFIRMATION, 1, 51),
        (GATE_DIRECTION_SELECTION, 1, 52),
        (GATE_ROUND_WAIT, 3, 53),
    ])
    annotation, direction, acceptance = ids
    _arm(monkeypatch)
    monkeypatch.setattr(prune_module, "_is_running", lambda _sid: False)

    assert prune_for_free_space() == [acceptance, direction, annotation]
    assert _remaining(session_id) == []


@pytest.mark.asyncio
async def test_the_capture_that_asked_is_never_the_one_reclaimed(snapshots, monkeypatch):
    """Otherwise a capture under pressure could delete what it just wrote."""
    session_id, ids = await snapshots([
        (GATE_ROUND_WAIT, 3, 51),
        (GATE_ROUND_WAIT, 4, 52),
    ])
    newest = ids[-1]
    _arm(monkeypatch)
    monkeypatch.setattr(prune_module, "_is_running", lambda _sid: False)

    removed = prune_for_free_space(keep_session=session_id, keep=newest)
    assert newest not in removed
    assert _remaining(session_id) == [newest]


@pytest.mark.asyncio
async def test_a_capture_reclaims_on_its_way_out(snapshots, monkeypatch):
    """The prune has exactly one caller, and this is it.

    Running it from the writer rather than from a sweeper is what makes an
    installation with the feature off prune nothing at all, and what keeps a
    failed capture from costing an existing restore point -- it runs after
    publication or not at all.
    """
    session_id, ids = await snapshots([(GATE_ROUND_WAIT, 3, 51)])
    _arm_until_one_goes(monkeypatch)

    digest = layout.snapshot_dir(session_id, ids[0])
    assert digest.is_dir()

    # A second decision on the same session, now under pressure.
    from backend.tests.test_round_snapshot_capture import _Sandbox as Sandbox

    sandbox = Sandbox(session_id, 51, "0" * 64, round_number=4)
    sandbox.notebook_state.runtime_persistence.update({
        "committed_revision": 51, "current_revision": 52,
        "committed_digest": _seeded_digest(session_id),
    })
    state = _state(session_id, round_number=4)
    newest = await capture_decision_snapshot(
        _Workflow(sandbox), state, gate=GATE_ROUND_WAIT, workflow_state={},
    )
    assert newest is not None
    assert _remaining(session_id) == [newest]


def _seeded_digest(session_id: str) -> str:
    import hashlib

    from backend.config import get_session_states

    pkl = get_session_states(session_id) / f"{session_id}_session.pkl"
    return hashlib.sha256(pkl.read_bytes()).hexdigest()


@pytest.mark.asyncio
async def test_a_running_session_keeps_its_earliest_decision(snapshots, monkeypatch):
    """Rank 1 is the annotation, and a running session may be about to need it.

    Everything else of that session is still fair game -- the protection is
    narrow on purpose, because a broad one would make the reserve unable to
    free anything on an installation with one busy session.
    """
    session_id, ids = await snapshots([
        (GATE_ANNOTATION_CONFIRMATION, 1, 51),
        (GATE_DIRECTION_SELECTION, 1, 52),
    ])
    annotation, direction = ids
    _arm(monkeypatch)
    monkeypatch.setattr(prune_module, "_is_running", lambda sid: sid == session_id)

    assert prune_for_free_space() == [direction]
    assert _remaining(session_id) == [annotation]


@pytest.mark.asyncio
async def test_a_finished_session_gives_up_its_annotation_too(snapshots, monkeypatch):
    session_id, ids = await snapshots([(GATE_ANNOTATION_CONFIRMATION, 1, 51)])
    _arm(monkeypatch)
    monkeypatch.setattr(prune_module, "_is_running", lambda _sid: False)

    assert prune_for_free_space() == ids
    assert _remaining(session_id) == []


@pytest.mark.asyncio
async def test_pressure_is_a_property_of_the_volume_not_of_one_session(
    snapshots, monkeypatch,
):
    """A finished session's round acceptance goes before a live one's.

    Reclaiming only within the session that happened to be capturing would
    leave the volume full while another session held restore points nobody is
    going to answer.
    """
    first, first_ids = await snapshots([(GATE_ROUND_WAIT, 5, 51)])
    second, second_ids = await snapshots([(GATE_ANNOTATION_CONFIRMATION, 1, 51)])
    _arm_until_one_goes(monkeypatch)
    monkeypatch.setattr(prune_module, "_is_running", lambda _sid: False)

    removed = prune_for_free_space(keep_session=second, keep=second_ids[0])
    assert removed == first_ids
    assert _remaining(second) == second_ids


# ------------------------------------------------------------- giving one up

# Found by deleting a real snapshot: the bundle went and 331 MB of inputs
# stayed. They are shared by a record rather than by the filesystem -- stored
# once per session, named by each bundle's manifest -- so the reference count
# that protects a shared dump does not protect them, and nothing else was
# going to notice they had become unreferenced.


def _inputs(session_id):
    from backend.services.round_snapshots.layout import uploads_root

    return uploads_root(session_id)


@pytest.mark.asyncio
async def test_giving_up_the_last_snapshot_reclaims_the_inputs_it_shared(snapshots):
    from backend.services.round_snapshots.layout import remove_snapshot, session_root

    session_id, ids = await snapshots([(GATE_ROUND_WAIT, 3, 51)])
    assert any(_inputs(session_id).rglob("*"))

    remove_snapshot(session_id, ids[0])

    assert not _inputs(session_id).exists()
    assert not session_root(session_id).exists()


@pytest.mark.asyncio
async def test_a_session_that_still_has_a_snapshot_keeps_its_inputs(snapshots):
    from backend.services.round_snapshots.layout import remove_snapshot

    session_id, ids = await snapshots([
        (GATE_ANNOTATION_CONFIRMATION, 1, 51),
        (GATE_ROUND_WAIT, 3, 52),
    ])

    remove_snapshot(session_id, ids[1])

    assert any(_inputs(session_id).rglob("*"))
    assert _remaining(session_id) == [ids[0]]


@pytest.mark.asyncio
async def test_the_record_of_a_decision_with_no_restore_point_survives(snapshots):
    """The inputs go; the account of what happened does not.

    A ``.failed.json`` is the only evidence that a decision was reached and no
    snapshot was written for it. It costs nothing to keep and it is the first
    thing an operator asks about.
    """
    import json

    from backend.services.round_snapshots.layout import (
        failed_record_path,
        remove_snapshot,
        snapshots_root,
    )

    session_id, ids = await snapshots([(GATE_ROUND_WAIT, 3, 51)])
    failed = failed_record_path(session_id, "r004-round-g60")
    failed.write_text(json.dumps({"reason": "save_failed"}), encoding="utf-8")

    remove_snapshot(session_id, ids[0])

    assert not _inputs(session_id).exists()
    assert failed.is_file()
    assert snapshots_root(session_id).is_dir()


@pytest.mark.asyncio
async def test_a_capture_that_is_staging_keeps_the_inputs_it_is_about_to_name(
    snapshots,
):
    """The one race this reclamation could lose.

    A capture stages its session files, then its inputs, then publishes. Delete
    the last existing bundle in that window and the inputs the in-flight
    capture is about to declare would go with it -- producing a published
    snapshot that cannot be restored, which is worse than the space.
    """
    from backend.services.round_snapshots.layout import remove_snapshot, staging_root

    session_id, ids = await snapshots([(GATE_ROUND_WAIT, 3, 51)])
    (staging_root(session_id) / "live-attempt").mkdir(parents=True, exist_ok=True)

    remove_snapshot(session_id, ids[0])

    assert _remaining(session_id) == []
    assert any(_inputs(session_id).rglob("*"))


@pytest.mark.asyncio
async def test_the_delete_route_gives_up_what_a_prune_would(snapshots):
    """One removal, two callers. Two implementations is how they drifted."""
    import types

    from backend.routes.round_snapshots import delete_session_snapshot

    session_id, ids = await snapshots([(GATE_ROUND_WAIT, 3, 51)])
    request = types.SimpleNamespace(state=types.SimpleNamespace(user_id="owner-1"))

    answer = await delete_session_snapshot(session_id, ids[0], request)

    assert answer["status"] == "deleted"
    assert not _inputs(session_id).exists()


# ------------------------------------------------------------------ accounting


@pytest.mark.asyncio
async def test_a_shared_dump_is_counted_once(snapshots):
    """Summing manifests would double-count what the filesystem shares.

    Two decisions reached without the kernel executing anything hold the same
    dump through one inode, so the space giving one up would free is not the
    sum of what the two manifests declare. The dump is made large enough here
    that the bundles' small files cannot mask the difference.
    """
    dump = b"x" * 200_000
    session_id, ids = await snapshots(
        [(GATE_ROUND_WAIT, 3, 51), (GATE_ROUND_WAIT, 4, 51)], dump=dump,
    )
    assert len(ids) == 2

    first = layout.snapshot_dir(session_id, ids[0]) / "runtime/session.pkl"
    second = layout.snapshot_dir(session_id, ids[1]) / "runtime/session.pkl"
    assert first.stat().st_ino == second.stat().st_ino

    listing = read_session_listing(session_id)
    declared = sum(row["bytes_total"] for row in listing["snapshots"])
    assert declared > 2 * len(dump) * 0.9
    # One copy of the dump plus the bundles' small files, not two copies.
    assert listing["bytes_total"] < declared
    assert listing["bytes_total"] > len(dump)


def _bytes_under(root, seen=None):
    """Regular files under *root*, an inode counted once -- ``du`` on a tree."""
    seen = set() if seen is None else seen
    total = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        info = path.lstat()
        key = (info.st_dev, info.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += info.st_size
    return total


@pytest.mark.asyncio
async def test_the_session_total_includes_the_inputs_every_bundle_depends_on(
    snapshots, monkeypatch, tmp_path,
):
    """The number an operator budgets and reclaims by.

    The inputs are stored once per session, they are usually the second largest
    thing a session's snapshots hold, and giving up the last bundle now gives
    them up too. A total that walked only ``snapshots/`` under-reported by the
    whole input set and made that reclamation look like it freed nothing.

    The input here is deliberately large relative to everything else in the
    fixture. With the seed's fifteen-byte matrix the assertion held either way,
    which is exactly the sort of green that proves nothing.
    """
    from backend.config import get_session_uploads
    from backend.services.round_snapshots.layout import snapshots_root

    big = b"i" * 400_000
    original = _seed_session

    def _seed_with_a_real_input(session_id, **kwargs):
        digest = original(session_id, **kwargs)
        (get_session_uploads(session_id) / "matrix.h5ad").write_bytes(big)
        return digest

    monkeypatch.setattr(
        "backend.tests.test_round_snapshot_prune._seed_session",
        _seed_with_a_real_input,
    )
    session_id, _ids = await snapshots([(GATE_ROUND_WAIT, 3, 51)])

    inputs = _bytes_under(_inputs(session_id))
    assert inputs >= len(big)

    total = read_session_listing(session_id)["bytes_total"]
    assert total >= _bytes_under(snapshots_root(session_id)) + inputs


@pytest.mark.asyncio
async def test_a_capture_in_flight_is_not_counted_as_occupancy(snapshots):
    """Otherwise the number jumps during a decision and settles afterwards."""
    from backend.services.round_snapshots.layout import staging_root

    session_id, _ids = await snapshots([(GATE_ROUND_WAIT, 3, 51)])
    settled = read_session_listing(session_id)["bytes_total"]

    staging = staging_root(session_id) / "live-attempt"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "half-copied.bin").write_bytes(b"y" * 500_000)

    assert read_session_listing(session_id)["bytes_total"] == settled


@pytest.mark.asyncio
async def test_an_installation_with_snapshots_off_reports_no_occupancy(monkeypatch):
    monkeypatch.setattr(_config, "SNAPSHOT_DIR", None)
    assert read_session_listing("nobody")["bytes_total"] == 0
