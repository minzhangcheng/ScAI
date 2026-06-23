"""Regression: forward (non-restore) auto-snapshots must form a parent chain.

Bug: during a normal forward run with no restores, every auto-snapshot got
parent_id=None (flat list, no chain), because parent_id was sourced only from
last_restored_snapshot_id which is None outside of a restore. The lineage needs
a head pointer (last_snapshot_id) that advances on every creation.
"""
from backend.models.schemas import AnalysisState, Snapshot
from backend.utils.snapshot import SnapshotManager


def _state():
    return AnalysisState(session_id="t", user_id="u")


def test_first_forward_snapshot_is_root():
    st = _state()
    ctx = SnapshotManager.compute_snapshot_context(st, "pre_annotation")
    assert ctx["parent_id"] is None
    assert ctx["branch_label"] == "main"


def test_second_forward_snapshot_chains_to_first():
    st = _state()
    # Simulate first snapshot having been created and registered.
    s1 = Snapshot(name="a", phase="pre_annotation", round_number=1,
                  parent_id=None, branch_label="main")
    st.snapshots.append(s1)
    st.last_snapshot_id = s1.id  # head advances on creation

    ctx = SnapshotManager.compute_snapshot_context(st, "pre_direction")
    assert ctx["parent_id"] == s1.id, "forward snapshot must chain to lineage head"
    assert ctx["branch_label"] == "main"


def test_long_forward_run_forms_linear_chain():
    st = _state()
    prev = None
    ids = []
    for i, phase in enumerate(["pre_annotation", "pre_direction", "pre_round_wait",
                               "pre_execution_full", "pre_round_wait"]):
        ctx = SnapshotManager.compute_snapshot_context(st, phase)
        assert ctx["parent_id"] == prev, f"snapshot {i} should chain to previous"
        s = Snapshot(name=ctx["name"], phase=phase, round_number=st.current_round,
                     parent_id=ctx["parent_id"], branch_label=ctx["branch_label"])
        st.snapshots.append(s)
        st.last_snapshot_id = s.id
        prev = s.id
        ids.append(s.id)
    # Every snapshot except the first has a non-None parent → a single chain.
    roots = [s for s in st.snapshots if s.parent_id is None]
    assert len(roots) == 1, "a forward run must produce exactly one root"


def test_restore_then_forward_chains_onto_restore_point():
    st = _state()
    # Existing linear chain a -> b
    a = Snapshot(name="a", phase="pre_annotation", round_number=1, parent_id=None, branch_label="main")
    b = Snapshot(name="b", phase="pre_direction", round_number=1, parent_id=a.id, branch_label="main")
    st.snapshots.extend([a, b])
    # User restores 'a': both markers point at a.
    st.last_restored_snapshot_id = a.id
    st.last_snapshot_id = a.id

    ctx1 = SnapshotManager.compute_snapshot_context(st, "pre_direction")
    assert ctx1["parent_id"] == a.id
    assert ctx1["branch_label"] != "main"  # diverging branch
    s1 = Snapshot(name=ctx1["name"], phase="pre_direction", round_number=1,
                  parent_id=ctx1["parent_id"], branch_label=ctx1["branch_label"])
    st.snapshots.append(s1)
    st.last_snapshot_id = s1.id
    st.last_restored_snapshot_id = None  # cleared after one snapshot

    # Next forward snapshot must chain onto s1 and stay on the same branch.
    ctx2 = SnapshotManager.compute_snapshot_context(st, "pre_round_wait")
    assert ctx2["parent_id"] == s1.id
    assert ctx2["branch_label"] == ctx1["branch_label"]
