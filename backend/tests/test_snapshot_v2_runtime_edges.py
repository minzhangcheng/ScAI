"""Runtime/boundary regressions in Snapshot V2 that only fire outside the happy path.

Each test here pins a defect that made the subsystem fail in normal use — which is
why Snapshot V2 was switched off by default. None of them are hypothetical: every
one was reproduced against the real code before being fixed.
"""
import asyncio
import re
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.snapshot_v2.runtime_registry import (
    _is_anndata_type,
    _serializer_for,
)
from backend.services.snapshot_v2.safe_point import (
    get_snapshot_safe_point_manager,
)


# --- 1. AnnData-derived helper types must not be routed to the h5ad writer ---
#
# The kernel introspection template tags a genuine AnnData as "AnnData(<shape>)"
# and everything else as "<module>.<qualname>". A substring match on "anndata"
# therefore also caught Raw / AxisArrays / ArrayView, none of which have .write().
# Because any match is marked *required* with serializer="h5ad", one ordinary
# `raw = adata.raw` left in the namespace failed every capture closed.


@pytest.mark.parametrize("type_str", [
    "AnnData((2, 13704))",          # what introspection emits for a real AnnData
    "AnnData((50, 10))",
    "AnnData",
    "anndata.AnnData",
    "anndata.core.AnnData",
])
def test_real_anndata_still_uses_h5ad(type_str):
    assert _is_anndata_type(type_str) is True
    assert _serializer_for("adata", type_str) == "h5ad"


@pytest.mark.parametrize("type_str", [
    "anndata._core.raw.Raw",                        # raw = adata.raw
    "anndata._core.aligned_mapping.AxisArrays",     # om = adata.obsm
    "anndata._core.views.ArrayView",                # x = adata[:2].X
    "anndata._core.views.DataFrameView",
    "anndata._core.aligned_mapping.Layers",
])
def test_anndata_internal_helpers_fall_through_to_dill(type_str):
    assert _is_anndata_type(type_str) is False
    assert _serializer_for("raw", type_str) == "dill"


def test_unrelated_types_are_unaffected():
    assert _is_anndata_type("pandas.core.frame.DataFrame") is False
    assert _is_anndata_type("") is False


# --- 2. The kernel templates are .format() strings ---
#
# A literal brace in one of these templates is read by str.format as a
# replacement field. A comment containing `{0: 'T cell'}` was enough to make
# every AnnData capture die with "Replacement index 0 out of range".


def test_kernel_templates_contain_no_stray_replacement_fields():
    from backend.services.snapshot_v2.runtime_parts import kernel_templates

    source = Path(kernel_templates.__file__).read_text(encoding="utf-8")
    allowed = {"var_name_repr!r", "h5ad_path_repr!r", "names_repr!r", "dill_path_repr!r"}

    offenders = {}
    for match in re.finditer(
        r'^(_[A-Z_]+)\s*=\s*"""(.*?)"""', source, re.S | re.M,
    ):
        name, body = match.group(1), match.group(2)
        if name == "_INTROSPECT_CODE":
            continue  # executed directly, never passed through .format()
        fields = set(re.findall(r"(?<!\{)\{([^{}]*)\}(?!\})", body)) - allowed
        if fields:
            offenders[name] = sorted(fields)

    assert not offenders, (
        "These .format() templates contain braces that are not known placeholders; "
        "str.format will treat them as replacement fields and the capture will fail "
        f"at runtime: {offenders}"
    )


def test_anndata_template_formats_with_only_its_declared_placeholders():
    from backend.services.snapshot_v2.runtime_parts.kernel_templates import (
        _SAVE_ANNDATA_TEMPLATE,
    )

    # Must not raise IndexError/KeyError.
    rendered = _SAVE_ANNDATA_TEMPLATE.format(
        var_name_repr="adata", h5ad_path_repr="/tmp/x.h5ad",
    )
    compile(rendered, "<anndata-template>", "exec")


# --- 3. Non-string uns keys must not fail the capture ---
#
# h5py group names must be str. A cluster-id -> label mapping under uns is a
# routine scanpy pattern and used to abort capture with
# "'int' object has no attribute 'split'" from deep inside h5py.


def _run_anndata_template(adata, tmp_path):
    from backend.services.snapshot_v2.runtime_parts.kernel_templates import (
        _SAVE_ANNDATA_TEMPLATE,
    )

    namespace = {"__builtins__": __builtins__, "probe": adata}
    code = _SAVE_ANNDATA_TEMPLATE.format(
        var_name_repr="probe", h5ad_path_repr=str(tmp_path / "o.h5ad"),
    )
    exec(code, namespace)


@pytest.mark.parametrize("uns_value", [
    pytest.param({"a": "T cell"}, id="string-keys-baseline"),
    pytest.param({0: "T cell", 1: "B cell"}, id="int-keys"),
    pytest.param({(1, 2): "x"}, id="tuple-key"),
    pytest.param({"l2": {2: "x"}}, id="nested-int-keys"),
])
def test_uns_keys_that_are_not_strings_do_not_break_capture(tmp_path, uns_value):
    anndata = pytest.importorskip("anndata")
    import numpy as np

    adata = anndata.AnnData(np.ones((4, 3)))
    adata.uns["cluster_labels"] = uns_value

    _run_anndata_template(adata, tmp_path)  # must not raise

    assert (tmp_path / "o.h5ad").exists()


def test_numpy_integer_uns_keys_do_not_break_capture(tmp_path):
    anndata = pytest.importorskip("anndata")
    import numpy as np

    adata = anndata.AnnData(np.ones((4, 3)))
    adata.uns["m"] = {np.int64(3): "X"}

    _run_anndata_template(adata, tmp_path)

    assert (tmp_path / "o.h5ad").exists()


def test_host_sanitizer_agrees_with_the_kernel_template(tmp_path):
    """The two implementations are documented as needing to stay in sync."""
    anndata = pytest.importorskip("anndata")
    import numpy as np
    from backend.services.snapshot_v2.runtime_parts.anndata_sanitize import (
        sanitize_anndata_for_h5ad,
    )

    adata = anndata.AnnData(np.ones((4, 3)))
    adata.uns["cluster_labels"] = {0: "T cell", 1: "B cell"}

    sanitize_anndata_for_h5ad(adata).write(str(tmp_path / "host.h5ad"))

    assert (tmp_path / "host.h5ad").exists()


# --- 4. The capture safe point must not outlive a cancelled capture ---
#
# asyncio.CancelledError is a BaseException, so the two `except` clauses in
# _capture_impl never saw it and _release_safe_point never ran. The hold lives in
# a process-wide singleton with no TTL and no reaper, so every later capture and
# restore for that session failed SAFE_POINT_BUSY until the server restarted.


def test_capture_releases_its_safe_point_in_a_finally():
    """Source guard: the release must be structural, not per-except-clause.

    Driving a real cancellation through _capture_impl needs a fully wired
    service, so assert the shape instead — which is exactly what was wrong: the
    releases sat inside `except SnapshotV2Error` / `except Exception`, and
    CancelledError is neither.
    """
    import inspect

    from backend.services.snapshot_v2.service import SnapshotService

    src = inspect.getsource(SnapshotService._capture_impl)

    assert "finally:" in src, (
        "_capture_impl has no finally: a cancelled capture (CancelledError is a "
        "BaseException) would leak the safe point and leave the session "
        "permanently un-snapshottable"
    )

    finally_body = src[src.index("finally:"):]
    assert "_release_safe_point" in finally_body, (
        "the finally block does not release the safe point"
    )


def test_capture_impl_does_not_rely_on_except_clauses_for_release():
    """Every release site must be reachable when a BaseException unwinds."""
    import inspect
    import textwrap

    from backend.services.snapshot_v2.service import SnapshotService

    src = textwrap.dedent(inspect.getsource(SnapshotService._capture_impl))
    tree = __import__("ast").parse(src)

    releases_in_finally = False
    for node in __import__("ast").walk(tree):
        if isinstance(node, __import__("ast").Try) and node.finalbody:
            body_src = "".join(
                __import__("ast").dump(stmt) for stmt in node.finalbody
            )
            if "_release_safe_point" in body_src:
                releases_in_finally = True

    assert releases_in_finally, (
        "no try/finally in _capture_impl releases the safe point"
    )


# --- 5. An orphan pending-import marker must not brick GC or deletion ---
#
# Import marks every planned id pending BEFORE publishing any manifest, so a
# crash in between leaves a marker whose manifest never lands. GC loaded a
# manifest per marker and raised, aborting the sweep; deletion `continue`d
# before reaching the marker unlink, so the orphan was unrecoverable and both
# stayed broken forever.


def _paths(tmp_path):
    class P:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"
    p = P()
    for d in (p.root, p.objects, p.staging, p.journals, p.integrity, p.committed):
        d.mkdir(parents=True, exist_ok=True)
    return p


def test_gc_sweep_survives_an_orphan_pending_import_marker(tmp_path):
    from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
    from backend.services.snapshot_v2.object_store import ManifestStore

    paths = _paths(tmp_path)
    ManifestStore._pending_marker_path(paths.committed, "never-published").touch()

    collector = SnapshotGarbageCollector("gc-probe", paths)

    # mark() is where the pending-import markers are walked; a marker whose
    # manifest never landed used to raise here and abort the whole sweep,
    # which permanently disabled GC *and* every delete for the session.
    marked = collector.mark()
    assert marked == set()

    result = collector.sweep(marked=marked)  # must not raise
    assert result is not None


def test_import_rollback_removes_an_orphan_marker(tmp_path):
    from backend.services.snapshot_v2.deletion import DeletionService
    from backend.services.snapshot_v2.object_store import ManifestStore

    paths = _paths(tmp_path)
    marker = ManifestStore._pending_marker_path(paths.committed, "orphan")
    marker.touch()

    DeletionService("del-probe", paths).delete_pending_import_manifests(["orphan"])

    assert not marker.exists(), (
        "orphan marker survived rollback; it would keep failing GC and every delete"
    )


def test_import_rollback_refuses_to_delete_a_published_snapshot(tmp_path):
    """Only a live pending-import marker authorizes deleting a manifest."""
    from backend.services.snapshot_v2.deletion import DeletionService

    from backend.tests.snapshot_v2_fixtures import (
        make_snapshot_v2_manifest,
        write_manifest_to_store,
    )

    paths = _paths(tmp_path)
    manifest = make_snapshot_v2_manifest(snapshot_id="already-visible")
    write_manifest_to_store(manifest, paths.committed, paths.integrity)
    published = paths.committed / "already-visible.json"
    sig = paths.integrity / "already-visible.sig"
    assert published.exists() and sig.exists()

    # A stale/retried rollback naming an id that has since been published.
    DeletionService("del-probe", paths).delete_pending_import_manifests(["already-visible"])

    assert published.exists(), "rollback destroyed a committed snapshot"
    assert sig.exists()


# --- 6. A pending safe-point reservation must be handed over, not deadlocked ---
#
# A restore job reserves the safe point and waits for the workflow to acknowledge
# it at a boundary. _acquire_unlocked refuses while any request is *pending*, so
# the workflow's own boundary capture got SAFE_POINT_BUSY -> RequiredSnapshotBlocked
# -> the graph invocation aborted -> the workflow never reached another
# acknowledgement site -> the restore waited out its timeout. The user's restore
# request destroyed the running analysis and then failed.


@pytest.mark.asyncio
async def test_boundary_capture_hands_over_a_pending_safe_point(monkeypatch):
    from backend.workflow.core import SingleCellAnalysisWorkflow
    from backend.services.snapshot_v2 import safe_point as safe_point_mod

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "handover-probe"

    acknowledged = []

    async def _await_restore():
        acknowledged.append(True)

    workflow.await_restore_safe_point = _await_restore

    class _Manager:
        def has_pending_request(self, _sid):
            return True

    monkeypatch.setattr(
        safe_point_mod, "get_snapshot_safe_point_manager", lambda: _Manager(),
    )

    # Stop right after the handover: consume_snapshot_skip returning True makes
    # the boundary a no-op, so the test exercises the handover in isolation.
    from backend.services.snapshot_v2 import failure_gate

    monkeypatch.setattr(failure_gate, "consume_snapshot_skip", lambda *_a: True)

    await SingleCellAnalysisWorkflow._create_auto_snapshot(
        workflow, MagicMock(), "pre_round_wait",
    )

    assert acknowledged, (
        "the boundary did not acknowledge the pending reservation; the capture "
        "would fail SAFE_POINT_BUSY and abort the graph"
    )


@pytest.mark.asyncio
async def test_boundary_capture_does_not_wait_when_nothing_is_pending(monkeypatch):
    """The handover must be conditional — an already-held foreign hold has to
    keep failing closed rather than turning into an unbounded wait."""
    from backend.workflow.core import SingleCellAnalysisWorkflow
    from backend.services.snapshot_v2 import safe_point as safe_point_mod
    from backend.services.snapshot_v2 import failure_gate

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "no-pending-probe"

    waited = []

    async def _await_restore():
        waited.append(True)

    workflow.await_restore_safe_point = _await_restore

    class _Manager:
        def has_pending_request(self, _sid):
            return False

    monkeypatch.setattr(
        safe_point_mod, "get_snapshot_safe_point_manager", lambda: _Manager(),
    )
    monkeypatch.setattr(failure_gate, "consume_snapshot_skip", lambda *_a: True)

    await SingleCellAnalysisWorkflow._create_auto_snapshot(
        workflow, MagicMock(), "pre_round_wait",
    )

    assert not waited


# --- 7. A restore must retire the run that predates it ---
#
# LangGraph opens the checkpoint DB in WAL mode, so it is three files, and apply()
# replaces only the main one. With a live connection still open, the stale
# -wal/-shm survive the replacement and are replayed over it — a rebuilt
# connection reads the PRE-restore position, and when the stale connection finally
# closes it checkpoints those frames into the restored file, destroying it. The
# restore still reports committed, because verification hashes the main file while
# it does briefly hold the staged bytes.


def test_replacing_a_wal_db_under_a_live_connection_loses_the_restore(tmp_path):
    """The hazard itself, pinned. If this ever stops holding, the retire step
    below can be simplified — until then it is load-bearing."""
    live_path = tmp_path / "checkpoints.sqlite"

    live = sqlite3.connect(live_path)
    live.execute("PRAGMA journal_mode=WAL")
    live.execute("CREATE TABLE t(v TEXT)")
    live.execute("INSERT INTO t VALUES('PRE_RESTORE')")
    live.commit()
    try:
        staged = tmp_path / "staged.sqlite"
        st = sqlite3.connect(staged)
        st.execute("PRAGMA journal_mode=WAL")
        st.execute("CREATE TABLE t(v TEXT)")
        st.execute("INSERT INTO t VALUES('RESTORED')")
        st.commit()
        st.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        st.close()

        # apply(): replace the main file only, connection still open
        tmp = live_path.with_suffix(".tmp")
        tmp.write_bytes(staged.read_bytes())
        tmp.replace(live_path)

        fresh = sqlite3.connect(live_path)
        seen = [r[0] for r in fresh.execute("SELECT v FROM t")]
        fresh.close()
    finally:
        live.close()

    assert seen == ["PRE_RESTORE"], (
        "expected the stale WAL to shadow the restored main file; if this now "
        "reads RESTORED, sqlite semantics changed and the retire step should be "
        "re-evaluated"
    )


def test_closing_the_connection_first_makes_the_restore_visible(tmp_path):
    """The fix, in miniature: close before replacing and everyone sees it."""
    live_path = tmp_path / "checkpoints.sqlite"

    live = sqlite3.connect(live_path)
    live.execute("PRAGMA journal_mode=WAL")
    live.execute("CREATE TABLE t(v TEXT)")
    live.execute("INSERT INTO t VALUES('PRE_RESTORE')")
    live.commit()
    live.close()   # <- what _retire_superseded_workflow does before apply

    staged = tmp_path / "staged.sqlite"
    st = sqlite3.connect(staged)
    st.execute("PRAGMA journal_mode=WAL")
    st.execute("CREATE TABLE t(v TEXT)")
    st.execute("INSERT INTO t VALUES('RESTORED')")
    st.commit()
    st.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    st.close()

    tmp = live_path.with_suffix(".tmp")
    tmp.write_bytes(staged.read_bytes())
    tmp.replace(live_path)

    fresh = sqlite3.connect(live_path)
    seen = [r[0] for r in fresh.execute("SELECT v FROM t")]
    fresh.close()

    assert seen == ["RESTORED"]


@pytest.mark.asyncio
async def test_retire_closes_the_checkpoint_connection_and_evicts(monkeypatch):
    from backend.services.snapshot_v2.service import SnapshotService
    from backend import state as state_mod

    service = SnapshotService.__new__(SnapshotService)
    service.session_id = "retire-probe"
    service.safe_point = get_snapshot_safe_point_manager()

    closed = []

    class _Conn:
        async def close(self):
            closed.append(True)

    class _Workflow:
        def __init__(self):
            self._db_conn = _Conn()
            self.checkpointer = object()
            self.compiled_graph = object()

    workflow = _Workflow()
    monkeypatch.setitem(state_mod.workflows, "retire-probe", workflow)

    await service._retire_superseded_workflow(workflow)

    assert get_snapshot_safe_point_manager().is_superseded("retire-probe"), (
        "the signal must live on the process-wide manager: the workflow object "
        "is evicted, so a flag stored on it is unreachable to the bioinformatics "
        "step loop and every other site frozen on this safe point"
    )
    get_snapshot_safe_point_manager().clear_superseded("retire-probe")
    assert closed, "the pre-restore checkpoint connection was not closed"
    # Cleared so core.py rebuilds them against the restored file.
    assert workflow._db_conn is None
    assert workflow.checkpointer is None
    assert workflow.compiled_graph is None
    assert "retire-probe" not in state_mod.workflows


@pytest.mark.asyncio
async def test_retire_does_not_evict_a_successor_workflow(monkeypatch):
    """Eviction is by identity: a workflow rebuilt concurrently is already
    reading the restored checkpoint and must survive."""
    from backend.services.snapshot_v2.service import SnapshotService
    from backend import state as state_mod

    service = SnapshotService.__new__(SnapshotService)
    service.session_id = "successor-probe"
    service.safe_point = get_snapshot_safe_point_manager()

    old = MagicMock(_db_conn=None)
    successor = MagicMock(_db_conn=None)
    monkeypatch.setitem(state_mod.workflows, "successor-probe", successor)

    await service._retire_superseded_workflow(old)

    assert state_mod.workflows.get("successor-probe") is successor


@pytest.mark.asyncio
async def test_superseded_run_unwinds_without_touching_session_state():
    """The restore already persisted the authoritative state; a terminal status
    written by the doomed run is exactly the corruption being avoided."""
    from backend.workflow.core import SingleCellAnalysisWorkflow
    from backend.workflow.helpers import WorkflowSupersededError
    from backend.services.snapshot_v2 import safe_point as safe_point_mod

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "superseded-probe"

    async def _ack():
        return None

    workflow.acknowledge_at_boundary = _ack

    class _Manager:
        def is_paused(self, _sid):
            return True

        async def wait_for(self, _sid):
            return None

        def is_superseded(self, _sid):
            # What a committed restore leaves behind for the frozen run.
            return True

    original = safe_point_mod.get_snapshot_safe_point_manager
    safe_point_mod.get_snapshot_safe_point_manager = lambda: _Manager()
    try:
        with pytest.raises(WorkflowSupersededError):
            await SingleCellAnalysisWorkflow.await_restore_safe_point(workflow)
    finally:
        safe_point_mod.get_snapshot_safe_point_manager = original


@pytest.mark.asyncio
async def test_a_normal_safe_point_release_still_resumes():
    """Only a restore retires the run; an ordinary capture pause must not."""
    from backend.workflow.core import SingleCellAnalysisWorkflow
    from backend.services.snapshot_v2 import safe_point as safe_point_mod

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "normal-probe"

    async def _ack():
        return None

    workflow.acknowledge_at_boundary = _ack

    class _Manager:
        def is_paused(self, _sid):
            return True

        async def wait_for(self, _sid):
            return None

        def is_superseded(self, _sid):
            return False

    original = safe_point_mod.get_snapshot_safe_point_manager
    safe_point_mod.get_snapshot_safe_point_manager = lambda: _Manager()
    try:
        await SingleCellAnalysisWorkflow.await_restore_safe_point(workflow)
    finally:
        safe_point_mod.get_snapshot_safe_point_manager = original


# --- 15. A Skip must not outlive the run it was armed for -------------------
#
# consume_snapshot_skip only clears the marker at a *matching* boundary, so a
# Skip armed at a boundary the graph never reached again stayed on the session
# indefinitely and suppressed a required capture in some later, unrelated run.
# The wildcard form ("*", armed when the failing phase is unknown) suppressed
# whichever capture happened to come first.


def test_terminate_clears_a_pending_skip_marker():
    from backend.services.snapshot_v2.failure_gate import terminate_snapshot_failure

    state = MagicMock()
    state.skip_required_snapshot_phase = "pre_annotation"

    terminate_snapshot_failure(state)

    assert state.skip_required_snapshot_phase is None, (
        "a Skip survived the Terminate that abandoned its run"
    )


@pytest.mark.asyncio
async def test_run_clears_a_stale_skip_marker(monkeypatch):
    from backend.models.schemas import TaskStatus
    from backend.routes import lifecycle

    state = MagicMock()
    state.status = TaskStatus.COMPLETED
    state.skip_required_snapshot_phase = "*"
    state.restore_resume_kind = None
    state.restore_branch_id = None
    state.snapshot_failure = None
    state.pending_required_snapshot = None
    state.wait_input_reason = None

    async def _owner(session_id, user_id):
        return state

    async def _save(*a, **k):
        return None

    monkeypatch.setattr(lifecycle, "require_session_ownership", _owner)
    monkeypatch.setattr(lifecycle, "async_save_session", _save)
    # A MagicMock attribute reads as a live journal to the recovery guard.
    monkeypatch.setattr(lifecycle, "assert_recovery_clear", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "is_snapshot_failure_blocked", lambda _s: False)

    class _Req:
        state = type("S", (), {"user_id": "u1"})()

    await lifecycle.run_analysis("skip-marker-probe", _Req())

    assert state.skip_required_snapshot_phase is None, (
        "a Skip from a previous run carried into this one and would suppress "
        "its first required capture"
    )


# --- 8. The superseded signal must reach every site frozen on the safe point ---
#
# The first version of this fix stored the flag on the workflow object — which the
# restore evicts, so the bioinformatics step loop (where an analysis spends most
# of its time, and which waits on the safe point directly rather than through
# await_restore_safe_point) could never see it. The run then resumed on a closed
# checkpoint connection and died as a generic failure.


def test_superseded_signal_lives_on_the_process_wide_manager():
    from backend.services.snapshot_v2.safe_point import (
        get_snapshot_safe_point_manager,
    )

    sp = get_snapshot_safe_point_manager()
    sid = "signal-scope-probe"
    try:
        assert sp.is_superseded(sid) is False
        sp.mark_superseded(sid)
        # Reachable from anywhere holding only the session id — no workflow
        # object required, because the restore evicted it.
        assert get_snapshot_safe_point_manager().is_superseded(sid) is True
    finally:
        sp.clear_superseded(sid)
    assert sp.is_superseded(sid) is False


def test_bioinformatics_step_loop_honours_the_superseded_signal():
    """The step loop waits on the safe point directly; it must unwind too."""
    import inspect

    from backend.agents.bioinformatics_agent import BioinformaticsAgent

    src = inspect.getsource(BioinformaticsAgent)
    idx = src.find("await sp.wait_for(session_id)")
    assert idx != -1, "step-loop safe-point wait not found"

    after = src[idx:idx + 900]
    assert "is_superseded" in after, (
        "the bioinformatics step loop resumes after a restore without checking "
        "whether that restore superseded this run — it would continue on a "
        "closed checkpoint connection"
    )
    assert "WorkflowSupersededError" in after


@pytest.mark.asyncio
async def test_superseded_close_does_not_persist_the_pre_restore_kernel():
    """close() ends in sandbox.stop(), which saves. After a restore installed the
    authoritative pkl/notebook, saving would overwrite them."""
    from backend.services.snapshot_v2.safe_point import (
        get_snapshot_safe_point_manager,
    )
    from backend.workflow.core import SingleCellAnalysisWorkflow

    sid = "close-probe"
    stop_calls = []

    class _Sandbox:
        async def stop(self, *, save=True):
            stop_calls.append(save)

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = sid
    workflow.sandbox = _Sandbox()
    workflow._db_conn = None
    workflow.llm_manager = None

    sp = get_snapshot_safe_point_manager()
    sp.mark_superseded(sid)
    try:
        await workflow.close()
    finally:
        sp.clear_superseded(sid)

    assert stop_calls == [False], (
        "a superseded close persisted the pre-restore kernel over the restored one"
    )


@pytest.mark.asyncio
async def test_ordinary_close_still_persists():
    from backend.workflow.core import SingleCellAnalysisWorkflow

    stop_calls = []

    class _Sandbox:
        async def stop(self, *, save=True):
            stop_calls.append(save)

    workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "ordinary-close-probe"
    workflow.sandbox = _Sandbox()
    workflow._db_conn = None
    workflow.llm_manager = None

    await workflow.close()

    assert stop_calls == [True]


# --- 13. Preflight must reject an unreachable resume target -----------------
#
# _check_known_node/_check_known_gate only ran in stage_checkpoint, which is
# AFTER apply() replaced the workspace, notebook, session.json, checkpoint and
# runtime. A snapshot from another instance (or from before a node rename)
# passed preflight, destroyed the live session, and only then was rejected —
# leaving the session recovery-blocked over a read-only check.


def _preflight_paths(tmp_path):
    class _P:
        def __init__(self, root):
            self.root = root / "v2"
            self.objects = root / "objects"
            self.staging = root / "staging"
            self.journals = root / "journals"
            self.integrity = root / "integrity"
            self.committed = root / "committed"
            for d in (self.root, self.objects, self.staging,
                      self.journals, self.integrity, self.committed):
                d.mkdir(parents=True, exist_ok=True)
    return _P(tmp_path)


@pytest.mark.parametrize("capsule,expected", [
    (
        {
            "kind": "graph_node",
            "graph_version": "2.0",
            "next_node": "a_node_this_build_does_not_have",
            "planning_stage": "full",
            "workflow_state_ref": {
                "sha256": "d" * 64, "size": 2, "media_type": "application/json",
            },
        },
        "unknown graph node",
    ),
    (
        {
            "kind": "semantic_gate",
            "graph_version": "2.0",
            "gate": "a_gate_this_build_does_not_have",
            "payload_ref": {
                "sha256": "e" * 64, "size": 2, "media_type": "application/json",
            },
        },
        "unknown gate",
    ),
])
def test_preflight_rejects_an_unknown_resume_target(tmp_path, capsule, expected):
    from backend.services.snapshot_v2.preflight import RestorePreflight
    from backend.services.snapshot_v2.models import SnapshotRestoreRequest
    from backend.tests.snapshot_v2_fixtures import (
        make_snapshot_v2_manifest,
        write_manifest_to_store,
    )

    paths = _preflight_paths(tmp_path)
    manifest = make_snapshot_v2_manifest(
        snapshot_id="snap-unknown-target",
        workspace=[],
        continuation=capsule,
    )
    write_manifest_to_store(manifest, paths.committed, paths.integrity)

    result = RestorePreflight(paths).run(
        "snap-unknown-target",
        SnapshotRestoreRequest(snapshot_id="snap-unknown-target"),
    )

    assert result.status != "ready", (
        "preflight declared a snapshot ready whose resume target does not exist "
        "in this build; the rejection would only happen after apply() had "
        "already overwritten the live session"
    )
    assert any(expected in e for e in result.compatibility_errors), (
        f"expected {expected!r} in {result.compatibility_errors}"
    )


# --- 14. post_report must not hijack another producer's failure gate --------
#
# ensure_post_report_retry_context is reachable from generate_report's entry
# guard with any gate already armed. It used to rewrite allowed_actions and the
# typed context unconditionally, converting a pre_restore gate — which forbids
# Skip — into a skippable post_report one, and sending Retry to a producer that
# cannot satisfy the original boundary.


@pytest.mark.parametrize("producer", ["pre_restore", "workflow_boundary", "post_chat"])
def test_post_report_ensure_leaves_a_foreign_gate_alone(producer):
    from backend.services.snapshot_v2.failure_gate import (
        ensure_post_report_retry_context,
    )

    state = MagicMock()
    original = {
        "code": "REQUIRED_SNAPSHOT_BLOCKED",
        "message": "boom",
        "allowed_actions": ["retry", "manage", "terminate"],
        "retry_context": {"producer": producer, "previous_status": "running"},
    }
    state.snapshot_failure = dict(original)

    ensure_post_report_retry_context(state, RuntimeError("boom"))

    assert state.snapshot_failure == original, (
        f"a {producer} gate was rewritten as post_report; Retry would dispatch "
        f"to the wrong producer"
    )
    assert "skip" not in state.snapshot_failure["allowed_actions"], (
        "a Skip was offered on a gate whose producer forbids it"
    )


def test_post_report_ensure_still_upgrades_its_own_gate():
    """The guard must not disable the helper's actual job."""
    from backend.services.snapshot_v2.failure_gate import (
        ensure_post_report_retry_context,
    )

    state = MagicMock()
    state.snapshot_failure = {
        "code": "REQUIRED_SNAPSHOT_BLOCKED",
        "message": "boom",
        "allowed_actions": ["retry", "manage", "terminate"],
        "retry_context": {"producer": "post_report"},   # incomplete
    }

    ensure_post_report_retry_context(state, RuntimeError("boom"))

    assert "skip" in state.snapshot_failure["allowed_actions"]
    assert state.snapshot_failure["retry_context"]["producer"] == "post_report"


def test_post_report_ensure_types_a_bare_gate():
    """A bare service gate (no retry_context at all) is still adopted."""
    from backend.services.snapshot_v2.failure_gate import (
        ensure_post_report_retry_context,
    )

    state = MagicMock()
    state.snapshot_failure = {
        "code": "REQUIRED_SNAPSHOT_BLOCKED",
        "message": "boom",
        "allowed_actions": ["retry", "manage", "terminate"],
    }

    ensure_post_report_retry_context(state, RuntimeError("boom"))

    assert state.snapshot_failure["retry_context"]["producer"] == "post_report"


# --- Restored: the remaining round-2 regression pins -------------------------


def test_with_retry_lets_a_superseded_run_through():
    """with_retry pass-throughs are a closed list. A superseded run caught by
    the generic handler would be retried into a graph position the restore
    deleted, on a connection the restore closed."""
    import inspect

    from backend.workflow import retry as retry_mod

    src = inspect.getsource(retry_mod)
    for exc in (
        "RequiredSnapshotBlocked",
        "WorkflowTerminatedError",
        "WorkflowSupersededError",
    ):
        assert f"except {exc}:" in src, (
            f"{exc} is not passed through with_retry; it would be swallowed by "
            f"the generic except and retried as an ordinary node failure"
        )


def test_run_job_releases_the_safe_point_in_a_finally():
    """Both except handlers persist a journal record BEFORE releasing. That
    write can fail on its own (ENOSPC, read-only mount), which skipped the
    release and stranded the hold — every later capture and restore for the
    session then failed SAFE_POINT_BUSY until a process restart."""
    import inspect

    from backend.services.snapshot_v2.jobs import RestoreJobManager

    src = inspect.getsource(RestoreJobManager._run_job)

    assert "finally:" in src, (
        "_run_job has no finally: a journal-write failure in an except handler "
        "leaks the acquired safe point permanently"
    )
    finally_body = src[src.rindex("finally:"):]
    assert (
        "cancel_safe_point_request" in finally_body or "release" in finally_body
    ), "the finally block does not release the safe point"


def test_every_superseded_entry_point_unwinds_quietly():
    """WorkflowSupersededError must never reach a generic handler that writes a
    terminal status — that would overwrite the state the restore persisted.

    resume()'s pre-graph freeze sat outside every handler and reached the
    headless runner's `except Exception`, which writes TaskStatus.FAILED.
    """
    import inspect

    from backend.workflow.core import SingleCellAnalysisWorkflow
    from backend.services import workflow_runner

    resume_src = inspect.getsource(SingleCellAnalysisWorkflow.resume)
    idx = resume_src.index("await self.await_restore_safe_point()")
    window = resume_src[max(0, idx - 400):idx]
    assert "try:" in window, (
        "resume()'s pre-graph await_restore_safe_point is not inside a try; a "
        "restore committing while frozen there escapes into the caller's "
        "generic handler and is recorded as a failure"
    )

    runner_src = inspect.getsource(workflow_runner)
    assert "except WorkflowSupersededError" in runner_src, (
        "the headless runner converts a superseded run into TaskStatus.FAILED, "
        "overwriting the restored state"
    )


@pytest.mark.asyncio
async def test_recovery_defers_while_a_snapshot_transaction_holds_the_lock():
    """Journal recovery is CRASH recovery, but it ran on every authenticated
    request — including the frontend's 2s poll of the restore's own status
    endpoint — while the restore holds the lock and yields."""
    from backend.services import session_store
    from backend.state import get_session_lock

    sid = "inflight-probe"
    lock = get_session_lock(sid)

    assert session_store._snapshot_transaction_in_flight(sid) is False
    async with lock:
        assert session_store._snapshot_transaction_in_flight(sid) is True, (
            "recovery would run mid-restore and roll back the transaction that "
            "owns the journal"
        )
    assert session_store._snapshot_transaction_in_flight(sid) is False


def test_failed_export_removes_its_partial_archive(tmp_path, monkeypatch):
    """The archive is a multi-GB temp file. Only the success path moves it
    somewhere owned, and the route turns a failure into an HTTP error without
    ever seeing the path."""
    import glob
    import os
    import tempfile

    from backend.services.snapshot_v2.exporter import SnapshotExporter

    staging = tmp_path / "tmp"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging))

    class _Paths:
        def __init__(self, root):
            self.root = root / "v2"
            self.objects = root / "objects"
            self.staging = root / "staging"
            self.journals = root / "journals"
            self.integrity = root / "integrity"
            self.committed = root / "committed"
            for d in (self.root, self.objects, self.staging,
                      self.journals, self.integrity, self.committed):
                d.mkdir(parents=True, exist_ok=True)

    paths = _Paths(tmp_path)
    exporter = SnapshotExporter("leak-probe", paths)

    before = set(glob.glob(os.path.join(str(staging), "v2-export-*")))
    try:
        exporter.export("leak-probe", include_runtime=True)
    except Exception:
        pass
    after = set(glob.glob(os.path.join(str(staging), "v2-export-*")))

    assert after == before, f"partial export archive leaked: {after - before}"


def test_analysis_state_tier_detects_a_missing_object(tmp_path):
    """It used to assert only that ref.sha256 was truthy — a pattern-validated
    non-optional field, so the check could never fail and never touched the
    store."""
    from backend.services.snapshot_v2.live_verify import (
        verify_live_restore_components,
    )
    from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

    objects = tmp_path / "objects"
    objects.mkdir()
    session_dir = tmp_path / "s"
    (session_dir / "states").mkdir(parents=True)
    (session_dir / "workspace").mkdir(parents=True)

    manifest = make_snapshot_v2_manifest(workspace=[])

    with pytest.raises(Exception) as exc:
        verify_live_restore_components(
            session_id="verify-probe",
            target_manifest=manifest,
            session_dir=session_dir,
            workspace_root=session_dir / "workspace",
            states_root=session_dir / "states",
            require_analysis_state=True,
            objects_root=objects,       # empty store
        )

    assert "analysis_state" in str(exc.value), (
        f"the tier passed against an empty object store: {exc.value}"
    )


def test_dynamic_freeze_records_the_gate_a_session_is_parked_at():
    """freeze_dynamic's descriptor branches read keys no producer writes, so
    every post_chat / pre_restore capture of a running session froze as Idle —
    including the one snapshot a user reaches for to undo a restore."""
    from backend.models.schemas import TaskStatus
    from backend.services.snapshot_v2.continuation import ContinuationRegistry

    class _Store:
        def put_bytes(self, data, media_type):
            import hashlib
            from backend.services.snapshot_v2.models import ObjectRef

            class _R:
                ref = ObjectRef(
                    sha256=hashlib.sha256(data).hexdigest(),
                    size=len(data),
                    media_type=media_type,
                )
            return _R()

    registry = ContinuationRegistry(_Store())

    parked = MagicMock()
    parked.status = TaskStatus.WAITING_ANNOTATION
    parked.wait_input_reason = None

    capsule = registry.freeze_dynamic(
        {"kernel_present": True, "current_round": 2},
        analysis_state=parked,
        workflow_state={"current_round": 2},
    )

    assert capsule.kind == "semantic_gate", (
        f"a session parked at the annotation gate froze as {capsule.kind!r}; "
        f"restoring it would resume nothing"
    )
    assert capsule.gate == "annotation_confirmation"


def test_dynamic_freeze_still_idles_when_nothing_is_parked():
    """Only an unambiguous parked gate overrides Idle — a mid-node position
    needs the checkpointer's next, which is not available here."""
    from backend.models.schemas import TaskStatus
    from backend.services.snapshot_v2.continuation import ContinuationRegistry

    registry = ContinuationRegistry(MagicMock())

    running = MagicMock()
    running.status = TaskStatus.RUNNING
    running.wait_input_reason = None

    capsule = registry.freeze_dynamic(
        {"kernel_present": True, "current_round": 1},
        analysis_state=running,
        workflow_state={"current_round": 1},
    )

    assert capsule.kind == "idle"


@pytest.mark.asyncio
async def test_deferred_recovery_does_not_clear_an_existing_block(monkeypatch):
    """Skipping recovery is not evidence that nothing is wrong — clearing the
    block on a deferred pass would unblock a genuinely broken session."""
    from backend.services import session_store
    from backend.state import get_session_lock

    sid = "defer-clear-probe"
    cleared = []
    ran = []

    async def _never(_sid):
        ran.append(True)
        return False, None

    async def _cached(_sid):
        return MagicMock()

    monkeypatch.setattr(session_store, "_recover_snapshot_v2_on_load", _never)
    monkeypatch.setattr(session_store, "_get_cached_session", _cached)
    monkeypatch.setattr(
        session_store, "clear_recovery_blocked",
        lambda *a, **k: (cleared.append(True), False)[1],
    )

    async with get_session_lock(sid):
        await session_store.ensure_session_loaded(sid)

    assert not ran, "recovery ran while a transaction held the session lock"
    assert not cleared, "a deferred recovery pass cleared the recovery block"


# --- 16. A session parked at a HITL gate can acknowledge a restore ----------
#
# The cooperative handshake needs a *running* graph to reach a boundary. A
# session parked at a wait gate never will, so _acknowledge_if_quiescent is its
# only acknowledgement site — but it refused whenever a workflow OBJECT existed,
# and that object outlives its run. A restore requested at a gate had nowhere to
# be acknowledged and could only time out, on a completely idle session.


@pytest.mark.asyncio
@pytest.mark.parametrize("status_value,expected", [
    ("waiting_input", True),
    ("waiting_annotation", True),
    ("running", False),
])
async def test_parked_session_with_a_live_workflow_object_is_quiescent(
    monkeypatch, status_value, expected,
):
    from backend.services.snapshot_v2.jobs import RestoreJobManager
    from backend.services.snapshot_v2.safe_point import SafePointRequest
    from backend import state as state_mod
    from backend.services import workflow_runner

    sid = "parked-probe"

    session = MagicMock()
    session.status = status_value
    session.current_round = 3

    monkeypatch.setitem(state_mod.sessions, sid, session)
    # A workflow OBJECT is registered, but nothing is executing.
    monkeypatch.setitem(state_mod.workflows, sid, MagicMock())
    monkeypatch.setattr(state_mod, "_run_tasks", {}, raising=False)
    monkeypatch.setattr(workflow_runner, "_report_tasks", {}, raising=False)
    monkeypatch.setattr(state_mod, "chat_tasks", {}, raising=False)
    monkeypatch.setattr(state_mod, "chat_sandboxes", {}, raising=False)
    monkeypatch.setattr(state_mod, "initializing_sessions", set(), raising=False)

    acknowledged = []

    class _SP:
        async def acknowledge_safe_point_request(self, *a, **k):
            acknowledged.append(True)
            return True

    mgr = RestoreJobManager.__new__(RestoreJobManager)
    mgr._safe_point = _SP()

    reservation = MagicMock(spec=SafePointRequest)
    reservation.generation = 1

    result = await mgr._acknowledge_if_quiescent(sid, reservation)

    assert result is expected, (
        f"status={status_value!r}: expected quiescent={expected}, got {result}"
    )
    assert bool(acknowledged) is expected


# --- 17. A frozen Skip must survive the restore ------------------------------
#
# resume() reads sandbox_state["skip_nodes"] and injects it into the graph state
# so @with_retry honours a Skip the user already chose. Capture froze it, but
# activate only stashed it under restore_gate_payload — a key nothing reads.


def test_restored_gate_carries_skip_nodes_to_its_consumer():
    from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

    dispatcher = ContinuationDispatcher(MagicMock())

    staged = MagicMock()
    staged.kind = "semantic_gate"
    staged.gate = "round_wait"
    staged.payload = {
        "gate": "round_wait",
        "current_round": 2,
        "skip_nodes": ["bioinformatics_analysis"],
    }
    staged.branch_id = "b"
    staged.branch_label = "b"
    staged.parent_snapshot_id = None
    staged.transaction_id = "tx"
    staged.workflow_state = {}

    analysis_state = MagicMock()
    analysis_state.sandbox_state = {}

    dispatcher.activate(analysis_state, staged)

    assert analysis_state.sandbox_state.get("skip_nodes") == [
        "bioinformatics_analysis"
    ], (
        "the frozen Skip did not reach sandbox_state['skip_nodes']; resume() "
        "would re-enter the node the user skipped past"
    )


# --- 15. A Skip must not outlive the run it was armed for -------------------
#
# consume_snapshot_skip only clears the marker at a *matching* boundary, so a
# Skip armed at a boundary the graph never reached again stayed on the session
# indefinitely and suppressed a required capture in some later, unrelated run.
# The wildcard form ("*", armed when the failing phase is unknown) suppressed
# whichever capture happened to come first.


def test_terminate_clears_a_pending_skip_marker():
    from backend.services.snapshot_v2.failure_gate import terminate_snapshot_failure

    state = MagicMock()
    state.skip_required_snapshot_phase = "pre_annotation"

    terminate_snapshot_failure(state)

    assert state.skip_required_snapshot_phase is None, (
        "a Skip survived the Terminate that abandoned its run"
    )


@pytest.mark.asyncio
async def test_run_clears_a_stale_skip_marker(monkeypatch):
    from backend.models.schemas import TaskStatus
    from backend.routes import lifecycle

    state = MagicMock()
    state.status = TaskStatus.COMPLETED
    state.skip_required_snapshot_phase = "*"
    state.restore_resume_kind = None
    state.restore_branch_id = None
    state.snapshot_failure = None
    state.pending_required_snapshot = None
    state.wait_input_reason = None

    async def _owner(session_id, user_id):
        return state

    async def _save(*a, **k):
        return None

    monkeypatch.setattr(lifecycle, "require_session_ownership", _owner)
    monkeypatch.setattr(lifecycle, "async_save_session", _save)
    # A MagicMock attribute reads as a live journal to the recovery guard.
    monkeypatch.setattr(lifecycle, "assert_recovery_clear", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "is_snapshot_failure_blocked", lambda _s: False)

    class _Req:
        state = type("S", (), {"user_id": "u1"})()

    await lifecycle.run_analysis("skip-marker-probe", _Req())

    assert state.skip_required_snapshot_phase is None, (
        "a Skip from a previous run carried into this one and would suppress "
        "its first required capture"
    )


# --- 15. A Skip must not outlive the run it was armed for -------------------
#
# consume_snapshot_skip only clears the marker at a *matching* boundary, so a
# Skip armed at a boundary the graph never reached again stayed on the session
# indefinitely and suppressed a required capture in some later, unrelated run.
# The wildcard form ("*", armed when the failing phase is unknown) suppressed
# whichever capture happened to come first.


def test_terminate_clears_a_pending_skip_marker():
    from backend.services.snapshot_v2.failure_gate import terminate_snapshot_failure

    state = MagicMock()
    state.skip_required_snapshot_phase = "pre_annotation"

    terminate_snapshot_failure(state)

    assert state.skip_required_snapshot_phase is None, (
        "a Skip survived the Terminate that abandoned its run"
    )


@pytest.mark.asyncio
async def test_run_clears_a_stale_skip_marker(monkeypatch):
    from backend.models.schemas import TaskStatus
    from backend.routes import lifecycle

    state = MagicMock()
    state.status = TaskStatus.COMPLETED
    state.skip_required_snapshot_phase = "*"
    state.restore_resume_kind = None
    state.restore_branch_id = None
    state.snapshot_failure = None
    state.pending_required_snapshot = None
    state.wait_input_reason = None

    async def _owner(session_id, user_id):
        return state

    async def _save(*a, **k):
        return None

    monkeypatch.setattr(lifecycle, "require_session_ownership", _owner)
    monkeypatch.setattr(lifecycle, "async_save_session", _save)
    # A MagicMock attribute reads as a live journal to the recovery guard.
    monkeypatch.setattr(lifecycle, "assert_recovery_clear", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "is_snapshot_failure_blocked", lambda _s: False)

    class _Req:
        state = type("S", (), {"user_id": "u1"})()

    await lifecycle.run_analysis("skip-marker-probe", _Req())

    assert state.skip_required_snapshot_phase is None, (
        "a Skip from a previous run carried into this one and would suppress "
        "its first required capture"
    )
