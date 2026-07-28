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

    assert workflow._restore_superseded is True
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
    workflow._restore_superseded = True

    async def _ack():
        return None

    workflow.acknowledge_at_boundary = _ack

    class _Manager:
        def is_paused(self, _sid):
            return True

        async def wait_for(self, _sid):
            return None

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

    original = safe_point_mod.get_snapshot_safe_point_manager
    safe_point_mod.get_snapshot_safe_point_manager = lambda: _Manager()
    try:
        await SingleCellAnalysisWorkflow.await_restore_safe_point(workflow)
    finally:
        safe_point_mod.get_snapshot_safe_point_manager = original
