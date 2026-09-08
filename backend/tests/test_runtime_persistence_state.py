import json
import io
import asyncio
import builtins
from contextlib import redirect_stdout
from types import SimpleNamespace
from pathlib import Path

import pytest

from backend.utils.sandbox import (
    NotebookState,
    LocalJupyterSandbox,
    _commit_pkl_transaction,
    _sha256_file,
    _sig_path,
    _verify_pkl_integrity,
    recover_regular_save_transaction,
)


def _long_runtime_diagnostic_traceback():
    frames = "\n".join(
        f'  File "/workspace/project/frame_{index}.py", line {index}, in helper\n'
        f"    frame_{index}()"
        for index in range(120)
    )
    return (
        "Traceback (most recent call last):\n"
        f"{frames}\n"
        "TypeError: boolean value of NA is ambiguous"
    )


def test_system_config_keeps_only_active_runtime_memory_ceilings():
    from backend.config._config import SystemConfig

    config = SystemConfig()
    assert config.runtime_max_anndata_total_bytes is None
    assert config.runtime_max_rss_bytes is None
    assert not hasattr(config, "runtime_max_namespace_count")
    assert not hasattr(config, "runtime_max_single_object_bytes")
    assert not hasattr(config, "runtime_max_pkl_bytes")
    assert not hasattr(config, "runtime_max_save_seconds")
    persisted = config.model_dump(mode="json")
    assert "runtime_max_rss_bytes" not in persisted
    assert "runtime_max_save_seconds" not in persisted


@pytest.mark.asyncio
async def test_runtime_metrics_count_dense_backing_allocation_once():
    np = pytest.importorskip("numpy")
    from backend.utils.sandbox_parts.state_tracking import StateTrackingMixin

    dense = np.zeros(1024, dtype=np.uint8)
    namespace = {
        "dense": dense,
        "dense_view": dense[128:512],
        "dense_alias": dense,
    }

    class _KernelProbe:
        async def _execute_unlocked(self, code, **_kwargs):
            output = io.StringIO()
            with redirect_stdout(output):
                exec(code, namespace)
            return SimpleNamespace(success=True, output=output.getvalue())

    metrics = await StateTrackingMixin._runtime_resource_metrics(_KernelProbe())
    offenders = {row["name"]: row for row in metrics["top_offenders"]}

    assert metrics["measurement_basis"] == (
        "estimated_known_buffers_identity_deduplicated"
    )
    assert 1024 <= offenders["dense"]["bytes"] < 1400
    assert offenders["dense_view"]["bytes"] < 512
    assert offenders["dense_alias"]["bytes"] == 0
    assert offenders["dense_alias"]["alias_of"] == "dense"


@pytest.mark.asyncio
async def test_runtime_metrics_deduplicate_anndata_dense_and_sparse_aliases():
    ad = pytest.importorskip("anndata")
    np = pytest.importorskip("numpy")
    sparse = pytest.importorskip("scipy.sparse")
    from backend.utils.sandbox_parts.state_tracking import StateTrackingMixin

    matrix = sparse.csr_matrix(np.arange(200, dtype=np.float32).reshape(20, 10))
    embedding = np.arange(40, dtype=np.float32).reshape(20, 2)
    adata = ad.AnnData(matrix)
    adata.obsm["embedding"] = embedding
    namespace = {
        "adata": adata,
        "adata_alias": adata,
        "sparse_alias": adata.X,
        "dense_alias": adata.obsm["embedding"],
    }

    class _KernelProbe:
        async def _execute_unlocked(self, code, **_kwargs):
            output = io.StringIO()
            with redirect_stdout(output):
                exec(code, namespace)
            return SimpleNamespace(success=True, output=output.getvalue())

    metrics = await StateTrackingMixin._runtime_resource_metrics(_KernelProbe())
    offenders = {row["name"]: row for row in metrics["top_offenders"]}

    assert offenders["adata"]["bytes"] >= matrix.data.nbytes
    assert offenders["adata_alias"] == {
        "name": "adata_alias",
        "type": "AnnData",
        "bytes": 0,
        "alias_of": "adata",
    }
    # The container owns both allocations. Top-level views keep only their
    # tiny Python shells rather than charging the same buffers again.
    assert offenders["sparse_alias"]["bytes"] < 512
    assert offenders["dense_alias"]["bytes"] < 512
    assert metrics["anndata_total_bytes"] == offenders["adata"]["bytes"]


@pytest.mark.asyncio
async def test_runtime_metrics_project_backed_anndata_materialization_peak(tmp_path):
    ad = pytest.importorskip("anndata")
    np = pytest.importorskip("numpy")
    from backend.utils.sandbox_parts.state_tracking import StateTrackingMixin

    source = ad.AnnData(np.ones((128, 32), dtype=np.float32))
    source_path = tmp_path / "backed.h5ad"
    source.write_h5ad(source_path)
    backed = ad.read_h5ad(source_path, backed="r")
    namespace = {
        "backed_adata": backed,
        "backed_alias": backed,
    }

    class _KernelProbe:
        async def _execute_unlocked(self, code, **_kwargs):
            output = io.StringIO()
            with redirect_stdout(output):
                exec(code, namespace)
            return SimpleNamespace(success=True, output=output.getvalue())

    try:
        metrics = await StateTrackingMixin._runtime_resource_metrics(_KernelProbe())
    finally:
        backed.file.close()

    assert metrics["backed_anndata_total_bytes"] > 0
    assert metrics["projected_materialized_rss_bytes"] == (
        metrics["rss_bytes"] + metrics["backed_anndata_total_bytes"]
    )


@pytest.mark.asyncio
async def test_projected_materialization_peak_uses_rss_budget_before_serialization(
    tmp_path,
):
    from backend.utils.runtime_budget import RuntimeBudgetExceeded

    budgets = {
        "max_anndata_total_bytes": 1024,
        "max_rss_bytes": 60,
    }

    class _Notebook:
        session_path = tmp_path / "projected-rss.pkl"
        state_path = tmp_path / "projected-rss.json"
        state_dir = tmp_path
        runtime_persistence = {"commit_source": None, "saved_revision": 0}

        def __init__(self):
            self.updates = []

        def has_session_dump(self):
            return False

        def update_runtime_persistence(self, status, **kwargs):
            self.updates.append((status, kwargs))

    class _Sandbox:
        session_id = "projected-rss"
        runtime_revision = 1
        saved_runtime_revision = 0
        runtime_budgets = budgets
        runtime_dependency_tracker = None
        _expected_lifecycle_generation = 0
        _session_restore_failed = False
        _exec_lock = asyncio.Lock()
        notebook_state = _Notebook()
        serialization_started = False

        async def _runtime_resource_metrics(self):
            return {
                "namespace_count": 1,
                "largest_object_bytes": 1,
                "anndata_total_bytes": 1,
                "rss_bytes": 50,
                "backed_anndata_total_bytes": 20,
                "projected_materialized_rss_bytes": 70,
                "top_offenders": [],
            }

        async def _snapshot_user_globals(self):
            self.serialization_started = True
            return set()

        def _prune_legacy_unobserved_required(self, _live_names):
            return set()

        def get_runtime_dependency_names(self):
            return set()

        async def _execute_unlocked(self, *_args, **_kwargs):
            self.serialization_started = True
            return SimpleNamespace(success=False, output="", error="unexpected")

    sandbox = _Sandbox()
    with pytest.raises(RuntimeBudgetExceeded) as blocked:
        await LocalJupyterSandbox.save_session(sandbox)

    assert sandbox.serialization_started is False
    assert [item.to_dict() for item in blocked.value.violations] == [{
        "resource": "projected_materialized_rss_bytes",
        "observed": 70,
        "limit": 60,
    }]
    status, failure = sandbox.notebook_state.updates[-1]
    assert status == "failed"
    assert failure["budget_violations"] == [{
        "resource": "projected_materialized_rss_bytes",
        "observed": 70,
        "limit": 60,
    }]


def test_effective_runtime_budgets_clamp_to_container_capacity():
    from backend.utils.runtime_budget import resolve_effective_runtime_budgets

    gib = 1024 ** 3
    effective = resolve_effective_runtime_budgets(
        {
            "max_anndata_total_bytes": 128 * gib,
            "max_rss_bytes": 192 * gib,
        },
        memory_capacity_bytes=8 * gib,
        capacity_source="test_cgroup",
    )

    assert effective["max_anndata_total_bytes"] == int(8 * gib * 0.45)
    assert effective["max_rss_bytes"] == int(8 * gib * 0.60)
    assert "max_single_object_bytes" not in effective
    assert "max_pkl_bytes" not in effective
    assert effective["effective_limit_basis"] == "test_cgroup"


def test_effective_runtime_budgets_default_to_capacity_fractions():
    from backend.utils.runtime_budget import resolve_effective_runtime_budgets

    gib = 1024 ** 3
    effective = resolve_effective_runtime_budgets(
        {
            "max_anndata_total_bytes": None,
            "max_rss_bytes": None,
        },
        memory_capacity_bytes=2 * 1024 * gib,
        capacity_source="test_cgroup",
    )

    assert effective["max_anndata_total_bytes"] == int(2 * 1024 * gib * 0.45)
    assert effective["max_rss_bytes"] == int(2 * 1024 * gib * 0.60)
    assert "max_single_object_bytes" not in effective
    assert "max_pkl_bytes" not in effective
    assert "max_namespace_count" not in effective
    assert "max_save_seconds" not in effective


def test_runtime_budget_environment_supports_numeric_caps_and_auto():
    from backend.utils.runtime_budget import apply_runtime_budget_environment

    gib = 1024 ** 3
    configured = {
        "max_anndata_total_bytes": None,
        "max_rss_bytes": None,
    }
    resolved = apply_runtime_budget_environment(
        configured,
        environ={
            "SCAI_RUNTIME_MAX_SINGLE_OBJECT_GB": "32.5",
            "SCAI_RUNTIME_MAX_ANNDATA_TOTAL_GB": "48",
            "SCAI_RUNTIME_MAX_RSS_GB": "64",
            "SCAI_RUNTIME_MAX_PKL_GB": "1",
            "SCAI_RUNTIME_MAX_NAMESPACE_COUNT": "3000",
            "SCAI_RUNTIME_MAX_SAVE_SECONDS": "900.5",
        },
    )

    assert resolved == {
        "max_anndata_total_bytes": 48 * gib,
        "max_rss_bytes": 64 * gib,
    }


def test_runtime_budget_environment_ignores_invalid_values(caplog):
    from backend.utils.runtime_budget import apply_runtime_budget_environment

    configured = {
        "max_anndata_total_bytes": None,
        "max_rss_bytes": None,
    }
    resolved = apply_runtime_budget_environment(
        configured,
        environ={
            "SCAI_RUNTIME_MAX_RSS_GB": "not-a-size",
        },
    )

    assert resolved == configured
    assert "SCAI_RUNTIME_MAX_RSS_GB" in caplog.text


@pytest.mark.parametrize(
    ("metric", "budget"),
    [
        ("pkl_bytes", "max_pkl_bytes"),
        ("namespace_count", "max_namespace_count"),
        ("largest_object_bytes", "max_single_object_bytes"),
        ("save_seconds", "max_save_seconds"),
    ],
)
def test_runtime_budget_does_not_gate_retired_resources(metric, budget):
    from backend.utils.runtime_budget import evaluate_runtime_budget

    violations = evaluate_runtime_budget(
        {metric: 1024 ** 4},
        {budget: 1},
    )

    assert violations == []


@pytest.mark.asyncio
async def test_preflight_budget_blocks_before_kernel_serialization(tmp_path):
    from backend.utils.runtime_budget import (
        RuntimeBudgetExceeded,
        resolve_effective_runtime_budgets,
    )

    gib = 1024 ** 3
    budgets = resolve_effective_runtime_budgets(
        {
            "max_anndata_total_bytes": 12 * gib,
            "max_rss_bytes": 16 * gib,
        },
        memory_capacity_bytes=4 * gib,
        capacity_source="test_cgroup",
    )

    class _Notebook:
        session_path = tmp_path / "preflight_session.pkl"
        runtime_persistence = {
            "commit_source": None,
            "saved_revision": 0,
        }

        def __init__(self):
            self.updates = []

        def has_session_dump(self):
            return False

        def update_runtime_persistence(self, status, **kwargs):
            self.updates.append((status, kwargs))

    class _Sandbox:
        session_id = "preflight"
        runtime_revision = 1
        saved_runtime_revision = 0
        runtime_budgets = budgets
        _expected_lifecycle_generation = 0
        _session_restore_failed = False
        _exec_lock = asyncio.Lock()
        notebook_state = _Notebook()
        serialization_started = False

        async def _runtime_resource_metrics(self):
            return {
                "namespace_count": 20,
                "largest_object_bytes": 512 * 1024 ** 2,
                "anndata_total_bytes": 1024 ** 3,
                "rss_bytes": budgets["max_rss_bytes"] + 1,
                "backed_anndata_total_bytes": 0,
                "projected_materialized_rss_bytes": budgets["max_rss_bytes"] + 1,
                "top_offenders": [{
                    "name": "adata", "type": "AnnData", "bytes": 1024 ** 3,
                }],
            }

        async def _snapshot_user_globals(self):
            self.serialization_started = True
            return set()

    sandbox = _Sandbox()
    with pytest.raises(RuntimeBudgetExceeded) as blocked:
        await LocalJupyterSandbox.save_session(sandbox)

    assert sandbox.serialization_started is False
    assert not sandbox.notebook_state.session_path.exists()
    status, failure = sandbox.notebook_state.updates[-1]
    assert status == "failed"
    assert failure["budget_violations"] == [
        {
            "resource": "rss_bytes",
            "observed": budgets["max_rss_bytes"] + 1,
            "limit": budgets["max_rss_bytes"],
        },
        {
            "resource": "projected_materialized_rss_bytes",
            "observed": budgets["max_rss_bytes"] + 1,
            "limit": budgets["max_rss_bytes"],
        },
    ]
    assert blocked.value.violations[0].limit == budgets["max_rss_bytes"]


def test_metadata_never_reports_stale_pkl_as_current(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "backend.utils.sandbox_parts.shared._integrity_dir",
        lambda _sid: tmp_path / "integrity",
    )
    state = NotebookState("s", str(tmp_path))
    state.session_path.write_bytes(b"old-generation")
    state.update_runtime_persistence(
        "live_dirty",
        current_revision=3,
        saved_revision=2,
    )

    payload = json.loads(state.state_path.read_text())
    assert payload["has_session"] is False
    assert payload["runtime_persistence"]["status"] == "live_dirty"
    assert payload["runtime_persistence"]["current_revision"] == 3
    assert payload["runtime_persistence"]["saved_revision"] == 2


def test_legacy_has_session_and_valid_hmac_do_not_imply_current(monkeypatch, tmp_path):
    integrity = tmp_path / "integrity"
    monkeypatch.setattr(
        "backend.utils.sandbox_parts.shared._integrity_dir",
        lambda _sid: integrity,
    )
    from backend.utils.sandbox import _write_pkl_signature

    pkl = tmp_path / "legacy_session.pkl"
    pkl.write_bytes(b"valid-but-unversioned")
    _write_pkl_signature(pkl, integrity)
    (tmp_path / "legacy_state.json").write_text(json.dumps({
        "session_id": "legacy",
        "has_session": True,
        "execution_count": 8,
    }))

    state = NotebookState("legacy", str(tmp_path))

    assert state.has_session_dump() is False
    assert state.runtime_persistence["status"] == "stale"
    assert state.runtime_persistence["commit_source"] is None


def test_regular_runtime_commit_rolls_back_pkl_when_metadata_fails(tmp_path):
    live = tmp_path / "s_session.pkl"
    staged = tmp_path / "s_session.pkl.tmp"
    integrity = tmp_path / "integrity"
    live.write_bytes(b"old")
    staged.write_bytes(b"new")

    # Establish a valid old pair so rollback can prove both bytes and identity.
    from backend.utils.sandbox import _write_pkl_signature

    _write_pkl_signature(live, integrity)

    with pytest.raises(RuntimeError, match="metadata"):
        _commit_pkl_transaction(
            staged,
            live,
            integrity,
            lambda: (_ for _ in ()).throw(RuntimeError("metadata failed")),
        )

    assert live.read_bytes() == b"old"
    assert _verify_pkl_integrity(live, integrity) is True




def _paired_fixture(tmp_path):
    """A dump about to be committed, and the three files it is paired with."""
    import sqlite3

    states = tmp_path / "states"
    states.mkdir()
    live = states / "s_session.pkl"
    staged = states / "s_session.pkl.tmp"
    staged.write_bytes(b"new-generation")
    state_json = states / "s_state.json"
    state_json.write_text(json.dumps({"session_id": "s", "state_version": 12}))
    checkpoint = states / "checkpoints.sqlite"
    with sqlite3.connect(checkpoint) as conn:
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT, blob TEXT)")
        conn.execute("INSERT INTO checkpoints VALUES ('s', 'position-7')")
    executor = states / "s_bioinfo_checkpoint.json"
    executor.write_text(json.dumps({"step_number": 40, "round_number": 2}))
    return states, live, staged, state_json, checkpoint, executor


def test_a_commit_captures_the_generation_it_can_be_continued_from(tmp_path):
    """Spec B2-1: a dump alone is not a restore point.

    The namespace is only continuable beside the workflow state, the graph
    position and the executor's own step position. Committing the dump without
    them is what leaves a healthy generation unusable after a crash, because
    the live copies of all three have moved on by then.
    """
    states, live, staged, state_json, checkpoint, executor = _paired_fixture(tmp_path)
    integrity = tmp_path / "integrity"

    _commit_pkl_transaction(
        staged,
        live,
        integrity,
        lambda: None,
        target_revision=7,
        generation_root=states / "generation",
        paired_sources={
            "state": state_json,
            "checkpoint": checkpoint,
            "executor_checkpoint": executor,
        },
    )

    pair = states / "generation" / "7"
    assert json.loads((pair / "state.json").read_text())["state_version"] == 12
    assert json.loads((pair / "executor_checkpoint.json").read_text())["step_number"] == 40

    import sqlite3

    with sqlite3.connect(pair / "checkpoints.sqlite") as conn:
        assert conn.execute("SELECT blob FROM checkpoints").fetchone()[0] == "position-7"


def test_an_absent_executor_checkpoint_is_recorded_as_absent(tmp_path):
    """A round boundary reached after the executor finished has none.

    Absent must not read as a component that failed to copy: the first would
    make every round-boundary save fail, and the second has to keep failing the
    commit.
    """
    states, live, staged, state_json, checkpoint, executor = _paired_fixture(tmp_path)
    executor.unlink()
    integrity = tmp_path / "integrity"

    _commit_pkl_transaction(
        staged,
        live,
        integrity,
        lambda: None,
        target_revision=3,
        generation_root=states / "generation",
        paired_sources={
            "state": state_json,
            "checkpoint": checkpoint,
            "executor_checkpoint": executor,
        },
    )

    pair = states / "generation" / "3"
    assert (pair / "state.json").is_file()
    assert not (pair / "executor_checkpoint.json").exists()


def test_a_failed_commit_leaves_no_half_written_pair(tmp_path):
    """The pair rolls back with the generation it describes.

    A pair that outlived its rollback would name a dump that is no longer on
    disk, which is worse than no pair at all -- recovery would offer it.
    """
    states, live, staged, state_json, checkpoint, executor = _paired_fixture(tmp_path)
    integrity = tmp_path / "integrity"
    live.write_bytes(b"old")
    from backend.utils.sandbox import _write_pkl_signature

    _write_pkl_signature(live, integrity)

    with pytest.raises(RuntimeError, match="metadata"):
        _commit_pkl_transaction(
            staged,
            live,
            integrity,
            lambda: (_ for _ in ()).throw(RuntimeError("metadata failed")),
            target_revision=9,
            generation_root=states / "generation",
            paired_sources={
                "state": state_json,
                "checkpoint": checkpoint,
                "executor_checkpoint": executor,
            },
        )

    assert live.read_bytes() == b"old"
    assert not (states / "generation" / "9" / "state.json").exists()
    assert not (states / "generation" / "9" / "checkpoints.sqlite").exists()


def test_only_the_newest_pair_is_kept(tmp_path):
    """Retention is one (B2-1). Never zero: the older pair goes only after the
    newer one is durable."""
    states, live, staged, state_json, checkpoint, executor = _paired_fixture(tmp_path)
    integrity = tmp_path / "integrity"

    _commit_pkl_transaction(
        staged, live, integrity, lambda: None,
        target_revision=4,
        generation_root=states / "generation",
        paired_sources={
            "state": state_json, "checkpoint": checkpoint,
            "executor_checkpoint": executor,
        },
    )
    assert (states / "generation" / "4").is_dir()

    staged.write_bytes(b"newer-generation")
    _commit_pkl_transaction(
        staged, live, integrity, lambda: None,
        target_revision=5,
        generation_root=states / "generation",
        paired_sources={
            "state": state_json, "checkpoint": checkpoint,
            "executor_checkpoint": executor,
        },
    )

    assert (states / "generation" / "5").is_dir()
    assert not (states / "generation" / "4").exists()


class _SimulatedProcessCrash(BaseException):
    pass


@pytest.mark.parametrize(
    ("crash_point", "expected"),
    [
        ("journal_applying", "old"),
        ("pkl_replaced", "old"),
        ("sig_replaced", "old"),
        ("metadata_replaced", "old"),
        ("runtime_manifest_replaced", "old"),
        ("commit_marker", "new"),
        ("cleanup_backup_0", "new"),
    ],
)
def test_regular_save_recovers_every_crash_boundary(
    monkeypatch,
    tmp_path,
    crash_point,
    expected,
):
    from backend.utils.runtime_manifest import (
        build_runtime_manifest,
        read_runtime_manifest,
        write_runtime_manifest,
    )
    from backend.utils.sandbox import _write_pkl_signature

    states = tmp_path / "states"
    integrity = tmp_path / "integrity"
    states.mkdir()
    live = states / "crash_session.pkl"
    staged = states / "crash_session.pkl.tmp"
    live.write_bytes(b"old-generation")
    staged.write_bytes(b"new-generation")
    _write_pkl_signature(live, integrity)
    monkeypatch.setattr(
        "backend.utils.sandbox_parts.shared._integrity_dir",
        lambda _sid: integrity,
    )
    notebook = NotebookState("crash", str(states))
    old_digest = _sha256_file(live)
    new_digest = _sha256_file(staged)
    notebook.update_runtime_persistence(
        "current",
        current_revision=1,
        saved_revision=1,
        commit_source="regular_save",
        runtime_digest=old_digest,
    )
    # The generation manifest is a transaction component, so the matrix has to
    # prove the same thing about it as about the dump: whichever generation
    # survives a crash, the manifest on disk describes *that* dump. A manifest
    # that outlived a rolled-back dump would declare names the restored bytes
    # do not contain, and integrity would then report corruption on an intact
    # generation.
    runtime_manifest = states / "crash_session.runtime.json"
    write_runtime_manifest(
        runtime_manifest,
        build_runtime_manifest(
            revision=1,
            state_revision=1,
            pkl_digest=old_digest,
            saved_names=["adata"],
            skipped_names=[],
            serializer_version="dill-zstd-1",
        ),
    )

    def commit_metadata():
        notebook.update_runtime_persistence(
            "current",
            current_revision=2,
            saved_revision=2,
            commit_source="regular_save",
            runtime_digest=new_digest,
        )

    def crash_at(point):
        if point == crash_point:
            raise _SimulatedProcessCrash(point)

    with pytest.raises(_SimulatedProcessCrash):
        _commit_pkl_transaction(
            staged,
            live,
            integrity,
            commit_metadata,
            metadata_path=notebook.state_path,
            runtime_manifest_path=runtime_manifest,
            runtime_manifest={
                "state_revision": 2,
                "saved_names": ["adata", "markers"],
                "skipped_names": [],
                "serializer_version": "dill-zstd-1",
            },
            target_revision=2,
            authoritative_pkl_names={"adata"},
            fault_hook=crash_at,
        )

    outcome = recover_regular_save_transaction(
        live,
        integrity,
        metadata_path=notebook.state_path,
        runtime_manifest_path=runtime_manifest,
        session_id="",
    )
    recovered_manifest = read_runtime_manifest(runtime_manifest)
    payload = json.loads(notebook.state_path.read_text())
    persistence = payload["runtime_persistence"]
    if expected == "old":
        assert outcome == "rollback"
        assert live.read_bytes() == b"old-generation"
        assert persistence["saved_revision"] == 1
        assert persistence["runtime_digest"] == old_digest
        assert recovered_manifest["revision"] == 1
        assert recovered_manifest["pkl_sha256"] == old_digest
        assert recovered_manifest["saved_names"] == ["adata"]
    else:
        assert outcome == "forward"
        assert live.read_bytes() == b"new-generation"
        assert persistence["saved_revision"] == 2
        assert persistence["runtime_digest"] == new_digest
        assert recovered_manifest["revision"] == 2
        assert recovered_manifest["pkl_sha256"] == new_digest
        assert recovered_manifest["saved_names"] == ["adata", "markers"]
    assert _verify_pkl_integrity(live, integrity)
    assert not live.with_name(f"{live.name}.regular-save.json").exists()
    assert not live.with_name(f"{live.name}.regular-save.tmp").exists()
    assert not live.with_name(f"{live.name}.regular-save.bak").exists()
    assert not _sig_path(live, integrity).with_name(
        f"{_sig_path(live, integrity).name}.regular-save.tmp"
    ).exists()


def test_regular_save_backup_copy_uses_fixed_size_reads(tmp_path, monkeypatch):
    from backend.utils.sandbox_parts import shared

    copy_chunk = shared._REGULAR_SAVE_COPY_CHUNK_BYTES
    source = tmp_path / "large-source.bin"
    target = tmp_path / "large-target.bin"
    source.write_bytes(b"x" * (copy_chunk * 2 + 17))
    real_open = builtins.open
    reads = []

    class _GuardedSource:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def read(self, size=-1):
            assert size > 0
            assert size <= copy_chunk
            reads.append(size)
            return self._handle.read(size)

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def guarded_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if Path(path) == source and "r" in mode and "b" in mode:
            return _GuardedSource(handle)
        return handle

    monkeypatch.setattr(builtins, "open", guarded_open)
    shared._copy_durable(source, target)
    assert reads
    assert max(reads) <= copy_chunk
    assert target.read_bytes() == source.read_bytes()


def test_mid_backup_copy_failure_cleans_unjournaled_products_and_preserves_committed_generation(
    tmp_path, monkeypatch,
):
    from backend.utils.sandbox import _sig_path, _write_pkl_signature
    from backend.utils.sandbox_parts import shared

    states = tmp_path / "states"
    integrity = tmp_path / "integrity"
    states.mkdir()
    live = states / "mid-copy.pkl"
    staged = states / "mid-copy.pkl.tmp"
    metadata = states / "mid-copy.json"
    live.write_bytes(b"old-generation")
    staged.write_bytes(b"new-generation")
    metadata.write_bytes(b"old-metadata")
    _write_pkl_signature(live, integrity)
    old = {
        "pkl": live.read_bytes(),
        "sig": _sig_path(live, integrity).read_bytes(),
        "metadata": metadata.read_bytes(),
    }
    foreign_temp = states / ".mid-copy.pkl.regular-save.bak.foreign.tmp"
    foreign_temp.write_bytes(b"foreign-temp-sentinel")
    calls = 0
    current_temp = None
    partial_write_calls = 0
    original_copy = shared._copy_durable
    real_fdopen = shared.os.fdopen

    def fail_on_later_backup(source, target):
        nonlocal calls, current_temp, partial_write_calls
        calls += 1
        if calls != 2:
            return original_copy(source, target)

        def guarded_fdopen(fd, mode="r", *args, **kwargs):
            nonlocal current_temp
            handle = real_fdopen(fd, mode, *args, **kwargs)
            if "wb" not in mode:
                return handle
            current_temp = Path(shared.os.readlink(f"/proc/self/fd/{fd}"))

            class _PartialTarget:
                def __enter__(self):
                    handle.__enter__()
                    return self

                def __exit__(self, *exit_args):
                    return handle.__exit__(*exit_args)

                def write(self, data):
                    nonlocal partial_write_calls
                    partial_write_calls += 1
                    if partial_write_calls >= 2:
                        raise OSError("backup copy failed after partial write")
                    return handle.write(data)

                def __getattr__(self, name):
                    return getattr(handle, name)

            return _PartialTarget()

        monkeypatch.setattr(shared.os, "fdopen", guarded_fdopen)
        try:
            return original_copy(source, target)
        finally:
            monkeypatch.setattr(shared.os, "fdopen", real_fdopen)

    monkeypatch.setattr(shared, "_REGULAR_SAVE_COPY_CHUNK_BYTES", 16)
    monkeypatch.setattr(shared, "_copy_durable", fail_on_later_backup)
    with pytest.raises(OSError, match="backup copy failed after partial write"):
        _commit_pkl_transaction(
            staged,
            live,
            integrity,
            lambda: None,
            metadata_path=metadata,
            target_revision=2,
        )

    assert calls >= 2
    assert partial_write_calls >= 2
    assert current_temp is not None
    assert not current_temp.exists()
    assert live.read_bytes() == old["pkl"]
    assert _sig_path(live, integrity).read_bytes() == old["sig"]
    assert metadata.read_bytes() == old["metadata"]
    assert foreign_temp.read_bytes() == b"foreign-temp-sentinel"
    assert not states.joinpath("mid-copy.pkl.regular-save.json").exists()
    assert not staged.exists()
    assert not states.joinpath("mid-copy.json.regular-save.bak").exists()
    assert not _sig_path(live, integrity).with_name(
        f"{_sig_path(live, integrity).name}.regular-save.tmp"
    ).exists()


def test_partial_staged_signature_write_is_removed_before_journal(
    tmp_path, monkeypatch,
):
    from backend.utils.sandbox import _sig_path, _write_pkl_signature

    states = tmp_path / "states"
    integrity = tmp_path / "integrity"
    states.mkdir()
    live = states / "partial-signature.pkl"
    staged = states / "partial-signature.pkl.tmp"
    metadata = states / "partial-signature.json"
    live.write_bytes(b"old-generation")
    staged.write_bytes(b"new-generation")
    metadata.write_bytes(b"old-metadata")
    _write_pkl_signature(live, integrity)
    old = {
        "pkl": live.read_bytes(),
        "sig": _sig_path(live, integrity).read_bytes(),
        "metadata": metadata.read_bytes(),
    }
    staged_sig = _sig_path(live, integrity).with_name(
        f"{_sig_path(live, integrity).name}.regular-save.tmp"
    )
    real_write_text = Path.write_text

    def partial_signature_write(path, data, *args, **kwargs):
        if path == staged_sig:
            path.parent.mkdir(parents=True, exist_ok=True)
            with builtins.open(path, "w", encoding="utf-8") as handle:
                handle.write(data[:3])
            raise OSError("partial staged signature write")
        return real_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", partial_signature_write)
    with pytest.raises(OSError, match="partial staged signature write"):
        _commit_pkl_transaction(
            staged,
            live,
            integrity,
            lambda: None,
            metadata_path=metadata,
            target_revision=2,
        )

    assert live.read_bytes() == old["pkl"]
    assert _sig_path(live, integrity).read_bytes() == old["sig"]
    assert metadata.read_bytes() == old["metadata"]
    assert not states.joinpath("partial-signature.pkl.regular-save.json").exists()
    assert not staged.exists()
    assert not staged_sig.exists()
    assert not list(states.glob("*.regular-save.bak"))
    assert not list(integrity.glob("*.regular-save.bak"))


@pytest.mark.parametrize(
    "corrupt_component",
    ["pkl", "sig", "metadata"],
)
def test_committed_regular_save_authenticates_every_component_or_rolls_back(
    monkeypatch, tmp_path, corrupt_component,
):
    """COMMITTED is not sufficient when any target byte no longer verifies."""
    from backend.utils.sandbox import _sig_path, _write_pkl_signature

    states = tmp_path / "states"
    integrity = tmp_path / "integrity"
    states.mkdir()
    live = states / "complete_session.pkl"
    staged = states / "complete_session.pkl.tmp"
    live.write_bytes(b"old-pkl")
    staged.write_bytes(b"new-pkl")
    _write_pkl_signature(live, integrity)
    notebook = NotebookState("complete", str(states))
    old_digest = _sha256_file(live)
    notebook.update_runtime_persistence(
        "current",
        current_revision=1,
        saved_revision=1,
        commit_source="regular_save",
        runtime_digest=old_digest,
    )
    old_bytes = {
        "pkl": live.read_bytes(),
        "sig": _sig_path(live, integrity).read_bytes(),
        "metadata": notebook.state_path.read_bytes(),
    }
    new_digest = _sha256_file(staged)

    def commit_metadata():
        notebook.update_runtime_persistence(
            "current",
            current_revision=2,
            saved_revision=2,
            commit_source="regular_save",
            runtime_digest=new_digest,
        )

    def crash_after_commit(point):
        if point == "commit_marker":
            raise _SimulatedProcessCrash(point)

    with pytest.raises(_SimulatedProcessCrash):
        _commit_pkl_transaction(
            staged,
            live,
            integrity,
            commit_metadata,
            metadata_path=notebook.state_path,
            target_revision=2,
            authoritative_pkl_names={"score"},
            fault_hook=crash_after_commit,
        )

    target_path = {
        "pkl": live,
        "sig": _sig_path(live, integrity),
        "metadata": notebook.state_path,
    }[corrupt_component]
    target_path.write_bytes(b"corrupt-target")

    outcome = recover_regular_save_transaction(
        live,
        integrity,
        metadata_path=notebook.state_path,
    )

    assert outcome == "rollback"
    assert live.read_bytes() == old_bytes["pkl"]
    assert _sig_path(live, integrity).read_bytes() == old_bytes["sig"]
    assert notebook.state_path.read_bytes() == old_bytes["metadata"]
    assert _verify_pkl_integrity(live, integrity)




def test_notebook_runtime_metadata_redacts_raw_fields_before_persisting(tmp_path):
    secret = "synthetic-notebook-runtime-secret"
    notebook = NotebookState("raw-redaction", str(tmp_path))

    persistence = notebook.update_runtime_persistence(
        "failed",
        current_revision=2,
        saved_revision=1,
        failed_variables=[f"api_key={secret}"],
        error_summary=(
            f"Bearer {secret} at /home/private-user/project/save.py"
        ),
        requires_recompute=True,
        recompute_variables=["/workspace/private-project/adata"],
        budget_violations=[{
            "resource": f"rss api_key={secret}",
            "observed": 10,
            "limit": 8,
        }],
        top_offenders=[{
            "name": "/data/sessions/private-session/states/adata",
            "type": f"AnnData Bearer {secret}",
            "bytes": 9,
        }],
        persist=False,
    )

    rendered = str(persistence)
    assert secret not in rendered
    assert "private-user" not in rendered
    assert "private-project" not in rendered
    assert "private-session" not in rendered
    assert "******" in rendered


def test_runtime_persistence_error_summary_retains_final_exception(tmp_path):
    notebook = NotebookState("runtime-tail", str(tmp_path))

    persistence = notebook.update_runtime_persistence(
        "failed",
        current_revision=2,
        saved_revision=1,
        error_summary=_long_runtime_diagnostic_traceback(),
        persist=False,
    )

    assert len(persistence["error_summary"]) <= 500
    assert "TypeError: boolean value of NA is ambiguous" in persistence["error_summary"]


def test_runtime_persistence_identifier_fields_remain_head_bounded(tmp_path):
    notebook = NotebookState("runtime-identifiers", str(tmp_path))
    identifier = "identifier-head-" + ("x" * 256) + "-identifier-tail"

    persistence = notebook.update_runtime_persistence(
        "failed",
        current_revision=2,
        saved_revision=1,
        failed_variables=[identifier],
        recompute_variables=[identifier],
        budget_violations=[{"resource": identifier, "observed": 2, "limit": 1}],
        top_offenders=[{"name": identifier, "type": identifier, "bytes": 2}],
        persist=False,
    )

    assert persistence["failed_variables"][0].startswith("identifier-head-")
    assert persistence["recompute_variables"][0].startswith("identifier-head-")
    assert persistence["budget_violations"][0]["resource"].startswith("identifier-head-")
    assert persistence["top_offenders"][0]["name"].startswith("identifier-head-")
    assert persistence["top_offenders"][0]["type"].startswith("identifier-head-")
    for value in (
        persistence["failed_variables"][0],
        persistence["recompute_variables"][0],
        persistence["budget_violations"][0]["resource"],
        persistence["top_offenders"][0]["name"],
        persistence["top_offenders"][0]["type"],
    ):
        assert len(value) <= 128
        assert "identifier-tail" not in value


def test_regular_save_fails_closed_before_partial_rollback_when_backup_corrupt(
    tmp_path,
):
    from backend.utils.sandbox import _write_pkl_signature

    states = tmp_path / "states"
    integrity = tmp_path / "integrity"
    states.mkdir()
    live = states / "closed_session.pkl"
    staged = states / "closed_session.pkl.tmp"
    live.write_bytes(b"old")
    staged.write_bytes(b"new")
    _write_pkl_signature(live, integrity)
    notebook = NotebookState("closed", str(states))
    notebook.update_runtime_persistence(
        "current",
        current_revision=1,
        saved_revision=1,
        commit_source="regular_save",
        runtime_digest=_sha256_file(live),
    )
    new_digest = _sha256_file(staged)

    def commit_metadata():
        notebook.update_runtime_persistence(
            "current",
            current_revision=2,
            saved_revision=2,
            commit_source="regular_save",
            runtime_digest=new_digest,
        )

    def crash_mid_apply(point):
        if point == "metadata_replaced":
            raise _SimulatedProcessCrash(point)

    with pytest.raises(_SimulatedProcessCrash):
        _commit_pkl_transaction(
            staged,
            live,
            integrity,
            commit_metadata,
            metadata_path=notebook.state_path,
            target_revision=2,
            authoritative_pkl_names={"adata"},
            fault_hook=crash_mid_apply,
        )

    metadata_backup = notebook.state_path.with_name(
        f"{notebook.state_path.name}.regular-save.bak"
    )
    metadata_backup.write_bytes(b"tampered-backup")
    target_before = live.read_bytes()
    journal = live.with_name(f"{live.name}.regular-save.json")

    with pytest.raises(RuntimeError, match="backup digest mismatch"):
        recover_regular_save_transaction(
            live,
            integrity,
            metadata_path=notebook.state_path,
        )

    # Preflight authenticated every backup before touching any target.
    assert live.read_bytes() == target_before
    assert journal.exists()


def test_record_runtime_restored_preserves_commit_audit(monkeypatch, tmp_path):
    integrity = tmp_path / "integrity"
    monkeypatch.setattr(
        "backend.utils.sandbox_parts.shared._integrity_dir",
        lambda _sid: integrity,
    )
    from backend.utils.sandbox import _write_pkl_signature

    notebook = NotebookState("restore-audit", str(tmp_path))
    notebook.session_path.write_bytes(b"generation")
    _write_pkl_signature(notebook.session_path, integrity)
    digest = _sha256_file(notebook.session_path)
    notebook.update_runtime_persistence(
        "current",
        current_revision=4,
        saved_revision=4,
        commit_source="regular_save",
        runtime_digest=digest,
    )
    saved_at = notebook.runtime_persistence["last_saved_at"]

    notebook.record_runtime_restored(
        current_revision=4,
        saved_revision=4,
        commit_source="regular_save",
    )

    assert notebook.runtime_persistence["last_saved_at"] == saved_at
    assert notebook.runtime_persistence["last_restored_at"] is not None
    assert notebook.runtime_persistence["commit_source"] == "regular_save"


def test_failed_new_attempt_keeps_last_committed_generation_recoverable(
    monkeypatch, tmp_path,
):
    integrity = tmp_path / "integrity"
    monkeypatch.setattr(
        "backend.utils.sandbox_parts.shared._integrity_dir",
        lambda _sid: integrity,
    )
    from backend.utils.sandbox import _write_pkl_signature

    notebook = NotebookState("attempt-audit", str(tmp_path))
    notebook.session_path.write_bytes(b"verified-generation")
    _write_pkl_signature(notebook.session_path, integrity)
    digest = _sha256_file(notebook.session_path)
    notebook.update_runtime_persistence(
        "current",
        current_revision=7,
        saved_revision=7,
        commit_source="regular_save",
        runtime_digest=digest,
    )

    notebook.update_runtime_persistence(
        "failed",
        current_revision=9,
        saved_revision=7,
        failed_variables=["unpicklable_model"],
        error_summary="serialization failed",
    )

    persistence = notebook.runtime_persistence
    assert persistence["status"] == "failed"
    assert persistence["current_revision"] == 9
    assert persistence["last_attempt_status"] == "failed"
    assert persistence["committed_revision"] == 7
    assert persistence["committed_digest"] == digest
    assert notebook.has_session_dump() is True
    assert json.loads(notebook.state_path.read_text())["has_session"] is True


def test_restore_of_older_committed_generation_is_explicitly_degraded(
    monkeypatch, tmp_path,
):
    integrity = tmp_path / "integrity"
    monkeypatch.setattr(
        "backend.utils.sandbox_parts.shared._integrity_dir",
        lambda _sid: integrity,
    )
    from backend.utils.sandbox import _write_pkl_signature

    notebook = NotebookState("degraded-audit", str(tmp_path))
    notebook.session_path.write_bytes(b"verified-generation")
    _write_pkl_signature(notebook.session_path, integrity)
    digest = _sha256_file(notebook.session_path)
    notebook.update_runtime_persistence(
        "current",
        current_revision=7,
        saved_revision=7,
        commit_source="regular_save",
        runtime_digest=digest,
    )
    notebook.update_runtime_persistence(
        "failed",
        current_revision=9,
        saved_revision=7,
        error_summary="serialization failed",
    )

    notebook.record_runtime_restored(
        current_revision=7,
        saved_revision=7,
        commit_source="regular_save",
        degraded_recovery=True,
        attempted_revision=9,
    )

    persistence = notebook.runtime_persistence
    assert persistence["status"] == "degraded"
    assert persistence["current_revision"] == 7
    assert persistence["saved_revision"] == 7
    assert persistence["degraded_recovery"] is True
    assert persistence["degraded_from_revision"] == 9
    assert persistence["degraded_to_revision"] == 7
    assert persistence["last_attempt_status"] == "failed"
    assert persistence["requires_recompute"] is True


def test_a_committed_generation_records_how_long_it_took(tmp_path):
    """Spec B3-2 ships this with the decision to keep N at 30.

    The duration of one save is the constraint that decision turns on -- six to
    nine minutes at the end of a long run, with the execution lock held -- and
    it existed only in container logs, which vanish on restart. Evaluating N
    had to extrapolate it from two other sessions for exactly that reason.
    """
    from backend.utils.sandbox import NotebookState

    state = NotebookState("timed", str(tmp_path))
    state.update_runtime_persistence(
        "current", current_revision=7, saved_revision=7,
        commit_source="regular_save", runtime_digest="c" * 64,
        save_seconds=124.5,
    )
    assert state.runtime_persistence["last_save_seconds"] == 124.5


def test_a_failed_attempt_leaves_the_last_good_duration_alone(tmp_path):
    """It describes the last generation that reached disk, like every other
    committed_* field beside it."""
    from backend.utils.sandbox import NotebookState

    state = NotebookState("timed-failure", str(tmp_path))
    state.update_runtime_persistence(
        "current", current_revision=7, saved_revision=7,
        commit_source="regular_save", runtime_digest="c" * 64,
        save_seconds=124.5,
    )
    state.update_runtime_persistence(
        "failed", current_revision=8, saved_revision=7,
        error_summary="no space left on device",
    )
    assert state.runtime_persistence["last_save_seconds"] == 124.5


def test_the_save_path_measures_what_it_records():
    """A field nothing populates is documentation, not a measurement."""
    import pathlib as _pathlib

    repo = _pathlib.Path(__file__).resolve().parents[2]
    source = (
        repo / "backend" / "utils" / "sandbox_parts" / "runtime_save.py"
    ).read_text()
    assert "save_seconds=" in source
    assert "monotonic()" in source
