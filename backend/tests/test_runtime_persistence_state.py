import json
import io
import asyncio
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest

from backend.services.snapshot_v2.runtime_registry import RuntimeDependencyTracker
from backend.utils.sandbox import (
    NotebookState,
    LocalJupyterSandbox,
    _commit_pkl_transaction,
    _sha256_file,
    _verify_pkl_integrity,
    recover_regular_save_transaction,
)


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


def test_effective_runtime_budgets_clamp_to_container_capacity():
    from backend.utils.runtime_budget import resolve_effective_runtime_budgets

    gib = 1024 ** 3
    effective = resolve_effective_runtime_budgets(
        {
            "max_namespace_count": 5000,
            "max_single_object_bytes": 32 * gib,
            "max_anndata_total_bytes": 128 * gib,
            "max_rss_bytes": 192 * gib,
            "max_pkl_bytes": 128 * gib,
            "max_save_seconds": 900,
        },
        memory_capacity_bytes=8 * gib,
        capacity_source="test_cgroup",
    )

    assert effective["max_single_object_bytes"] == 2 * gib
    assert effective["max_anndata_total_bytes"] == int(8 * gib * 0.45)
    assert effective["max_rss_bytes"] == int(8 * gib * 0.60)
    assert effective["max_pkl_bytes"] == int(8 * gib * 0.30)
    assert effective["effective_limit_basis"] == "test_cgroup"


@pytest.mark.asyncio
async def test_preflight_budget_blocks_before_kernel_serialization(tmp_path):
    from backend.utils.runtime_budget import (
        RuntimeBudgetExceeded,
        resolve_effective_runtime_budgets,
    )

    gib = 1024 ** 3
    budgets = resolve_effective_runtime_budgets(
        {
            "max_namespace_count": 1500,
            "max_single_object_bytes": 4 * gib,
            "max_anndata_total_bytes": 12 * gib,
            "max_rss_bytes": 16 * gib,
            "max_pkl_bytes": 8 * gib,
            "max_save_seconds": 300,
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
    assert failure["budget_violations"] == [{
        "resource": "rss_bytes",
        "observed": budgets["max_rss_bytes"] + 1,
        "limit": budgets["max_rss_bytes"],
    }]
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


def test_required_dependency_poisoning_requires_explicit_lifecycle_release():
    tracker = RuntimeDependencyTracker()
    tracker.record_execution(created={"score"}, declared={"score"})

    assert tracker.record_persistence_failure({"score"}) == set()
    assert tracker.record_persistence_failure({"score"}) == set()
    assert tracker.record_persistence_failure({"score"}) == {"score"}
    with pytest.raises(ValueError, match="unsupported"):
        tracker.release({"score"}, lifecycle="best_effort", reason="drop it")

    remaining = tracker.release(
        {"score"},
        lifecycle="recomputed",
        reason="The continuation now consumes score_v2.",
    )
    assert remaining == set()
    assert tracker.to_state_dict()["release_log"][-1]["name"] == "score"


def test_same_name_explicit_recompute_clears_poison_but_keeps_dependency():
    tracker = RuntimeDependencyTracker()
    tracker.record_execution(created={"score"}, declared={"score"})
    for _ in range(3):
        tracker.record_persistence_failure({"score"})

    tracker.confirm_recomputed(
        {"score"},
        reason="Successful declared output replaced the unpersistable value.",
    )

    assert tracker.required_outputs == {"score"}
    assert tracker.record_persistence_failure({"score"}) == set()
    assert tracker.to_state_dict()["release_log"][-1] == {
        "name": "score",
        "lifecycle": "recomputed",
        "reason": "Successful declared output replaced the unpersistable value.",
        "retained": "true",
    }


@pytest.mark.asyncio
async def test_snapshot_commit_journal_is_durable_before_metadata_stamp(monkeypatch):
    from backend.services.snapshot_v2.restore_parts import coordinator as module

    events = []
    journal = {
        "transaction_id": "restore-order",
        "session_id": "session-order",
        "status": "APPLYING",
        "runtime_restore_status": "complete",
    }
    monkeypatch.setattr(module, "_read_journal", lambda *_args: dict(journal))
    monkeypatch.setattr(
        module,
        "_write_journal_hmac_signed",
        lambda _paths, _tx, payload: events.append(f"write:{payload['status']}"),
    )
    monkeypatch.setattr(
        "backend.utils.runtime_apply.commit_runtime_persistence_metadata",
        lambda *_args, **_kwargs: events.append("stamp"),
    )

    class _Fault:
        def check(self, point):
            if point == "journal_committed":
                events.append("fault")
                raise RuntimeError("crash after committed journal")

    coordinator = module.RestoreCoordinator(fault_barrier=_Fault())
    with pytest.raises(RuntimeError, match="crash"):
        await coordinator.commit(
            "restore-order",
            object(),
            verification_proof={
                "verified_live_root_hash": "root",
                "component_hashes": {},
                "verified_at": "now",
            },
        )

    assert events == ["write:COMMITTED", "fault"]


class _SimulatedProcessCrash(BaseException):
    pass


@pytest.mark.parametrize(
    ("crash_point", "expected"),
    [
        ("journal_applying", "old"),
        ("pkl_replaced", "old"),
        ("sig_replaced", "old"),
        ("manifest_replaced", "old"),
        ("metadata_replaced", "old"),
        ("commit_marker", "new"),
        ("sidecars_retired", "new"),
        ("cleanup_backup_0", "new"),
    ],
)
def test_regular_save_recovers_every_crash_boundary(
    monkeypatch,
    tmp_path,
    crash_point,
    expected,
):
    from backend.utils.files import atomic_write_json
    from backend.utils.sandbox import _write_pkl_signature

    states = tmp_path / "states"
    integrity = tmp_path / "integrity"
    states.mkdir()
    live = states / "crash_session.pkl"
    staged = states / "crash_session.pkl.tmp"
    manifest = states / "runtime_h5ad_manifest.json"
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
    atomic_write_json(manifest, {"version": 1, "entries": {"old": {}}})

    def commit_manifest():
        atomic_write_json(
            manifest,
            {
                "version": 2,
                "entries": {},
                "authoritative_pkl_names": ["adata"],
            },
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
            manifest_path=manifest,
            commit_manifest=commit_manifest,
            retire_sidecars=lambda: None,
            target_revision=2,
            authoritative_pkl_names={"adata"},
            fault_hook=crash_at,
        )

    outcome = recover_regular_save_transaction(
        live,
        integrity,
        metadata_path=notebook.state_path,
        manifest_path=manifest,
        session_id="",
    )
    payload = json.loads(notebook.state_path.read_text())
    persistence = payload["runtime_persistence"]
    if expected == "old":
        assert outcome == "rollback"
        assert live.read_bytes() == b"old-generation"
        assert persistence["saved_revision"] == 1
        assert persistence["runtime_digest"] == old_digest
        assert json.loads(manifest.read_text())["version"] == 1
    else:
        assert outcome == "forward"
        assert live.read_bytes() == b"new-generation"
        assert persistence["saved_revision"] == 2
        assert persistence["runtime_digest"] == new_digest
        assert json.loads(manifest.read_text())["version"] == 2
    assert _verify_pkl_integrity(live, integrity)
    assert not live.with_name(f"{live.name}.regular-save.json").exists()


@pytest.mark.parametrize(
    "corrupt_component",
    ["pkl", "sig", "metadata", "manifest", "h5ad"],
)
def test_committed_regular_save_authenticates_every_component_or_rolls_back(
    monkeypatch, tmp_path, corrupt_component,
):
    """COMMITTED is not sufficient when any target byte no longer verifies."""
    from backend.utils.runtime_apply import write_runtime_h5ad_manifest
    from backend.utils.sandbox import _sig_path, _write_pkl_signature

    states = tmp_path / "states"
    integrity = tmp_path / "integrity"
    states.mkdir()
    live = states / "complete_session.pkl"
    staged = states / "complete_session.pkl.tmp"
    sidecar = states / "adata.h5ad"
    manifest = states / "runtime_h5ad_manifest.json"
    live.write_bytes(b"old-pkl")
    staged.write_bytes(b"new-pkl")
    sidecar.write_bytes(b"old-h5ad")
    _write_pkl_signature(live, integrity)
    write_runtime_h5ad_manifest(
        "complete",
        {"adata": _sha256_file(sidecar)},
        authoritative_pkl_names={"score"},
        states_dir=states,
    )
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
        "manifest": manifest.read_bytes(),
        "h5ad": sidecar.read_bytes(),
    }
    new_digest = _sha256_file(staged)

    def commit_manifest():
        sidecar.write_bytes(b"new-h5ad")
        write_runtime_h5ad_manifest(
            "complete",
            {"adata": _sha256_file(sidecar)},
            authoritative_pkl_names={"score"},
            states_dir=states,
        )

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
            manifest_path=manifest,
            commit_manifest=commit_manifest,
            retire_sidecars=lambda: None,
            target_revision=2,
            authoritative_pkl_names={"score"},
            fault_hook=crash_after_commit,
        )

    target_path = {
        "pkl": live,
        "sig": _sig_path(live, integrity),
        "metadata": notebook.state_path,
        "manifest": manifest,
        "h5ad": sidecar,
    }[corrupt_component]
    target_path.write_bytes(b"corrupt-target")

    outcome = recover_regular_save_transaction(
        live,
        integrity,
        metadata_path=notebook.state_path,
        manifest_path=manifest,
    )

    assert outcome == "rollback"
    assert live.read_bytes() == old_bytes["pkl"]
    assert _sig_path(live, integrity).read_bytes() == old_bytes["sig"]
    assert notebook.state_path.read_bytes() == old_bytes["metadata"]
    assert manifest.read_bytes() == old_bytes["manifest"]
    assert sidecar.read_bytes() == old_bytes["h5ad"]
    assert _verify_pkl_integrity(live, integrity)


def test_applying_rollback_removes_new_h5ad_created_before_journal_extension(
    tmp_path,
):
    """A kill immediately after manifest publication cannot leave an orphan."""
    from backend.utils.runtime_apply import write_runtime_h5ad_manifest
    from backend.utils.sandbox import _write_pkl_signature

    states = tmp_path / "states"
    integrity = tmp_path / "integrity"
    states.mkdir()
    live = states / "new-sidecar_session.pkl"
    staged = states / "new-sidecar_session.pkl.tmp"
    manifest = states / "runtime_h5ad_manifest.json"
    new_sidecar = states / "new_adata.h5ad"
    live.write_bytes(b"old")
    staged.write_bytes(b"new")
    _write_pkl_signature(live, integrity)
    write_runtime_h5ad_manifest(
        "new-sidecar", {}, authoritative_pkl_names={"old_object"},
        states_dir=states,
    )
    old_manifest = manifest.read_bytes()
    notebook = NotebookState("new-sidecar", str(states))
    notebook.update_runtime_persistence(
        "current",
        current_revision=1,
        saved_revision=1,
        commit_source="regular_save",
        runtime_digest=_sha256_file(live),
    )

    def commit_manifest():
        new_sidecar.write_bytes(b"new-sidecar-bytes")
        write_runtime_h5ad_manifest(
            "new-sidecar",
            {"new_adata": _sha256_file(new_sidecar)},
            authoritative_pkl_names={"score"},
            states_dir=states,
        )

    def crash_after_manifest(point):
        if point == "manifest_replaced":
            raise _SimulatedProcessCrash(point)

    with pytest.raises(_SimulatedProcessCrash):
        _commit_pkl_transaction(
            staged,
            live,
            integrity,
            lambda: None,
            metadata_path=notebook.state_path,
            manifest_path=manifest,
            commit_manifest=commit_manifest,
            target_revision=2,
            authoritative_pkl_names={"score"},
            fault_hook=crash_after_manifest,
        )

    assert new_sidecar.exists()
    assert recover_regular_save_transaction(
        live,
        integrity,
        metadata_path=notebook.state_path,
        manifest_path=manifest,
    ) == "rollback"
    assert not new_sidecar.exists()
    assert manifest.read_bytes() == old_manifest


def test_in_process_regular_save_failure_removes_new_h5ad_product(tmp_path):
    from backend.utils.runtime_apply import write_runtime_h5ad_manifest
    from backend.utils.sandbox import _write_pkl_signature

    states = tmp_path / "states"
    integrity = tmp_path / "integrity"
    states.mkdir()
    live = states / "failed-sidecar_session.pkl"
    staged = states / "failed-sidecar_session.pkl.tmp"
    manifest = states / "runtime_h5ad_manifest.json"
    new_sidecar = states / "failed_adata.h5ad"
    live.write_bytes(b"old")
    staged.write_bytes(b"new")
    _write_pkl_signature(live, integrity)
    write_runtime_h5ad_manifest(
        "failed-sidecar", {}, authoritative_pkl_names={"old_object"},
        states_dir=states,
    )
    old_manifest = manifest.read_bytes()
    notebook = NotebookState("failed-sidecar", str(states))
    notebook.update_runtime_persistence(
        "current",
        current_revision=1,
        saved_revision=1,
        commit_source="regular_save",
        runtime_digest=_sha256_file(live),
    )

    def commit_manifest():
        new_sidecar.write_bytes(b"failed-new-sidecar")
        write_runtime_h5ad_manifest(
            "failed-sidecar",
            {"failed_adata": _sha256_file(new_sidecar)},
            authoritative_pkl_names={"score"},
            states_dir=states,
        )

    with pytest.raises(RuntimeError, match="metadata failed"):
        _commit_pkl_transaction(
            staged,
            live,
            integrity,
            lambda: (_ for _ in ()).throw(RuntimeError("metadata failed")),
            metadata_path=notebook.state_path,
            manifest_path=manifest,
            commit_manifest=commit_manifest,
            target_revision=2,
            authoritative_pkl_names={"score"},
        )

    assert not new_sidecar.exists()
    assert manifest.read_bytes() == old_manifest
    assert live.read_bytes() == b"old"
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


def test_regular_save_fails_closed_before_partial_rollback_when_backup_corrupt(
    tmp_path,
):
    from backend.utils.runtime_apply import write_runtime_h5ad_manifest
    from backend.utils.sandbox import _write_pkl_signature

    states = tmp_path / "states"
    integrity = tmp_path / "integrity"
    states.mkdir()
    live = states / "closed_session.pkl"
    staged = states / "closed_session.pkl.tmp"
    manifest = states / "runtime_h5ad_manifest.json"
    live.write_bytes(b"old")
    staged.write_bytes(b"new")
    _write_pkl_signature(live, integrity)
    write_runtime_h5ad_manifest(
        "closed", {}, authoritative_pkl_names={"adata"}, states_dir=states,
    )
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
            manifest_path=manifest,
            commit_manifest=lambda: None,
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
            manifest_path=manifest,
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


def test_regular_generation_manifest_names_pkl_and_retires_stale_h5ad(tmp_path):
    from backend.utils.runtime_apply import (
        quarantine_unreferenced_runtime_h5ads,
        write_runtime_h5ad_manifest,
    )

    stale = tmp_path / "stale_adata.h5ad"
    stale.write_bytes(b"old sidecar")
    manifest = write_runtime_h5ad_manifest(
        "regular-manifest",
        {},
        authoritative_pkl_names={"score_table", "adata"},
        states_dir=tmp_path,
    )
    quarantine_unreferenced_runtime_h5ads(
        "regular-manifest", set(), states_dir=tmp_path,
    )

    assert json.loads(manifest.read_text()) == {
        "version": 2,
        "entries": {},
        "authoritative_pkl_names": ["adata", "score_table"],
    }
    assert not stale.exists()
    assert (tmp_path / "stale_adata.h5ad.quarantine").read_bytes() == b"old sidecar"


def test_committed_sidecar_retire_failure_is_retried_without_pkl_rollback(
    monkeypatch, tmp_path,
):
    from backend.utils.sandbox import _write_pkl_signature

    states = tmp_path / "states"
    states.mkdir()
    integrity = tmp_path / "integrity"
    live = states / "retire_session.pkl"
    staged = states / "retire_session.pkl.tmp"
    manifest = states / "runtime_h5ad_manifest.json"
    stale = states / "stale.h5ad"
    live.write_bytes(b"old")
    staged.write_bytes(b"new")
    stale.write_bytes(b"stale")
    _write_pkl_signature(live, integrity)
    notebook = NotebookState("retire", str(states))
    old_digest = _sha256_file(live)
    new_digest = _sha256_file(staged)
    notebook.update_runtime_persistence(
        "current",
        current_revision=1,
        saved_revision=1,
        commit_source="regular_save",
        runtime_digest=old_digest,
    )

    def commit_manifest():
        from backend.utils.runtime_apply import write_runtime_h5ad_manifest
        write_runtime_h5ad_manifest(
            "retire", {}, authoritative_pkl_names={"adata"}, states_dir=states,
        )

    def commit_metadata():
        notebook.update_runtime_persistence(
            "current",
            current_revision=2,
            saved_revision=2,
            commit_source="regular_save",
            runtime_digest=new_digest,
        )

    _commit_pkl_transaction(
        staged,
        live,
        integrity,
        commit_metadata,
        metadata_path=notebook.state_path,
        manifest_path=manifest,
        commit_manifest=commit_manifest,
        retire_sidecars=lambda: (_ for _ in ()).throw(OSError("busy")),
        target_revision=2,
        authoritative_pkl_names={"adata"},
    )

    journal = live.with_name(f"{live.name}.regular-save.json")
    assert journal.exists()
    assert live.read_bytes() == b"new"
    assert stale.exists()

    outcome = recover_regular_save_transaction(
        live,
        integrity,
        metadata_path=notebook.state_path,
        manifest_path=manifest,
        session_id="retire",
    )

    assert outcome == "forward"
    assert live.read_bytes() == b"new"
    assert not stale.exists()
    assert (states / "stale.h5ad.quarantine").read_bytes() == b"stale"
    assert not journal.exists()
