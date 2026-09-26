import json
import asyncio
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


def test_a_commit_rejects_a_checkpoint_copy_that_changed_pairing(
    monkeypatch, tmp_path,
):
    """The signed identity must describe the SQLite snapshot actually copied."""
    import sqlite3

    from backend.utils.runtime_pairing import checkpoint_identity_at_path
    from backend.utils.sandbox_parts import regular_save

    states = tmp_path / "states"
    states.mkdir()
    live = states / "s_session.pkl"
    staged = states / "s_session.pkl.tmp"
    staged.write_bytes(b"new-generation")
    state_json = states / "s_state.json"
    state_json.write_text(json.dumps({"session_id": "s", "state_version": 12}))
    checkpoint = states / "checkpoints.sqlite"
    with sqlite3.connect(checkpoint) as conn:
        conn.execute(
            "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, "
            "checkpoint_id TEXT, checkpoint BLOB)"
        )
        conn.execute("INSERT INTO checkpoints VALUES ('s', '', 'c1', X'00')")
    expected = checkpoint_identity_at_path("s", checkpoint)
    real_copy = regular_save._copy_sqlite_durable

    def copy_then_change(source, target):
        real_copy(source, target)
        with sqlite3.connect(target) as conn:
            conn.execute(
                "CREATE TABLE writes (thread_id TEXT, checkpoint_ns TEXT, "
                "checkpoint_id TEXT, task_id TEXT, idx INTEGER, value BLOB)"
            )
            conn.execute(
                "INSERT INTO writes VALUES ('s', '', 'c1', 'task', 0, X'01')"
            )

    monkeypatch.setattr(regular_save, "_copy_sqlite_durable", copy_then_change)
    with pytest.raises(RuntimeError, match="pairing changed"):
        _commit_pkl_transaction(
            staged,
            live,
            tmp_path / "integrity",
            lambda: None,
            runtime_manifest_path=states / "s_session.runtime.json",
            runtime_manifest={
                "state_revision": 12,
                "pairing": {"state_version": 12, "checkpoint": expected},
                "saved_names": [],
                "skipped_names": [],
                "serializer_version": "dill-zstd-1",
            },
            target_revision=7,
            generation_root=states / "generation",
            paired_sources={"state": state_json, "checkpoint": checkpoint},
            pairing_session_id="s",
        )

    assert not live.exists()
    assert not (states / "generation" / "7" / "checkpoints.sqlite").exists()


@pytest.mark.asyncio
async def test_executor_pair_refresh_bypasses_unchanged_runtime_reuse(
    monkeypatch, tmp_path,
):
    """A new executor transcript needs a pair even without new kernel cells."""
    import logging

    from backend.state import sessions
    from backend.utils.sandbox_parts import runtime_save

    sid = "executor-pair-refresh"
    state = SimpleNamespace(state_version=7)
    monkeypatch.setitem(sessions, sid, state)
    pairing = {"state_version": 7, "checkpoint": None}
    session_path = tmp_path / f"{sid}_session.pkl"
    session_path.write_bytes(b"committed")
    notebook = SimpleNamespace(
        state_dir=tmp_path,
        session_path=session_path,
        runtime_persistence={
            "committed_pairing": pairing,
            "commit_source": "regular_save",
            "saved_revision": 4,
        },
        has_session_dump=lambda: True,
    )
    sandbox = SimpleNamespace(
        session_id=sid,
        notebook_state=notebook,
        runtime_revision=4,
        saved_runtime_revision=4,
        _session_restore_failed=False,
        _restore_failed_at_revision=None,
        _expected_lifecycle_generation=1,
        runtime_budgets={},
    )
    bindings = SimpleNamespace(asyncio=asyncio, logger=logging.getLogger(__name__))

    assert await runtime_save._prepare_save(sandbox, bindings=bindings) is True
    sandbox._force_generation_pair_refresh = True
    prepared = await runtime_save._prepare_save(sandbox, bindings=bindings)

    assert isinstance(prepared, runtime_save.SaveContext)


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
