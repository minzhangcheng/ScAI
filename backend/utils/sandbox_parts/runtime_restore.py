"""Runtime restore orchestration for LocalJupyterSandbox."""
from __future__ import annotations


async def restore_session(sandbox, *, bindings):
    """Restore session from per-variable streaming dill dump if available.

        New dumps contain a short format marker followed by a streaming zstd
        sequence of dill frames. Legacy uncompressed frame streams remain
        readable so existing sessions do not require migration.
        """
    # Recover any interrupted runtime-pair apply FIRST — before the
    # has_session_dump() early-return — so a crashed apply that left no live
    # pkl cannot leave a journal lingering until a later start (where it could
    # otherwise interfere with a subsequently saved pkl).
    from ..runtime_apply import (
        RUNTIME_H5AD_MANIFEST_FILENAME,
        recover_runtime_apply_journal,
        verify_live_pkl_or_raise,
        RuntimeApplyError,
    )
    from backend.services.session_lifecycle import (
        assert_session_generation_active,
        lifecycle_file_lock,
    )

    from backend.utils.runtime_manifest import runtime_manifest_path

    with lifecycle_file_lock(sandbox.session_id):
        assert_session_generation_active(sandbox.session_id)
        regular_recovery = bindings.recover_regular_save_transaction(
            bindings.Path(sandbox.notebook_state.session_path),
            bindings._integrity_dir(sandbox.session_id),
            metadata_path=bindings.Path(sandbox.notebook_state.state_path),
            manifest_path=(
                bindings.Path(sandbox.notebook_state.state_dir)
                / RUNTIME_H5AD_MANIFEST_FILENAME
            ),
            runtime_manifest_path=runtime_manifest_path(
                sandbox.session_id,
                states_dir=sandbox.notebook_state.state_dir,
            ),
            session_id=sandbox.session_id,
        )
        # Snapshot apply and regular save mutate the same live runtime
        # files. Recover both journals while holding the same lifecycle
        # transaction lock used by commit and deletion.
        recover_runtime_apply_journal(sandbox.session_id)
    if regular_recovery is not None:
        # NotebookState was constructed before recovery inspected the
        # journal. Reload the generation metadata that rollback/forward
        # just made authoritative, even when no .ipynb exists yet.
        sandbox.notebook_state._load_runtime_persistence_metadata()
        persistence = sandbox.notebook_state.runtime_persistence
        sandbox.runtime_revision = int(persistence.get("current_revision") or 0)
        sandbox.saved_runtime_revision = int(persistence.get("saved_revision") or 0)

    # Snapshot V2: unresolved APPLYING journals block kernel rehydrate
    from backend.services.session_store import assert_recovery_clear
    assert_recovery_clear(session_id=sandbox.session_id)

    sandbox.notebook_state.load()

    states_dir = bindings.Path(sandbox.notebook_state.session_path).parent
    h5ad_manifest = states_dir / RUNTIME_H5AD_MANIFEST_FILENAME
    has_h5ad_manifest = h5ad_manifest.exists()
    persistence = dict(sandbox.notebook_state.runtime_persistence)
    attempted_revision = int(
        persistence.get("last_attempt_revision")
        or persistence.get("current_revision")
        or 0
    )
    current_revision = int(persistence.get("current_revision") or 0)
    saved_revision = int(persistence.get("saved_revision") or 0)
    committed_dump = sandbox.notebook_state.has_session_dump()
    committed_revision = int(
        persistence.get("committed_revision") or saved_revision or 0
    )
    committed_source = (
        persistence.get("committed_source")
        or persistence.get("commit_source")
    )
    degraded_recovery = bool(
        committed_dump
        and persistence.get("last_attempt_status") == "failed"
        and attempted_revision > committed_revision
    )
    if committed_dump:
        # A failed newer attempt may have advanced current_revision without
        # publishing a pkl. Rehydrate the verified committed generation,
        # not that abandoned in-memory revision.
        current_revision = committed_revision
        saved_revision = committed_revision

    if (
        sandbox.notebook_state.session_path.exists()
        and not committed_dump
    ):
        sandbox.notebook_state.update_runtime_persistence(
            "stale",
            current_revision=current_revision,
            saved_revision=saved_revision,
            error_summary=(
                "A previous runtime dump exists but does not represent the "
                "latest live revision. Recompute required continuation state."
            ),
        )
        raise RuntimeError(
            f"Stale runtime generation for session {sandbox.session_id}: "
            f"live revision {current_revision}, saved revision {saved_revision}."
        )

    has_session_dump = committed_dump
    if not has_session_dump and not has_h5ad_manifest:
        bindings.logger.info("No session dump to restore")
        return

    # A present pkl with a missing/invalid HMAC is a hard error, not a silent
    # skip: continuing would rehydrate a kernel inconsistent with the state
    # that referenced this pkl. Raising lets the caller surface a typed
    # runtime-failure state (Stage 1 safety contract).
    if has_session_dump:
        try:
            verify_live_pkl_or_raise(sandbox.session_id)
        except RuntimeApplyError:
            # Symmetric with the dill-restore-failure path below: mark the failure
            # so start()'s "restore_failed" result reflects it, then abort startup.
            sandbox._session_restore_failed = True
            sandbox.notebook_state.update_runtime_persistence(
                "failed",
                current_revision=current_revision,
                saved_revision=saved_revision,
                error_summary="Live runtime integrity verification failed.",
            )
            bindings.logger.error(
                "[%s] live pkl present but HMAC invalid; aborting session restore",
                sandbox.session_id,
            )
            raise

    session_path = str(sandbox.notebook_state.session_path.absolute())
    h5ad_manifest_path = str(h5ad_manifest.absolute())
    escaped_path = repr(session_path)
    escaped_h5ad_manifest_path = repr(h5ad_manifest_path)
    bindings.logger.info(f"Restoring session from {session_path}")

    restore_code = f"""
import dill
import hashlib
import json
from pathlib import Path

_restored_names = []
_compression_magic = b"SCAI_ZSTD_DILL_V1\\n"

def _restore_dill_frames(_stream):
    while True:
        try:
            _frame = dill.load(_stream)
        except EOFError:
            break
        if (
            isinstance(_frame, tuple)
            and len(_frame) == 3
            and _frame[0] == "__SCAI_ALIAS__"
        ):
            _, _name, _alias_target = _frame
            if _alias_target not in globals():
                raise RuntimeError(
                    f"runtime alias target missing: {{_name}} -> {{_alias_target}}"
                )
            _obj = globals()[_alias_target]
        else:
            _name, _obj = _frame
        globals()[_name] = _obj
        _restored_names.append(_name)

try:
    if {has_session_dump!r}:
        with open({escaped_path}, 'rb') as _raw:
            _marker = _raw.read(len(_compression_magic))
            if _marker == _compression_magic:
                import zstandard as _zstd
                with _zstd.ZstdDecompressor().stream_reader(_raw) as _f:
                    _restore_dill_frames(_f)
            else:
                _raw.seek(0)
                _restore_dill_frames(_raw)

    _h5ad_manifest_path = Path({escaped_h5ad_manifest_path})
    if _h5ad_manifest_path.exists():
        try:
            import anndata as _runtime_anndata
        except Exception:
            import scanpy as _runtime_scanpy
            _runtime_read_h5ad = _runtime_scanpy.read_h5ad
        else:
            _runtime_read_h5ad = _runtime_anndata.read_h5ad
        _payload = json.loads(_h5ad_manifest_path.read_text())
        _states_root = _h5ad_manifest_path.parent.resolve()
        import re as _re
        _name_re = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]{{0,127}}$")
        for _name, _entry in sorted((_payload.get("entries") or {{}}).items()):
            if not isinstance(_name, str) or not _name_re.fullmatch(_name):
                raise RuntimeError(f"invalid h5ad sidecar variable name: {{_name!r}}")
            if not isinstance(_entry, dict):
                raise RuntimeError(f"invalid h5ad sidecar entry for {{_name}}")
            _expected_filename = f"{{_name}}.h5ad"
            _raw_path = _entry.get("path")
            if not isinstance(_raw_path, str) or _raw_path != _expected_filename:
                raise RuntimeError(
                    f"h5ad sidecar path must be exactly {{_expected_filename!r}}, got {{_raw_path!r}}"
                )
            # The live pkl wins whenever it provides the name. The sidecar set
            # is written only by snapshot restore / recovery and is never
            # retired: save_session() dumps every live AnnData into the pkl but
            # leaves runtime_h5ad_manifest.json frozen at the restored
            # snapshot's content, hashes and all. Overwriting unconditionally
            # therefore rolled the namespace back to the restored snapshot on
            # every later rehydrate (kernel OOM, session reopen), silently
            # discarding every cell run since the restore -- and the sidecar
            # hash still matched, so nothing flagged it. A restore's own pkl
            # never carries the h5ad names (RuntimeCapture routes AnnData to
            # h5ad and everything else to the dill residual), so the sidecar is
            # still authoritative for the rehydrate that immediately follows it.
            if _name in _restored_names:
                continue
            # Never join raw manifest path; only the validated basename.
            _path = (_states_root / _expected_filename).resolve()
            if _path.parent != _states_root:
                raise RuntimeError(f"h5ad sidecar path escapes states dir: {{_path}}")
            if not _path.is_file():
                raise RuntimeError(f"h5ad sidecar missing for {{_name}}: {{_path}}")
            _expected = _entry.get("sha256", "")
            _actual = hashlib.sha256(_path.read_bytes()).hexdigest()
            if _actual != _expected:
                raise RuntimeError(f"h5ad sidecar hash mismatch for {{_name}}")
            globals()[_name] = _runtime_read_h5ad(_path)
            _restored_names.append(_name)

    print(f"Session restored: {{len(_restored_names)}} variables")
    print(f"Restored: {{_restored_names}}")
except Exception as e:
    print(f"__SCAI_RESTORE_ERROR__: {{type(e).__name__}}: {{e}}")
    raise
"""
    result = await sandbox._execute_unlocked(restore_code, save_to_history=False, skip_safety_check=True)

    if result.success and "Session restored:" in (result.output or ""):
        sandbox._session_restored = True
        sandbox._session_restore_failed = False
        sandbox.saved_runtime_revision = saved_revision
        sandbox.runtime_revision = current_revision
        sandbox.notebook_state.record_runtime_restored(
            current_revision=current_revision,
            saved_revision=saved_revision,
            commit_source=committed_source,
            degraded_recovery=degraded_recovery,
            attempted_revision=attempted_revision,
        )
        if degraded_recovery:
            bindings.logger.warning(
                "[%s] Restored committed runtime revision %d after newer "
                "revision %d failed to persist; recovery is degraded",
                sandbox.session_id,
                committed_revision,
                attempted_revision,
            )
        bindings.logger.info("Session restored successfully")
    else:
        # P1-11: dill restore failure leaves the kernel without previously-saved
        # variables while the notebook metadata has already been loaded. Continuing
        # would yield silent NameErrors downstream. Surface the failure loudly and
        # abort start-up so the workflow can react (retry / fresh kernel / restore
        # from earlier snapshot).
        sandbox._session_restore_failed = True
        # Mark of how much work the kernel had done when persistence was
        # latched off. Cheap short-circuit only: a kernel that has run
        # nothing since cannot have rebuilt anything -- see save_session.
        sandbox._restore_failed_at_revision = getattr(sandbox, "runtime_revision", 0)
        # What the aborted restore left the namespace holding. This, not the
        # cell counter, is what save_session compares against: a counter
        # cannot tell "the agent reloaded the data and recomputed the
        # analysis" from "one `import scanpy as sc` ran", and both advance it
        # by exactly one. Taken here because the kernel is still alive at
        # this point; None means it could not be read, and save_session then
        # keeps refusing rather than guessing.
        sandbox._restore_failed_globals = await sandbox._snapshot_persistable_globals()
        detail = result.error or result.output or "unknown error"
        sandbox.notebook_state.update_runtime_persistence(
            "failed",
            current_revision=current_revision,
            saved_revision=saved_revision,
            failed_variables=sorted(sandbox.get_runtime_dependency_names()),
            error_summary=detail,
        )
        bindings.logger.error(
            f"[{sandbox.session_id}] dill session restore failed: {detail}"
        )
        raise RuntimeError(
            f"Sandbox restore failed for session {sandbox.session_id}: {detail}. "
            f"Previous kernel variable state is unavailable. Re-run the analysis "
            f"from your data to rebuild kernel state; snapshot restore is only "
            f"available where Snapshot V2 is enabled."
        )
