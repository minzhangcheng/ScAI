"""Runtime restore orchestration for LocalJupyterSandbox."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class RuntimeIntegrityError(RuntimeError):
    """The dump does not contain what its own manifest declared."""


@dataclass(frozen=True)
class RestorePlan:
    """What a generation promised, decided before the kernel loads anything."""

    declared: set[str] | None
    legacy: bool
    pairing: dict | None = None


def plan_restore(manifest_path, live_pkl) -> RestorePlan:
    """Decide what to expect before asking the kernel to load a single frame.

    A generation written before manifests existed declares nothing, and that is
    age rather than corruption: it loads with coverage unknown and is upgraded
    by the next successful save. A manifest that is present but unsigned,
    malformed, or describing different bytes is the opposite -- something is
    wrong with this generation, and it must not be quietly downgraded to the
    legacy path.
    """
    from backend.utils.runtime_manifest import (
        RuntimeManifestError,
        read_runtime_manifest,
        runtime_manifest_sig_path,
    )
    from backend.utils.sandbox_parts.regular_save import _sha256_file

    manifest_path = Path(manifest_path)
    signature_path = runtime_manifest_sig_path(manifest_path)
    if not manifest_path.exists() and not signature_path.exists():
        return RestorePlan(declared=None, legacy=True)

    try:
        manifest = read_runtime_manifest(manifest_path)
    except RuntimeManifestError as exc:
        raise RuntimeIntegrityError(str(exc)) from exc

    # Chunked: real dumps reach several gigabytes, and read_bytes() would put
    # all of it in the backend process at once.
    digest = _sha256_file(Path(live_pkl))
    if digest != manifest["pkl_sha256"]:
        raise RuntimeIntegrityError(
            "runtime dump digest does not match its manifest; refusing to load it"
        )
    return RestorePlan(declared=set(manifest["saved_names"]), legacy=False, pairing=manifest.get("pairing"))


def verify_restored_generation(plan: RestorePlan, restored) -> None:
    """Refuse a generation whose declared contents did not all come back.

    Names the manifest recorded as skipped are expected to be absent; that is
    coverage, and it warns elsewhere. Names it declared saved are a promise, and
    a broken promise means the dump and the manifest disagree.
    """
    if plan.declared is None:
        return
    missing = sorted(plan.declared - set(restored))
    if missing:
        raise RuntimeIntegrityError(
            "runtime generation is incomplete; declared but not restored: "
            + ", ".join(missing)
        )


def _names_only_in_retired_sidecars(manifest_path) -> set[str]:
    """Names a retired h5ad sidecar manifest claims, if any.

    Written only by the snapshot restore, which routed AnnData to h5ad files
    and everything else to the dill residual -- so a name listed here is a name
    the pkl beside it does not carry. The ordinary save also wrote this file
    for a while, always with an empty ``entries`` map; that form claims nothing
    and must not refuse anything. An unreadable manifest is treated as empty:
    it cannot name a variable, so it cannot be the reason a namespace is
    incomplete.
    """
    import json

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if not isinstance(payload, dict):
        return set()
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return set()
    return {name for name in entries if isinstance(name, str) and name}


def _mark_runtime_unavailable_if_work_exists(sandbox, *, bindings) -> None:
    """Record that a session which has run has no namespace to continue from.

    A fresh session restores nothing and that is correct. One that has completed
    rounds and restores nothing has lost its runtime, and the difference matters:
    the first should start, the second must be offered a re-run rather than
    continue the graph over variables that no longer exist.

    ``_mark_runtime_availability_after_import`` makes this call for an imported
    session. Nothing made it for a local process restart, so the status stayed
    "complete" and the run proceeded.
    """
    from backend.state import sessions as live_sessions

    from backend.utils.runtime_pairing import has_completed_runtime_work

    state = live_sessions.get(sandbox.session_id)
    if getattr(state, "runtime_recompute_pending", False):
        return
    if state is None or not has_completed_runtime_work(state):
        return
    if getattr(state, "runtime_restore_status", "complete") != "complete":
        return
    state.runtime_restore_status = "unavailable"
    state.runtime_restore_warning = (
        "This session has completed rounds but no runtime state could be "
        "restored, so in-memory values from the earlier run (adata and friends) "
        "are gone. Re-run the analysis from the imported data to recompute them."
    )
    bindings.logger.warning(
        "[%s] session has %s completed round(s) but nothing to restore; marked "
        "runtime unavailable so the graph is not run over a missing namespace",
        sandbox.session_id,
        getattr(state, "current_round", 0),
    )


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
            # A journal written before the sidecar was retired can still name it
            # as a component; recovery has to resolve that path to roll such a
            # transaction back.
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
        # A snapshot apply from the previous build and the ordinary save
        # mutate the same live runtime files. Recover both journals while
        # holding the same lifecycle transaction lock used by commit and
        # deletion; nothing writes the first kind any more.
        recover_runtime_apply_journal(sandbox.session_id)
    if regular_recovery is not None:
        # NotebookState was constructed before recovery inspected the
        # journal. Reload the generation metadata that rollback/forward
        # just made authoritative, even when no .ipynb exists yet.
        sandbox.notebook_state._load_runtime_persistence_metadata()
        persistence = sandbox.notebook_state.runtime_persistence
        sandbox.runtime_revision = int(persistence.get("current_revision") or 0)
        sandbox.saved_runtime_revision = int(persistence.get("saved_revision") or 0)

    # An unresolved APPLYING restore journal from the previous build blocks
    # kernel rehydrate: the session's files describe different points in the run.
    from backend.services.session_store import assert_recovery_clear
    assert_recovery_clear(session_id=sandbox.session_id)

    sandbox.notebook_state.load()

    states_dir = bindings.Path(sandbox.notebook_state.session_path).parent
    depends_on_retired_sidecars = _names_only_in_retired_sidecars(
        states_dir / RUNTIME_H5AD_MANIFEST_FILENAME
    )
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
        and max(attempted_revision, current_revision) > committed_revision
    )
    attempted_revision = max(attempted_revision, current_revision)
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
    if depends_on_retired_sidecars:
        # The previous build kept these names outside the pkl, in h5ad
        # sidecars its snapshot restore installed. Nothing here can read them,
        # and the pkl beside them does not carry them -- loading it alone would
        # bring the kernel up quietly missing the session's main AnnData. Say
        # so and offer the re-run instead.
        bindings.logger.warning(
            "[%s] runtime depends on retired h5ad sidecars (%s); refusing to "
            "rehydrate a partial namespace",
            sandbox.session_id,
            ", ".join(sorted(depends_on_retired_sidecars)),
        )
        _mark_runtime_unavailable_if_work_exists(sandbox, bindings=bindings)
        return
    if not has_session_dump:
        bindings.logger.info("No session dump to restore")
        _mark_runtime_unavailable_if_work_exists(sandbox, bindings=bindings)
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
    escaped_path = repr(session_path)
    bindings.logger.info(f"Restoring session from {session_path}")

    # Decided here, on the host, before a single frame reaches the kernel: a
    # dump whose digest disagrees with its manifest is never loaded at all.
    try:
        restore_plan = plan_restore(
            runtime_manifest_path(
                sandbox.session_id,
                states_dir=sandbox.notebook_state.state_dir,
            ),
            sandbox.notebook_state.session_path,
        ) if has_session_dump else RestorePlan(declared=None, legacy=True)
    except RuntimeIntegrityError as integrity_error:
        sandbox._session_restore_failed = True
        sandbox.notebook_state.update_runtime_persistence(
            "failed",
            current_revision=current_revision,
            saved_revision=saved_revision,
            error_summary=f"Runtime generation integrity check failed: {integrity_error}",
        )
        bindings.logger.error(
            "[%s] refusing to restore a generation that fails its own manifest: %s",
            sandbox.session_id,
            integrity_error,
        )
        raise

    restore_code = f"""
import dill as _scai_dill
import hashlib as _scai_hashlib
import json as _scai_json
from pathlib import Path as _scai_Path

# Imported under private names: this runs in the kernel's own globals, so plain
# names would rebind any user variable that happened to share one.

_restored_names = []
# Frames land here first and are published in one step. Assigning straight into
# globals() left a partially rebuilt namespace behind whenever a later frame
# raised -- and the integrity check that would have caught it cannot run until
# every frame has been read.
_scai_pending = {{}}
_compression_magic = b"SCAI_ZSTD_DILL_V1\\n"

def _restore_dill_frames(_stream):
    while True:
        try:
            _frame = _scai_dill.load(_stream)
        except EOFError:
            break
        if (
            isinstance(_frame, tuple)
            and len(_frame) == 3
            and _frame[0] == "__SCAI_ALIAS__"
        ):
            _, _name, _alias_target = _frame
            if _alias_target in _scai_pending:
                _obj = _scai_pending[_alias_target]
            elif _alias_target in globals():
                _obj = globals()[_alias_target]
            else:
                raise RuntimeError(
                    f"runtime alias target missing: {{_name}} -> {{_alias_target}}"
                )
        else:
            _name, _obj = _frame
        _scai_pending[_name] = _obj
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

    # Every frame read: publish the namespace in one step, then report what
    # it holds in a form the host can parse.
    globals().update(_scai_pending)
    # Released immediately: this dict holds a strong reference to every restored
    # object, so leaving it bound would keep a replaced AnnData alive for the
    # rest of the session.
    _scai_pending.clear()
    print(f"Session restored: {{len(_restored_names)}} variables")
    print("__SCAI_RESTORED_NAMES__:" + _scai_json.dumps(sorted(_restored_names)))
except Exception as e:
    print(f"__SCAI_RESTORE_ERROR__: {{type(e).__name__}}: {{e}}")
    raise
"""
    result = await sandbox._execute_unlocked(restore_code, records_history=False, skip_safety_check=True)

    if result.success and "Session restored:" in (result.output or ""):
        restored_names: set[str] = set()
        for line in (result.output or "").splitlines():
            if line.startswith("__SCAI_RESTORED_NAMES__:"):
                try:
                    parsed = bindings.json.loads(
                        line[len("__SCAI_RESTORED_NAMES__:"):]
                    )
                except (bindings.json.JSONDecodeError, TypeError):
                    parsed = None
                if isinstance(parsed, list):
                    restored_names = {str(name) for name in parsed}
        try:
            verify_restored_generation(restore_plan, restored_names)
        except RuntimeIntegrityError as integrity_error:
            sandbox._session_restore_failed = True
            sandbox.notebook_state.update_runtime_persistence(
                "failed",
                current_revision=current_revision,
                saved_revision=saved_revision,
                error_summary=(
                    f"Runtime generation integrity check failed: {integrity_error}"
                ),
            )
            bindings.logger.error(
                "[%s] restored namespace does not match its manifest: %s",
                sandbox.session_id,
                integrity_error,
            )
            raise
        sandbox._session_restored = True
        sandbox._session_restore_failed = False
        sandbox.saved_runtime_revision = saved_revision
        sandbox.runtime_revision = current_revision
        sandbox.notebook_state.runtime_persistence["committed_pairing"] = restore_plan.pairing
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
