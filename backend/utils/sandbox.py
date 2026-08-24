"""
Local Jupyter Kernel Executor with State Persistence

Uses dill.dump_session/load_session to save and restore the complete Python environment.
Supports pause/resume by saving and restoring the full session state.
"""

import os
import re
import json
import base64
import asyncio
import time
import hmac
import hashlib
import logging
from queue import Empty as QueueEmpty
from typing import Optional, Dict, Any, List, Tuple, Callable
from datetime import datetime
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_output
from jupyter_client import KernelManager

from backend.utils.crypto import get_integrity_secret, sign_hmac_sha256
from backend.config._config import DATA_BASE_DIR

# Re-exported so `backend.utils.sandbox.<name>` keeps resolving
# for existing importers and for tests that patch these by name.
from .sandbox_parts.shared import (  # noqa: F401
    os,
    re,
    json,
    base64,
    asyncio,
    time,
    hmac,
    hashlib,
    logging,
    QueueEmpty,
    Optional,
    Dict,
    Any,
    List,
    Tuple,
    datetime,
    Path,
    nbformat,
    new_notebook,
    new_code_cell,
    new_output,
    KernelManager,
    get_integrity_secret,
    sign_hmac_sha256,
    DATA_BASE_DIR,
    IOPUB_POLL_CAP,
    _sig_path,
    _pkl_hmac,
    _pkl_hmac_stream,
    _sha256_file,
    _write_pkl_signature,
    _commit_pkl_transaction,
    recover_regular_save_transaction,
    _verify_pkl_integrity,
    _integrity_dir,
    _path_is_within,
    compute_remaining_timeout,
    logger,
    _is_kernel_protocol_error,
    DANGEROUS_PATTERNS,
    _is_safe_os_system,
    _is_safe_subprocess,
    _is_safe_importlib,
    _is_safe_file_write,
    _EXCEPTION_CHECKS,
    _DANGEROUS_AST_FUNCS,
    _DANGEROUS_AST_ATTRS,
    _DANGEROUS_AST_MODULES,
    _SUBPROCESS_EXEC_FUNCS,
    _validate_code_safety_ast,
    validate_code_safety,
    extract_code_from_markdown,
    CodeExecutionResult,
    NotebookState,
    _TPL_NOOP_INSTALL,
)
from .sandbox_parts import (  # noqa: F401
    KernelMixin,
    ExecutionMixin,
    StateTrackingMixin,
    PersistenceMixin,
    PathsMixin,
)


async def _run_durable_host_operation(
    operation: Callable[[], Any],
    *,
    session_id: str,
    description: str,
) -> Any:
    """Run blocking durable I/O in the owned bulk worker.

    A caller cancellation cannot stop a filesystem worker halfway through a
    digest or transaction. The owned worker is therefore drained before the
    cancellation is re-delivered to the caller.
    """
    from backend.utils.asyncio_tasks import run_in_owned_bulk_thread

    try:
        return await run_in_owned_bulk_thread(operation)
    except asyncio.CancelledError:
        logger.warning(
            "[%s] Cancellation requested during %s was deferred until the "
            "durable file operation reached a safe boundary",
            session_id,
            description,
        )
        raise


class LocalJupyterSandbox(
    KernelMixin,
    ExecutionMixin,
    StateTrackingMixin,
    PersistenceMixin,
    PathsMixin,
):
    """
    Local Jupyter kernel executor with full session persistence using dill.
    
    Features:
    - Executes code in a local Jupyter kernel
    - Saves complete session state using dill.dump_session()
    - Restores session state using dill.load_session()
    - Saves notebook history as .ipynb file
    """
    
    def __init__(
        self,
        session_id: str,
        state_dir: str = "states",
        working_dir: str = "workspace",
        timeout: int = 1200,
        runtime_budgets: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        from backend.services.session_lifecycle import (
            lifecycle_file_lock,
            read_lifecycle_record,
        )

        with lifecycle_file_lock(session_id):
            lifecycle_record = read_lifecycle_record(session_id)
            self._expected_lifecycle_generation = int(
                (lifecycle_record or {}).get("generation") or 0
            )
        import tempfile as _tmpfile_mod
        self._session_tmp = Path(_tmpfile_mod.gettempdir()) / f"scai-{self.session_id}"
        if self._session_tmp.is_symlink():
            raise RuntimeError("session temporary root must not be a symlink")
        self._session_tmp.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._session_tmp.chmod(0o700)
        self.state_dir = state_dir
        # Caller passes session-specific working dir (e.g. sessions/{id}/workspace)
        self.working_dir = Path(working_dir).absolute()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.runtime_budgets = dict(runtime_budgets or {})
        
        self.km: Optional[KernelManager] = None
        self.kc = None
        self.notebook_state = NotebookState(session_id, state_dir)
        self._kernel_staging_root = (
            self.notebook_state.state_dir / ".kernel-staging"
        )
        if self._kernel_staging_root.is_symlink():
            raise RuntimeError("kernel staging root must not be a symlink")
        self._initialized = False
        persistence = self.notebook_state.runtime_persistence
        self.runtime_revision: int = int(persistence.get("current_revision") or 0)
        self.saved_runtime_revision: int = int(persistence.get("saved_revision") or 0)
        self._session_restored = False
        self._session_restore_failed = False  # Track if session restore failed
        self._kernel_dead = False  # Set when kernel restart fails
        self._restarting = False  # reentrancy guard for _restart_kernel
        # Monotonic generation of the live kernel namespace. Bumped by every
        # start and restart, so a multi-cell operation (snapshot runtime
        # capture in particular) can assert that all of its cells ran against
        # one namespace instead of silently mixing generations.
        self.kernel_epoch = 0
        self._pending_save_tasks: set = set()
        self._pending_save_task: asyncio.Task[bool] | None = None
        # Snapshot V2 runtime dependency tracking (created/loaded user globals).
        self._known_user_globals: set[str] = set()
        self._last_execution_diff: Dict[str, Any] = {
            "created": set(),
            "loaded": set(),
            "removed": set(),
        }
        self.runtime_dependency_tracker = None  # set by agent / capture pipeline
        self._init_exec_lock()



    
    
    
    async def _restore_session(self):
        """Restore session from per-variable streaming dill dump if available.

        New dumps contain a short format marker followed by a streaming zstd
        sequence of dill frames. Legacy uncompressed frame streams remain
        readable so existing sessions do not require migration.
        """
        # Recover any interrupted runtime-pair apply FIRST — before the
        # has_session_dump() early-return — so a crashed apply that left no live
        # pkl cannot leave a journal lingering until a later start (where it could
        # otherwise interfere with a subsequently saved pkl).
        from .runtime_apply import (
            RUNTIME_H5AD_MANIFEST_FILENAME,
            recover_runtime_apply_journal,
            verify_live_pkl_or_raise,
            RuntimeApplyError,
        )
        from backend.services.session_lifecycle import (
            assert_session_generation_active,
            lifecycle_file_lock,
        )

        with lifecycle_file_lock(self.session_id):
            assert_session_generation_active(self.session_id)
            regular_recovery = recover_regular_save_transaction(
                Path(self.notebook_state.session_path),
                _integrity_dir(self.session_id),
                metadata_path=Path(self.notebook_state.state_path),
                manifest_path=(
                    Path(self.notebook_state.state_dir)
                    / RUNTIME_H5AD_MANIFEST_FILENAME
                ),
                session_id=self.session_id,
            )
            # Snapshot apply and regular save mutate the same live runtime
            # files. Recover both journals while holding the same lifecycle
            # transaction lock used by commit and deletion.
            recover_runtime_apply_journal(self.session_id)
        if regular_recovery is not None:
            # NotebookState was constructed before recovery inspected the
            # journal. Reload the generation metadata that rollback/forward
            # just made authoritative, even when no .ipynb exists yet.
            self.notebook_state._load_runtime_persistence_metadata()
            persistence = self.notebook_state.runtime_persistence
            self.runtime_revision = int(persistence.get("current_revision") or 0)
            self.saved_runtime_revision = int(persistence.get("saved_revision") or 0)

        # Snapshot V2: unresolved APPLYING journals block kernel rehydrate
        from backend.services.session_store import assert_recovery_clear
        assert_recovery_clear(session_id=self.session_id)

        self.notebook_state.load()

        states_dir = Path(self.notebook_state.session_path).parent
        h5ad_manifest = states_dir / RUNTIME_H5AD_MANIFEST_FILENAME
        has_h5ad_manifest = h5ad_manifest.exists()
        persistence = dict(self.notebook_state.runtime_persistence)
        attempted_revision = int(
            persistence.get("last_attempt_revision")
            or persistence.get("current_revision")
            or 0
        )
        current_revision = int(persistence.get("current_revision") or 0)
        saved_revision = int(persistence.get("saved_revision") or 0)
        committed_dump = self.notebook_state.has_session_dump()
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
            self.notebook_state.session_path.exists()
            and not committed_dump
        ):
            self.notebook_state.update_runtime_persistence(
                "stale",
                current_revision=current_revision,
                saved_revision=saved_revision,
                error_summary=(
                    "A previous runtime dump exists but does not represent the "
                    "latest live revision. Recompute required continuation state."
                ),
            )
            raise RuntimeError(
                f"Stale runtime generation for session {self.session_id}: "
                f"live revision {current_revision}, saved revision {saved_revision}."
            )

        has_session_dump = committed_dump
        if not has_session_dump and not has_h5ad_manifest:
            logger.info("No session dump to restore")
            return

        # A present pkl with a missing/invalid HMAC is a hard error, not a silent
        # skip: continuing would rehydrate a kernel inconsistent with the state
        # that referenced this pkl. Raising lets the caller surface a typed
        # runtime-failure state (Stage 1 safety contract).
        if has_session_dump:
            try:
                verify_live_pkl_or_raise(self.session_id)
            except RuntimeApplyError:
                # Symmetric with the dill-restore-failure path below: mark the failure
                # so start()'s "restore_failed" result reflects it, then abort startup.
                self._session_restore_failed = True
                self.notebook_state.update_runtime_persistence(
                    "failed",
                    current_revision=current_revision,
                    saved_revision=saved_revision,
                    error_summary="Live runtime integrity verification failed.",
                )
                logger.error(
                    "[%s] live pkl present but HMAC invalid; aborting session restore",
                    self.session_id,
                )
                raise

        session_path = str(self.notebook_state.session_path.absolute())
        h5ad_manifest_path = str(h5ad_manifest.absolute())
        escaped_path = repr(session_path)
        escaped_h5ad_manifest_path = repr(h5ad_manifest_path)
        logger.info(f"Restoring session from {session_path}")

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
        result = await self._execute_unlocked(restore_code, save_to_history=False, skip_safety_check=True)

        if result.success and "Session restored:" in (result.output or ""):
            self._session_restored = True
            self._session_restore_failed = False
            self.saved_runtime_revision = saved_revision
            self.runtime_revision = current_revision
            self.notebook_state.record_runtime_restored(
                current_revision=current_revision,
                saved_revision=saved_revision,
                commit_source=committed_source,
                degraded_recovery=degraded_recovery,
                attempted_revision=attempted_revision,
            )
            if degraded_recovery:
                logger.warning(
                    "[%s] Restored committed runtime revision %d after newer "
                    "revision %d failed to persist; recovery is degraded",
                    self.session_id,
                    committed_revision,
                    attempted_revision,
                )
            logger.info("Session restored successfully")
        else:
            # P1-11: dill restore failure leaves the kernel without previously-saved
            # variables while the notebook metadata has already been loaded. Continuing
            # would yield silent NameErrors downstream. Surface the failure loudly and
            # abort start-up so the workflow can react (retry / fresh kernel / restore
            # from earlier snapshot).
            self._session_restore_failed = True
            # Mark of how much work the kernel had done when persistence was
            # latched off. Cheap short-circuit only: a kernel that has run
            # nothing since cannot have rebuilt anything -- see save_session.
            self._restore_failed_at_revision = getattr(self, "runtime_revision", 0)
            # What the aborted restore left the namespace holding. This, not the
            # cell counter, is what save_session compares against: a counter
            # cannot tell "the agent reloaded the data and recomputed the
            # analysis" from "one `import scanpy as sc` ran", and both advance it
            # by exactly one. Taken here because the kernel is still alive at
            # this point; None means it could not be read, and save_session then
            # keeps refusing rather than guessing.
            self._restore_failed_globals = await self._snapshot_persistable_globals()
            detail = result.error or result.output or "unknown error"
            self.notebook_state.update_runtime_persistence(
                "failed",
                current_revision=current_revision,
                saved_revision=saved_revision,
                failed_variables=sorted(self.get_runtime_dependency_names()),
                error_summary=detail,
            )
            logger.error(
                f"[{self.session_id}] dill session restore failed: {detail}"
            )
            raise RuntimeError(
                f"Sandbox restore failed for session {self.session_id}: {detail}. "
                f"Previous kernel variable state is unavailable. Re-run the analysis "
                f"from your data to rebuild kernel state; snapshot restore is only "
                f"available where Snapshot V2 is enabled."
            )
    
    async def save_session(self):
        """Save the current session state by streaming dill per-variable.

        Each variable is an independent dill frame written through a streaming
        zstd writer. Compression therefore does not build a full in-memory pickle,
        while repetitive sparse/dataframe state no longer expands to multi-GB raw
        files. Required-variable failure leaves the previous atomic file intact.

        The whole user namespace is a candidate, not a fixed name list. Two tiers:

        * **Required** — tracker-declared names, canonical artifacts that are
          live, and *every* AnnData regardless of its variable name. These are
          fail-closed: missing or unpersistable aborts the save and keeps the
          previous pkl.
        * **Everything else** — saved best-effort. Values that cannot be pickled
          (open loom/h5py handles, sockets, locks) are skipped and reported via
          ``_last_save_skipped`` instead of failing the save, since one stray
          file handle must not cost the user their entire runtime state.

        Modules, functions, classes and methods are excluded — kernel init and
        notebook replay re-establish them, and by-reference frames are what most
        often break a restore in a different environment.

        A kernel whose restore failed is never persisted. ``_restore_session``
        populates ``globals()`` frame by frame, so a mid-stream failure leaves a
        partial namespace; dumping that over the live pkl and re-signing it
        destroys the only good copy and makes the *next* start succeed with a
        silently truncated runtime, still reporting ``runtime_restore_status`` as
        complete. The pkl on disk is better evidence than anything this kernel
        now holds.

        Normal start and restart paths now discard a partially restored kernel.
        The latch remains defense in depth for legacy/manual recovery paths that
        may still hand save_session a live restore-failed namespace. If such a
        path explicitly rebuilds the analysis from source data, refusing forever
        would keep that recomputation from disk and the next resume would silently
        rehydrate the pre-crash pkl instead.

        The evidence that it has rebuilt is what the namespace now *holds*, not
        how many cells have run. ``runtime_revision`` advances on any successful
        cell, so requiring it to pass the mark once meant a single
        ``import scanpy as sc`` disarmed the guard: modules are never written to
        the pkl, so that save dumped a namespace with no analysis state in it
        over the last good copy, and the next resume rehydrated it and reported
        ``runtime_restore_status`` complete. The comparison is instead against
        the persistable globals the aborted restore left behind
        (``_restore_failed_globals``); only values this kernel added on top of
        that residue count as a rebuild.
        """
        async with self._exec_lock:
            # Select the live generation only after waiting for any in-flight
            # execution.  The same lock owns serialization and host commit
            # below, so the selected revision remains immutable throughout.
            restore_failed = bool(getattr(self, "_session_restore_failed", False))
            mark = getattr(self, "_restore_failed_at_revision", None)
            revision = getattr(self, "runtime_revision", 0)
            saved_revision = getattr(self, "saved_runtime_revision", -1)
            previous_persistence = dict(
                getattr(self.notebook_state, "runtime_persistence", {}) or {}
            )
            expected_lifecycle_generation = int(
                self._expected_lifecycle_generation
            )

            from backend.services.session_lifecycle import (
                SessionGenerationError,
                SessionGoneError,
                assert_session_generation_active,
                lifecycle_file_lock,
            )

            def _restore_previous_persistence() -> None:
                self.saved_runtime_revision = int(
                    previous_persistence.get("saved_revision") or 0
                )
                if hasattr(self.notebook_state, "runtime_persistence"):
                    self.notebook_state.runtime_persistence = dict(
                        previous_persistence
                    )

            def _record_generation_failure(
                error_summary: str,
                *,
                failed_variables=None,
                budget_violations=None,
                top_offenders=None,
                recommended_action=None,
                requires_recompute=False,
                recompute_variables=None,
            ) -> bool:
                """Persist one bounded failure fact for the frozen generation."""
                _restore_previous_persistence()
                try:
                    with lifecycle_file_lock(self.session_id):
                        assert_session_generation_active(
                            self.session_id,
                            expected_generation=expected_lifecycle_generation,
                        )
                        updater = getattr(
                            self.notebook_state,
                            "update_runtime_persistence",
                            None,
                        )
                        if callable(updater):
                            updater(
                                "failed",
                                current_revision=revision,
                                saved_revision=self.saved_runtime_revision,
                                failed_variables=list(failed_variables or []),
                                error_summary=error_summary,
                                budget_violations=list(budget_violations or []),
                                top_offenders=list(top_offenders or []),
                                recommended_action=recommended_action,
                                requires_recompute=requires_recompute,
                                recompute_variables=list(
                                    recompute_variables or []
                                ),
                            )
                except (SessionGenerationError, SessionGoneError) as lifecycle_error:
                    _restore_previous_persistence()
                    logger.warning(
                        "[%s] Suppressed runtime failure fact for frozen "
                        "lifecycle generation %d (%s)",
                        self.session_id,
                        expected_lifecycle_generation,
                        type(lifecycle_error).__name__,
                    )
                    return False
                return True

            has_committed_dump = getattr(self.notebook_state, "has_session_dump", None)
            if (
                not restore_failed
                and revision == saved_revision
                and (
                    getattr(self.notebook_state, "runtime_persistence", {}) or {}
                ).get("commit_source") == "regular_save"
                and callable(has_committed_dump)
                and has_committed_dump()
            ):
                logger.debug(
                    "[%s] Reusing integrity-validated runtime generation %d",
                    self.session_id,
                    revision,
                )
                return True
            if restore_failed and (mark is None or revision <= mark):
                # Cheap short-circuit: nothing has run, so nothing can have
                # been rebuilt, and the kernel need not be touched to establish
                # that.  This decision is deliberately made under the lock.
                logger.error(
                    "[%s] refusing to persist a kernel whose restore failed and "
                    "which has executed nothing since; keeping the previous pkl",
                    self.session_id,
                )
                _record_generation_failure(
                    "Runtime restore failed and no recomputation has completed.",
                    failed_variables=sorted(self.get_runtime_dependency_names()),
                )
                return False
            session_path = str(self.notebook_state.session_path.absolute())
            escaped_path = repr(session_path)

            if restore_failed:
                rebuilt = await self._rebuilt_since_restore_failure()
                if not rebuilt:
                    logger.error(
                        "[%s] refusing to persist a kernel whose restore failed: "
                        "%d cell(s) have run since, but the namespace still holds "
                        "nothing this kernel built, so the previous pkl stays the "
                        "better evidence and is kept",
                        self.session_id,
                        revision - mark,
                    )
                    _record_generation_failure(
                        "Runtime restore failed and the live namespace has not "
                        "rebuilt required continuation state.",
                        failed_variables=sorted(self.get_runtime_dependency_names()),
                    )
                    return False
                logger.warning(
                    "[%s] restore failed earlier; the kernel has since rebuilt "
                    "%d value(s) (%s), persisting the rebuilt namespace over the "
                    "previous pkl",
                    self.session_id,
                    len(rebuilt),
                    ", ".join(sorted(rebuilt)[:8]),
                )
                # The latch is cleared only once the write has actually landed --
                # see the end of this method. Clearing it here would drop the
                # protection permanently the first time a save raised.

            from backend.utils.runtime_budget import (
                RuntimeBudgetExceeded,
                RuntimeBudgetViolation,
                evaluate_runtime_budget,
            )

            budgets = dict(getattr(self, "runtime_budgets", {}) or {})
            resource_metrics: Dict[str, Any] = {"top_offenders": []}

            def _record_budget_failure(violations, *, error_summary: str) -> None:
                serialized = [
                    item.to_dict() if hasattr(item, "to_dict") else dict(item)
                    for item in violations
                ]
                _record_generation_failure(
                    error_summary,
                    budget_violations=serialized,
                    top_offenders=resource_metrics.get("top_offenders") or [],
                    recommended_action="reduce_runtime_footprint",
                )

            if budgets:
                resource_metrics = await self._runtime_resource_metrics()
                required_metrics = {
                    "namespace_count",
                    "largest_object_bytes",
                    "anndata_total_bytes",
                    "rss_bytes",
                    "backed_anndata_total_bytes",
                    "projected_materialized_rss_bytes",
                }
                missing_metrics = sorted(required_metrics - set(resource_metrics))
                if missing_metrics:
                    measurement_failure = [{
                        "resource": "measurement",
                        "observed": "unavailable",
                        "limit": ",".join(missing_metrics),
                    }]
                    _record_budget_failure(
                        measurement_failure,
                        error_summary=(
                            "Runtime resource measurement failed; persistence was "
                            "stopped before serialization."
                        ),
                    )
                    raise RuntimeError(
                        "Sandbox runtime resource measurement failed for session "
                        f"{self.session_id}: {', '.join(missing_metrics)}"
                    )
                preflight_violations = evaluate_runtime_budget(
                    resource_metrics, budgets
                )
                if any(
                    violation.resource == "namespace_count"
                    for violation in preflight_violations
                ):
                    pruned = await self.prune_unrequired_runtime_globals()
                    if pruned:
                        logger.warning(
                            "[%s] Pruned %d unconsumed scratch global(s) after "
                            "runtime namespace pressure: %s",
                            self.session_id,
                            len(pruned),
                            ", ".join(pruned[:20]),
                        )
                    resource_metrics = await self._runtime_resource_metrics()
                    preflight_violations = evaluate_runtime_budget(
                        resource_metrics, budgets
                    )
                if preflight_violations:
                    _record_budget_failure(
                        preflight_violations,
                        error_summary="Runtime resource budget exceeded before serialization.",
                    )
                    raise RuntimeBudgetExceeded(
                        preflight_violations,
                        resource_metrics.get("top_offenders"),
                    )

            logger.info(f"Saving session to {session_path}")

            # Compatibility cleanup for dependency trackers written by older
            # execute() code. That code could mark a declared output as required
            # even when a swallowed exception meant the name never existed. Only
            # prune missing names that were never observed as created; genuinely
            # tracked cross-cell dependencies remain fail-closed below.
            live_names = await self._snapshot_user_globals()
            stale_names = self._prune_legacy_unobserved_required(live_names)
            if stale_names:
                logger.warning(
                    "[%s] Pruned legacy required outputs never observed in the "
                    "kernel: %s",
                    self.session_id,
                    sorted(stale_names),
                )

            # Canonical names only matter when they actually exist; a canonical
            # name that is simply absent is not an error. Tracker-declared names
            # stay fail-closed whether or not they are live (see _missing_required).
            from backend.services.snapshot_v2.runtime_registry import (
                CANONICAL_RUNTIME_ARTIFACTS,
            )
            required_names = set(self.get_runtime_dependency_names())
            required_names |= set(CANONICAL_RUNTIME_ARTIFACTS) & set(live_names)
            escaped_required = repr(sorted(required_names))

            save_code = """\
import dill
import json as _scai_json
import os
import tempfile
import types
import zstandard as zstd

_required_names = set(""" + escaped_required + """)


def _scai_is_anndata(_o):
    try:
        _t = type(_o)
        return _t.__name__ == "AnnData" and "anndata" in (_t.__module__ or "").lower()
    except Exception:
        return False


# Handles and connections are the dangerous case: dill often "succeeds" on them
# by recording a path or fd, then reopens it on restore. On another machine that
# path may not exist, and a raising frame aborts the whole restore -- so an open
# loom/h5py/sqlite handle could cost the user every other variable. Exclude by
# type rather than waiting for a pickling error that may never come.
_scai_volatile_prefixes = (
    "_io.", "io.", "socket.", "_socket.", "ssl.",
    "_thread.", "threading.", "multiprocessing.", "concurrent.", "asyncio.",
    "sqlite3.", "h5py.", "tables.", "loompy.", "zarr.",
    "subprocess.", "tempfile.",
)


def _scai_is_volatile(_o):
    _t = type(_o)
    _q = ((_t.__module__ or "") + "." + _t.__name__).lower()
    return _q.startswith(_scai_volatile_prefixes)


# Modules, functions, classes and methods are re-established by kernel init and
# notebook replay. They bloat the dump, and a by-reference frame is the single
# most likely thing to fail on restore in a different environment.
_scai_skip_kinds = (
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    type,
)

_live = {}
for _scai_name in list(globals().keys()):
    if not _scai_name.startswith('_'):
        _live[_scai_name] = globals()[_scai_name]

# Every AnnData is required no matter what it is called. Matching canonical
# names alone silently dropped things like `adata_hepatocyte`, which is exactly
# the state a resumed analysis cannot recompute.
#
# A backed AnnData wraps an open h5py handle that dill cannot pickle. It is
# still authoritative runtime state: silently removing it and committing the
# remaining namespace made the metadata claim `current` while a real analysis
# object had disappeared. Materialize one in-memory copy per object identity
# for the dump, preserving aliases. If materialization fails, abort the whole
# generation so the runtime-persistence gate remains armed on the live kernel.
_scai_materialized_backed = {}
for _scai_name, _scai_obj in list(_live.items()):
    if not _scai_is_anndata(_scai_obj):
        continue
    if getattr(_scai_obj, 'isbacked', False):
        _scai_object_id = id(_scai_obj)
        try:
            _scai_memory_obj = _scai_materialized_backed.get(_scai_object_id)
            if _scai_memory_obj is None:
                _scai_memory_obj = _scai_obj.to_memory()
                if getattr(_scai_memory_obj, 'isbacked', False):
                    raise RuntimeError('AnnData.to_memory() returned a backed object')
                _scai_materialized_backed[_scai_object_id] = _scai_memory_obj
            _live[_scai_name] = _scai_memory_obj
        except Exception as _e:
            raise RuntimeError(
                "backed AnnData could not be materialized for persistence: "
                + _scai_name + " (" + type(_e).__name__ + ")"
            ) from _e
    _required_names.add(_scai_name)

_missing_required = sorted(_n for _n in _required_names if _n not in _live)
if _missing_required:
    raise RuntimeError(
        "required runtime variables are missing: " + ", ".join(_missing_required)
    )

# Required values first so a failure aborts before any optional bytes are
# written; the rest of the namespace follows on a best-effort basis.
_to_dump = []
for _scai_name in sorted(_live):
    if _scai_name in _required_names:
        _to_dump.append((_scai_name, _live[_scai_name], True))
for _scai_name in sorted(_live):
    _scai_obj = _live[_scai_name]
    if _scai_name in _required_names:
        continue
    if isinstance(_scai_obj, _scai_skip_kinds) or _scai_is_volatile(_scai_obj):
        continue
    _to_dump.append((_scai_name, _scai_obj, False))

_skipped_vars = []
_saved_object_names = {}
_saved_names = []

# Write to a temp file.  The host signs and installs it together with committed
# generation metadata; kernel code must never replace the authoritative pkl.
_tmp_path = """ + escaped_path + """ + '.tmp'
_saved_count = 0
_compression_magic = b"SCAI_ZSTD_DILL_V1\\n"
try:
    with open(_tmp_path, 'wb') as _raw:
        _raw.write(_compression_magic)
        with zstd.ZstdCompressor(level=1).stream_writer(_raw, closefd=False) as _f:
            for _var_name, _obj, _is_required in _to_dump:
                _object_id = id(_obj)
                if _object_id in _saved_object_names:
                    _frame = (
                        "__SCAI_ALIAS__", _var_name,
                        _saved_object_names[_object_id],
                    )
                else:
                    _frame = (_var_name, _obj)
                # Required values may stream because any failure aborts the
                # generation. Optional values first serialize into a spooled
                # file, bounding memory without trusting a library's entire
                # top-level namespace to be dill-safe.
                _streamed = _is_required
                try:
                    if _streamed:
                        dill.dump(_frame, _f)
                    else:
                        with tempfile.SpooledTemporaryFile(
                            max_size=8 * 1024 * 1024, mode='w+b'
                        ) as _frame_file:
                            dill.dump(_frame, _frame_file)
                            _frame_file.seek(0)
                            # Serialization is complete. A failure from this
                            # point may have partially modified the compressed
                            # stream, so it must abort the generation.
                            _streamed = True
                            while True:
                                _frame_chunk = _frame_file.read(1024 * 1024)
                                if not _frame_chunk:
                                    break
                                _f.write(_frame_chunk)
                except Exception as _e:
                    _skipped_vars.append((
                        _var_name,
                        type(_obj).__name__,
                        type(_e).__name__,
                        str(_e)[:200],
                    ))
                    _sv = _skipped_vars[-1]
                    print(f"__SCAI_SAVE_SKIPPED__\\t{_sv[0]}\\t{_sv[1]}\\t{_sv[2]}\\t{_sv[3]}")
                    if _streamed:
                        raise RuntimeError(
                            ("required runtime variable could not be persisted: "
                             if _is_required else
                             "streamed runtime variable failed mid-write, "
                             "aborting to avoid a desynced dump: ") + _var_name
                        ) from _e
                    continue
                _saved_object_names.setdefault(_object_id, _var_name)
                _saved_names.append(_var_name)
                _saved_count += 1
except Exception:
    try:
        os.remove(_tmp_path)
    except OSError:
        pass
    raise

print(f"Session saved: {_saved_count} variables")
print("__SCAI_SAVED_NAMES__:" + _scai_json.dumps(sorted(_saved_names)))
"""
            save_started = time.monotonic()
            save_timeout = budgets.get("max_save_seconds") if budgets else None
            execute_kwargs = {
                "save_to_history": False,
                "skip_safety_check": True,
            }
            if save_timeout is not None and float(save_timeout) > 0:
                execute_kwargs["execution_timeout"] = float(save_timeout)
            result = await self._execute_unlocked(save_code, **execute_kwargs)

            save_elapsed = time.monotonic() - save_started
            if (
                not result.success
                and save_timeout is not None
                and float(save_timeout) > 0
                and getattr(result, "timed_out", False) is True
            ):
                staged_timeout_path = Path(str(self.notebook_state.session_path) + ".tmp")
                staged_timeout_path.unlink(missing_ok=True)
                resource_metrics["save_seconds"] = save_elapsed
                timeout_violation = RuntimeBudgetViolation(
                    "save_seconds", save_elapsed, float(save_timeout)
                )
                _record_budget_failure(
                    [timeout_violation],
                    error_summary=(
                        "Runtime serialization exceeded its configured duration "
                        "budget; the previous generation was retained."
                    ),
                )
                raise RuntimeBudgetExceeded([timeout_violation])

            skipped = []
            for line in (result.output or "").splitlines():
                if line.startswith("__SCAI_SAVE_SKIPPED__\t"):
                    parts = line.split("\t")
                    if len(parts) >= 5:
                        skipped.append({
                            "name": parts[1],
                            "type": parts[2],
                            "exc": parts[3],
                            "msg": parts[4],
                        })
            self._last_save_skipped = skipped

            saved_names: set[str] = set()
            for line in (result.output or "").splitlines():
                if not line.startswith("__SCAI_SAVED_NAMES__:"):
                    continue
                try:
                    parsed_names = json.loads(line.split(":", 1)[1])
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(parsed_names, list):
                    saved_names = {str(name) for name in parsed_names}

            if result.success:
                if skipped:
                    # Best-effort tier only: a required failure would have made
                    # result.success False. Surfaced so a later restore that is
                    # missing a value has a breadcrumb explaining why.
                    logger.info(
                        "[%s] Session saved; %d unpicklable value(s) skipped: %s",
                        self.session_id,
                        len(skipped),
                        ", ".join(f"{sv['name']}({sv['type']})" for sv in skipped),
                    )
                pkl_path = Path(self.notebook_state.session_path)
                staged_path = Path(str(pkl_path) + ".tmp")
                production_transaction = staged_path.exists()
                if budgets:
                    measured_path = staged_path if production_transaction else pkl_path
                    resource_metrics["pkl_bytes"] = (
                        measured_path.stat().st_size if measured_path.exists() else 0
                    )
                    resource_metrics["save_seconds"] = time.monotonic() - save_started
                    postflight_violations = evaluate_runtime_budget(
                        resource_metrics, budgets
                    )
                    if postflight_violations:
                        staged_path.unlink(missing_ok=True)
                        _record_budget_failure(
                            postflight_violations,
                            error_summary=(
                                "Runtime serialization exceeded its configured "
                                "size or duration budget; the previous generation "
                                "was retained."
                            ),
                        )
                        raise RuntimeBudgetExceeded(
                            postflight_violations,
                            resource_metrics.get("top_offenders"),
                        )
                integrity_path = _integrity_dir(self.session_id)

                def _cleanup_unjournaled_products() -> None:
                    """Remove only this attempt's unjournaled staged pkl."""
                    journal_path = pkl_path.with_name(
                        f"{pkl_path.name}.regular-save.json"
                    )
                    if journal_path.exists():
                        return
                    staged_path.unlink(missing_ok=True)

                manifest_path = None
                target_digest = None
                if production_transaction:
                    from backend.utils.runtime_apply import runtime_h5ad_manifest_path

                    manifest_path = runtime_h5ad_manifest_path(
                        self.session_id,
                        states_dir=self.notebook_state.state_dir,
                    )
                    try:
                        target_digest = await _run_durable_host_operation(
                            lambda: _sha256_file(staged_path),
                            session_id=self.session_id,
                            description="runtime digest calculation",
                        )
                    except BaseException as exc:
                        _cleanup_unjournaled_products()
                        _record_generation_failure(
                            "Runtime digest calculation did not complete: "
                            f"{type(exc).__name__}."
                        )
                        raise

                def _commit_generation_manifest():
                    if not production_transaction:
                        return
                    from backend.utils.runtime_apply import (
                        write_runtime_h5ad_manifest,
                    )

                    write_runtime_h5ad_manifest(
                        self.session_id,
                        {},
                        authoritative_pkl_names=saved_names,
                        states_dir=self.notebook_state.state_dir,
                    )

                def _retire_generation_sidecars():
                    from backend.utils.runtime_apply import (
                        quarantine_unreferenced_runtime_h5ads,
                    )

                    quarantine_unreferenced_runtime_h5ads(
                        self.session_id,
                        set(),
                        states_dir=self.notebook_state.state_dir,
                    )

                def _commit_generation_metadata():
                    self.saved_runtime_revision = revision
                    updater = getattr(
                        self.notebook_state, "update_runtime_persistence", None
                    )
                    if callable(updater):
                        updater(
                            "current",
                            current_revision=revision,
                            saved_revision=revision,
                            commit_source="regular_save",
                            runtime_digest=target_digest,
                        )

                commit_outcome: Dict[str, Any] = {
                    "completed": False,
                    "error": None,
                }

                try:
                    def _commit_on_host() -> None:
                        try:
                            with lifecycle_file_lock(self.session_id):
                                assert_session_generation_active(
                                    self.session_id,
                                    expected_generation=expected_lifecycle_generation,
                                )
                                if staged_path.exists():
                                    _commit_pkl_transaction(
                                        staged_path,
                                        pkl_path,
                                        integrity_path,
                                        _commit_generation_metadata,
                                        metadata_path=Path(self.notebook_state.state_path),
                                        manifest_path=manifest_path,
                                        commit_manifest=_commit_generation_manifest,
                                        retire_sidecars=_retire_generation_sidecars,
                                        target_revision=revision,
                                        authoritative_pkl_names=saved_names,
                                        fault_hook=getattr(
                                            self, "_regular_save_fault_hook", None
                                        ),
                                    )
                                else:
                                    # Compatibility for test doubles and pre-transaction
                                    # custom kernels. Production save code always leaves
                                    # the staged path and never reaches this branch.
                                    _write_pkl_signature(pkl_path, integrity_path)
                                    assert_session_generation_active(
                                        self.session_id,
                                        expected_generation=expected_lifecycle_generation,
                                    )
                                    _commit_generation_metadata()
                        except BaseException as exc:
                            commit_outcome["error"] = exc
                            raise
                        else:
                            commit_outcome["completed"] = True

                    await _run_durable_host_operation(
                        _commit_on_host,
                        session_id=self.session_id,
                        description="runtime generation commit",
                    )
                    logger.info("Session saved successfully")
                except BaseException as exc:
                    worker_error = commit_outcome.get("error")
                    if isinstance(exc, asyncio.CancelledError):
                        if commit_outcome.get("completed"):
                            # The worker published a complete generation; preserve
                            # its metadata before re-delivering cancellation.
                            raise
                        _cleanup_unjournaled_products()
                        if isinstance(worker_error, SessionGoneError):
                            _restore_previous_persistence()
                        else:
                            _record_generation_failure(
                                "Runtime generation commit was cancelled after "
                                "the durable worker reached its terminal boundary."
                            )
                        raise

                    _cleanup_unjournaled_products()
                    if isinstance(exc, SessionGoneError):
                        # A tombstoned generation must not write new metadata.
                        _restore_previous_persistence()
                        raise
                    if isinstance(exc, SessionGenerationError):
                        _record_generation_failure(
                            "Runtime generation rejected by the session lifecycle."
                        )
                        raise
                    _record_generation_failure(
                        f"Runtime commit failed: {type(exc).__name__}"
                    )
                    raise RuntimeError(
                        f"Sandbox runtime commit failed for session {self.session_id}"
                    ) from exc
            else:
                if skipped:
                    logger.error(
                        "[%s] save_session rejected incomplete runtime; required "
                        "variables not persisted: %s",
                        self.session_id,
                        [sv["name"] for sv in skipped],
                    )
                logger.warning(f"Failed to save session: {result.error}")
                failed_names = {
                    sv["name"] for sv in skipped if sv.get("name") in required_names
                }
                error_text = str(result.error or "")
                failed_names.update(
                    name for name in required_names if name in error_text
                )
                poisoned: set[str] = set()
                tracker = getattr(self, "runtime_dependency_tracker", None)
                if tracker is not None and failed_names:
                    poisoned = tracker.record_persistence_failure(failed_names)
                _record_generation_failure(
                    result.error or "Runtime serialization failed.",
                    failed_variables=sorted(failed_names),
                    requires_recompute=bool(poisoned),
                    recompute_variables=sorted(poisoned),
                )
                raise RuntimeError(
                    f"Sandbox runtime save failed for session {self.session_id}: "
                    f"{result.error or 'unknown error'}"
                )

            # A rebuilt namespace is now on disk, so the pkl the latch was
            # protecting no longer exists to protect. Clearing here rather than
            # at the guard keeps the protection in place across a save that
            # raised, which would otherwise disarm it permanently.
            if getattr(self, "_session_restore_failed", False):
                self._session_restore_failed = False
                self._restore_failed_at_revision = None
                self._restore_failed_globals = None

            return True

    async def _snapshot_persistable_globals(self) -> set[str] | None:
        """User globals ``save_session`` would actually dump, or None if unknown.

        The same exclusion the save code applies via ``_scai_skip_kinds``:
        modules, functions, classes and methods are re-established by kernel init
        and notebook replay and never reach the pkl, so a namespace holding only
        those holds no analysis state at all. ``None`` is deliberately distinct
        from the empty set -- "the kernel could not tell us" and "the kernel has
        nothing" lead to the same refusal, but only one of them is a fact about
        the namespace.
        """
        code = (
            "import json as _pg_json, types as _pg_types\n"
            "_pg_kinds = (_pg_types.ModuleType, _pg_types.FunctionType,\n"
            "             _pg_types.BuiltinFunctionType, _pg_types.MethodType, type)\n"
            "print('__PERSISTABLE_GLOBALS__:' + _pg_json.dumps(sorted(\n"
            "    _n for _n, _v in list(globals().items())\n"
            "    if not _n.startswith('_') and not isinstance(_v, _pg_kinds)\n"
            ")))\n"
        )
        result = await self._execute_unlocked(
            code, save_to_history=False, skip_safety_check=True
        )
        if not result.success:
            return None
        for line in (result.output or "").splitlines():
            if line.startswith("__PERSISTABLE_GLOBALS__:"):
                raw = line[len("__PERSISTABLE_GLOBALS__:"):].strip()
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    return None
                if isinstance(parsed, list):
                    return {str(x) for x in parsed}
        return None

    async def _rebuilt_since_restore_failure(self) -> set[str]:
        """Values this kernel produced since its restore failed.

        Empty means "no evidence of a rebuild", which includes the case where
        the baseline was never measured or the kernel cannot answer now: with
        nothing to compare against, the pkl on disk stays the better evidence.
        """
        baseline = getattr(self, "_restore_failed_globals", None)
        if baseline is None:
            return set()
        live = await self._snapshot_persistable_globals()
        if live is None:
            return set()
        return live - set(baseline)











    
    
    
    
    
    

    
    
    
    




# Convenience function to create sandbox
async def create_sandbox(
    session_id: str,
    state_dir: str = "states",
    working_dir: str = "workspace"
) -> LocalJupyterSandbox:
    """Create and start a local Jupyter sandbox"""
    sandbox = LocalJupyterSandbox(
        session_id=session_id,
        state_dir=state_dir,
        working_dir=working_dir
    )
    await sandbox.start()
    return sandbox
