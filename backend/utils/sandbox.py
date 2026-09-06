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
import sys
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
from .sandbox_parts import runtime_restore as _runtime_restore_owner
from .sandbox_parts import runtime_save as _runtime_save_owner


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
        self.notebook_state = NotebookState(
            session_id,
            state_dir,
            lifecycle_generation=self._expected_lifecycle_generation,
        )
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
        # Workflow-owned projection publisher. The sandbox remains the sole
        # runtime metadata authority and calls this only after a durable commit.
        self.runtime_persistence_commit_callback: Optional[Callable] = None
        self._init_exec_lock()



    
    
    
    async def _restore_session(self):
        """Restore session from per-variable streaming dill dump if available.

        New dumps contain a short format marker followed by a streaming zstd
        sequence of dill frames. Legacy uncompressed frame streams remain
        readable so existing sessions do not require migration.
        """
        return await _runtime_restore_owner.restore_session(
            self, bindings=sys.modules[__name__]
        )
    
    async def save_session(self):
        """Save the current session state by streaming dill per-variable.

        Each variable is an independent dill frame written through a streaming
        zstd writer. Compression therefore does not build a full in-memory pickle,
        while repetitive sparse/dataframe state no longer expands to multi-GB raw
        files. A failed save leaves the previous atomic file intact.

        The whole user namespace is a candidate, not a fixed name list, and no
        value is privileged. Every value serializes into a spooled file first,
        so one that dill refuses (an open loom/h5py handle, a socket, a lock) is
        skipped and reported via ``_last_save_skipped`` before a byte of it
        reaches the compressed stream. The generation still commits, and what
        was skipped travels with it as coverage — spec 4.4, where coverage warns
        and only integrity blocks. One stray file handle must not cost the user
        their entire runtime state, and it must not wedge the save either.

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
        return await _runtime_save_owner.save_session(
            self, bindings=sys.modules[__name__]
        )

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
