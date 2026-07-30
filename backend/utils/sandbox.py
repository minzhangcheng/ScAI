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
from typing import Optional, Dict, Any, List, Tuple
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
    _write_pkl_signature,
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
        timeout: int = 1200
    ):
        self.session_id = session_id
        import tempfile as _tmpfile_mod
        self._session_tmp = Path(_tmpfile_mod.gettempdir()) / f"scai-{self.session_id}"
        self._session_tmp.mkdir(parents=True, exist_ok=True)
        self.state_dir = state_dir
        # Caller passes session-specific working dir (e.g. sessions/{id}/workspace)
        self.working_dir = Path(working_dir).absolute()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        
        self.km: Optional[KernelManager] = None
        self.kc = None
        self.notebook_state = NotebookState(session_id, state_dir)
        self._initialized = False
        self.runtime_revision: int = 0
        self.saved_runtime_revision: int = 0
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
        recover_runtime_apply_journal(self.session_id)

        # Snapshot V2: unresolved APPLYING journals block kernel rehydrate
        from backend.services.session_store import assert_recovery_clear
        assert_recovery_clear(session_id=self.session_id)

        self.notebook_state.load()

        states_dir = Path(self.notebook_state.session_path).parent
        h5ad_manifest = states_dir / RUNTIME_H5AD_MANIFEST_FILENAME
        has_h5ad_manifest = h5ad_manifest.exists()

        if not self.notebook_state.has_session_dump() and not has_h5ad_manifest:
            logger.info("No session dump to restore")
            return

        # A present pkl with a missing/invalid HMAC is a hard error, not a silent
        # skip: continuing would rehydrate a kernel inconsistent with the state
        # that referenced this pkl. Raising lets the caller surface a typed
        # runtime-failure state (Stage 1 safety contract).
        if self.notebook_state.has_session_dump():
            try:
                verify_live_pkl_or_raise(self.session_id)
            except RuntimeApplyError:
                # Symmetric with the dill-restore-failure path below: mark the failure
                # so start()'s "restore_failed" result reflects it, then abort startup.
                self._session_restore_failed = True
                logger.error(
                    "[%s] live pkl present but HMAC invalid; aborting session restore",
                    self.session_id,
                )
                raise

        session_path = str(self.notebook_state.session_path.absolute())
        h5ad_manifest_path = str(h5ad_manifest.absolute())
        escaped_path = repr(session_path)
        escaped_h5ad_manifest_path = repr(h5ad_manifest_path)
        has_session_dump = self.notebook_state.has_session_dump()
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
            logger.info("Session restored successfully")
        else:
            # P1-11: dill restore failure leaves the kernel without previously-saved
            # variables while the notebook metadata has already been loaded. Continuing
            # would yield silent NameErrors downstream. Surface the failure loudly and
            # abort start-up so the workflow can react (retry / fresh kernel / restore
            # from earlier snapshot).
            self._session_restore_failed = True
            detail = result.error or result.output or "unknown error"
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
        """
        if getattr(self, "_session_restore_failed", False):
            logger.error(
                "[%s] refusing to persist a kernel whose restore failed; "
                "keeping the previous pkl",
                self.session_id,
            )
            return

        async with self._exec_lock:
            session_path = str(self.notebook_state.session_path.absolute())
            escaped_path = repr(session_path)
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
import os
import types
import zstandard as zstd

_required_names = set(""" + escaped_required + """)


def _scai_is_anndata(_o):
    try:
        _t = type(_o)
        return _t.__name__ == "AnnData" and "anndata" in (_t.__module__ or "").lower()
    except Exception:
        return False


def _scai_stream_safe(_o):
    # Numeric containers are both the largest values here and reliably
    # dill-able, so they stream straight through instead of being buffered.
    return (type(_o).__module__ or "").split(".")[0] in (
        "numpy", "pandas", "scipy", "anndata",
    )


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
for _scai_name, _scai_obj in _live.items():
    if _scai_is_anndata(_scai_obj):
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

# Write to a temp file then atomically replace, so a mid-write crash can't
# corrupt the only good pkl.
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
                # Streaming writes bytes as it goes, so a failure mid-frame
                # leaves the stream desynced and nothing after it can be read
                # back. Only values we are prepared to abort on may stream;
                # anything skippable is serialized in full first and written
                # only once it is known good.
                _streamed = _is_required or _scai_stream_safe(_obj)
                try:
                    if _streamed:
                        dill.dump(_frame, _f)
                    else:
                        _f.write(dill.dumps(_frame))
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
                _saved_count += 1
    os.replace(_tmp_path, """ + escaped_path + """)
except Exception:
    try:
        os.remove(_tmp_path)
    except OSError:
        pass
    raise

print(f"Session saved: {_saved_count} variables")
"""
            result = await self._execute_unlocked(save_code, save_to_history=False, skip_safety_check=True)

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
                logger.info("Session saved successfully")

                # Write HMAC signature (host-side, outside sandbox-writable roots)
                try:
                    pkl_path = Path(self.notebook_state.session_path)
                    if pkl_path.exists():
                        _write_pkl_signature(
                            pkl_path, _integrity_dir(self.session_id)
                        )
                except Exception:
                    logger.exception("Failed to write pkl integrity signature")
            else:
                if skipped:
                    logger.error(
                        "[%s] save_session rejected incomplete runtime; required "
                        "variables not persisted: %s",
                        self.session_id,
                        [sv["name"] for sv in skipped],
                    )
                logger.warning(f"Failed to save session: {result.error}")
                raise RuntimeError(
                    f"Sandbox runtime save failed for session {self.session_id}: "
                    f"{result.error or 'unknown error'}"
                )

            return True











    
    
    
    
    
    

    
    
    
    




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
