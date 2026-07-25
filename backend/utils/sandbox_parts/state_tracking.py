# Tracking of user globals, declared outputs, and runtime dependencies.
# Moved verbatim out of backend/utils/sandbox.py; composed into LocalJupyterSandbox.
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


# ---------------------------------------------------------------------------
# Session pkl HMAC integrity helpers (host-side only)
# ---------------------------------------------------------------------------

from .shared import (  # noqa: F401
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


class StateTrackingMixin:
    """Tracking of user globals, declared outputs, and runtime dependencies."""

    def mark_runtime_committed(self, runtime_revision: int | None = None) -> None:
        """Called after a snapshot capture transaction commits the runtime generation.

        Sets ``saved_runtime_revision`` to *runtime_revision* (or the current
        ``runtime_revision`` if omitted), marking the point at which the last
        successful commit occurred.
        """
        if runtime_revision is None:
            runtime_revision = self.runtime_revision
        self.saved_runtime_revision = runtime_revision

    def get_runtime_dependency_names(self) -> set[str]:
        """Return accumulated required runtime names from the dependency tracker."""
        tracker = self.runtime_dependency_tracker
        if tracker is None:
            return set()
        return set(tracker.required_outputs)

    def record_declared_outputs(self, names) -> None:
        """Merge explicit persistent_outputs into the dependency tracker."""
        if not names:
            return
        if self.runtime_dependency_tracker is None:
            from backend.services.snapshot_v2.runtime_registry import (
                RuntimeDependencyTracker,
            )
            self.runtime_dependency_tracker = RuntimeDependencyTracker()
        self.runtime_dependency_tracker.merge_declared(set(names))

    def _prune_legacy_unobserved_required(self, live_names: set[str]) -> set[str]:
        """Drop legacy declarations that were required but never created.

        Before declared-output verification was added, a successful tool call
        could swallow an exception and still poison the persisted tracker with
        a name that had never existed. Created dependencies are deliberately not
        pruned here: if one of those is unexpectedly absent, save_session must
        retain the fail-closed missing-runtime error.
        """
        tracker = self.runtime_dependency_tracker
        if tracker is None or not tracker.needs_legacy_declared_cleanup:
            return set()
        stale = (
            tracker.required_outputs
            - tracker.created_outputs
            - set(live_names)
        )
        if stale:
            tracker.record_execution(removed=stale)
        tracker.mark_legacy_declared_cleanup_complete()
        return stale

    async def _snapshot_user_globals(self) -> set[str]:
        """Return non-underscore user global names currently in the kernel."""
        code = (
            "import json as _jg\n"
            "_names = [n for n in list(globals().keys()) if not n.startswith('_')]\n"
            "print('__USER_GLOBALS__:' + _jg.dumps(sorted(_names)))\n"
        )
        result = await self._execute_unlocked(
            code, save_to_history=False, skip_safety_check=True
        )
        if not result.success:
            return set(self._known_user_globals)
        for line in (result.output or "").splitlines():
            if line.startswith("__USER_GLOBALS__:"):
                raw = line[len("__USER_GLOBALS__:"):].strip()
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        return set(str(x) for x in parsed)
                except (json.JSONDecodeError, TypeError):
                    pass
        return set(self._known_user_globals)

    def _record_globals_diff(
        self,
        before: set[str],
        after: set[str],
        declared: set[str] | None = None,
        loaded: set[str] | None = None,
    ) -> Dict[str, Any]:
        """Update tracker from a created/loaded user-global diff.

        - created: names present after but not before
        - loaded: names statically read by the executed cell and still present
        - removed: names present before but not after
        """
        created = after - before
        removed = before - after
        # ``loaded`` comes from the executed cell's AST. Treating the full
        # before/after intersection as "read" promoted every untouched scratch
        # variable on the next cell and recreated an all-globals snapshot sweep.
        loaded = (before & after) if loaded is None else (set(loaded) & before & after)
        self._last_execution_diff = {
            "created": set(created),
            "loaded": set(loaded),
            "removed": set(removed),
        }
        self._known_user_globals = set(after)
        if self.runtime_dependency_tracker is None:
            from backend.services.snapshot_v2.runtime_registry import (
                RuntimeDependencyTracker,
            )
            self.runtime_dependency_tracker = RuntimeDependencyTracker()
        # Only promote loaded names that were previously created by a cell.
        # Passing the full intersection would promote every pre-existing name.
        # `removed` names (del'd from the namespace) are pruned from the required
        # set so a declared-then-deleted transient (e.g. a loompy connection)
        # cannot permanently break snapshot capture.
        self.runtime_dependency_tracker.record_execution(
            created=created,
            loaded=loaded,
            declared=set(declared or set()),
            removed=removed,
        )
        return self._last_execution_diff

    @staticmethod
    def _loaded_global_names_from_code(code: str) -> set[str]:
        """Best-effort static names read by a cell; dynamic reads must be declared."""
        import ast

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        return {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
