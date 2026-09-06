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

    # Enough to reach the handle inside an ordinary scratch value, far too
    # little to serialize an analysis object by accident.
    # Containers whose persistability is not in question and whose size is.

    def mark_runtime_committed(self, runtime_revision: int | None = None) -> None:
        """Called after a snapshot capture transaction commits the runtime generation.

        Sets ``saved_runtime_revision`` to *runtime_revision* (or the current
        ``runtime_revision`` if omitted), marking the point at which the last
        successful commit occurred.
        """
        if runtime_revision is None:
            runtime_revision = self.runtime_revision
        self.saved_runtime_revision = runtime_revision

    def runtime_persistence_snapshot(self) -> Dict[str, Any]:
        """Return the persisted runtime lifecycle for API/state synchronization."""
        return dict(self.notebook_state.runtime_persistence)

    def sync_runtime_persistence_state(self, state) -> Dict[str, Any]:
        """Copy the authoritative notebook lifecycle into AnalysisState."""
        snapshot = self.runtime_persistence_snapshot()
        state.runtime_persistence = snapshot
        sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
        sandbox_state["runtime_revision"] = self.runtime_revision
        sandbox_state["saved_runtime_revision"] = self.saved_runtime_revision
        sandbox_state["runtime_persistence"] = snapshot
        state.sandbox_state = sandbox_state
        return snapshot

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

    async def _runtime_resource_metrics(self) -> Dict[str, Any]:
        """Measure bounded namespace/object/AnnData/RSS signals in the kernel."""
        code = r'''
import json as _scai_json
import os as _scai_os
import sys as _scai_sys

def _scai_dense_root(_value):
    _root = _value
    _guard = 0
    while getattr(_root, "base", None) is not None and _guard < 64:
        _next = getattr(_root, "base", None)
        if _next is _root:
            break
        _root = _next
        _guard += 1
    return _root

def _scai_dense_bytes(_value, _seen_buffers):
    _root = _scai_dense_root(_value)
    _key = id(_root)
    # Backed arrays can be returned as a fresh proxy on every attribute access.
    # Their stable allocation identity is the file plus dataset path, not the
    # short-lived Python proxy id.
    try:
        _file = getattr(_root, "file", None)
        _filename = getattr(_file, "filename", None)
        _dataset_name = getattr(_root, "name", None)
        if _filename and _dataset_name:
            _key = ("backed", _scai_os.path.realpath(str(_filename)), str(_dataset_name))
    except Exception:
        pass
    if _key in _seen_buffers:
        return 0
    _seen_buffers.add(_key)
    # Count the backing allocation, not a view's smaller logical window. This
    # is an RSS/persistence budget estimate and the base keeps those bytes live.
    _nbytes = getattr(_root, "nbytes", getattr(_value, "nbytes", None))
    if _nbytes is None:
        try:
            _count = 1
            for _dimension in getattr(_root, "shape", ()):
                _count *= int(_dimension)
            _nbytes = _count * int(getattr(getattr(_root, "dtype", None), "itemsize", 0))
        except Exception:
            _nbytes = 0
    return int(_nbytes or 0)

def _scai_array_bytes(_value, _seen_buffers):
    # Dense ndarrays expose ``.data`` as a memoryview of the same allocation;
    # adding both is a systematic 2x error. Sparse matrices instead own three
    # distinct dense buffers, so count those buffers individually.
    if all(hasattr(_value, _attr) for _attr in ("data", "indices", "indptr")):
        return sum(
            _scai_dense_bytes(getattr(_value, _attr), _seen_buffers)
            for _attr in ("data", "indices", "indptr")
        )
    if hasattr(_value, "nbytes") or (
        hasattr(_value, "shape") and hasattr(_value, "dtype")
    ):
        return _scai_dense_bytes(_value, _seen_buffers)
    return 0

def _scai_table_bytes(_value, _seen_buffers):
    _size = 0
    try:
        for _block in getattr(getattr(_value, "_mgr", None), "blocks", ()):
            _size += _scai_array_bytes(getattr(_block, "values", None), _seen_buffers)
    except Exception:
        pass
    try:
        _size += _scai_array_bytes(getattr(_value.index, "values", None), _seen_buffers)
    except Exception:
        pass
    return _size

def _scai_shallow_overhead(_value):
    # AnnData.__sizeof__ is deep: it already walks X/layers/raw/obsm/etc. Adding
    # the known buffers below therefore counted real objects twice. Call the
    # base implementation explicitly for the container shell only.
    if type(_value).__name__ == "AnnData":
        try:
            return int(object.__sizeof__(_value) or 0)
        except Exception:
            return 0
    try:
        return int(_scai_sys.getsizeof(_value, 0) or 0)
    except Exception:
        return 0

def _scai_size(_value, _seen_buffers=None):
    if _seen_buffers is None:
        _seen_buffers = set()
    _size = _scai_shallow_overhead(_value)
    if type(_value).__name__ == "AnnData":
        _x = getattr(_value, "X", None)
        if _x is not None:
            _size += _scai_array_bytes(_x, _seen_buffers)
        for _table_name in ("obs", "var"):
            _size += _scai_table_bytes(
                getattr(_value, _table_name, None), _seen_buffers
            )
        _raw = getattr(_value, "raw", None)
        if _raw is not None:
            _raw_x = getattr(_raw, "X", None)
            if _raw_x is not None:
                _size += _scai_array_bytes(_raw_x, _seen_buffers)
            _size += _scai_table_bytes(getattr(_raw, "var", None), _seen_buffers)
            try:
                for _part in getattr(_raw, "varm", {}).values():
                    _size += _scai_array_bytes(_part, _seen_buffers)
            except Exception:
                pass
        for _mapping_name in ("layers", "obsm", "varm", "obsp", "varp"):
            _mapping = getattr(_value, _mapping_name, None)
            if _mapping is None:
                continue
            try:
                for _part in _mapping.values():
                    _size += _scai_array_bytes(_part, _seen_buffers)
            except Exception:
                pass
    else:
        _known_buffers = _scai_array_bytes(_value, _seen_buffers)
        _known_buffers += _scai_table_bytes(_value, _seen_buffers)
        _size = max(_size, _known_buffers)
    return max(0, _size)

_scai_rows = []
_scai_anndata_total = 0
_scai_backed_anndata_total = 0
_scai_seen_globals = {}
_scai_global_seen_buffers = set()
_scai_public_globals = [
    (_name, _value)
    for _name, _value in list(globals().items())
    if not _name.startswith("_")
]
# Attribute shared backing allocations to their owning AnnData container before
# standalone ndarray/sparse aliases. This makes the total independent of global
# insertion order while the one shared seen-set prevents double accounting.
_scai_public_globals.sort(
    key=lambda _item: (type(_item[1]).__name__ != "AnnData", _item[0])
)
for _scai_name, _scai_value in _scai_public_globals:
    _scai_alias_of = _scai_seen_globals.get(id(_scai_value))
    if _scai_alias_of is None:
        try:
            _scai_bytes = _scai_size(
                _scai_value, _scai_global_seen_buffers
            )
        except Exception:
            _scai_bytes = 0
        _scai_seen_globals[id(_scai_value)] = _scai_name
    else:
        _scai_bytes = 0
    _scai_type = type(_scai_value).__name__
    _scai_row = {"name": _scai_name, "type": _scai_type, "bytes": _scai_bytes}
    if _scai_alias_of is not None:
        _scai_row["alias_of"] = _scai_alias_of
    _scai_rows.append(_scai_row)
    if _scai_type == "AnnData" and _scai_alias_of is None:
        _scai_anndata_total += _scai_bytes
        if bool(getattr(_scai_value, "isbacked", False)):
            _scai_backed_anndata_total += _scai_bytes
try:
    _scai_pages = int(open("/proc/self/statm").read().split()[1])
    _scai_rss = _scai_pages * int(_scai_os.sysconf("SC_PAGE_SIZE"))
except Exception:
    _scai_rss = 0
_scai_rows.sort(key=lambda _row: _row["bytes"], reverse=True)
print("__SCAI_RUNTIME_METRICS__:" + _scai_json.dumps({
    "namespace_count": len(_scai_rows),
    "largest_object_bytes": (_scai_rows[0]["bytes"] if _scai_rows else 0),
    "anndata_total_bytes": _scai_anndata_total,
    "backed_anndata_total_bytes": _scai_backed_anndata_total,
    "rss_bytes": _scai_rss,
    "projected_materialized_rss_bytes": _scai_rss + _scai_backed_anndata_total,
    "top_offenders": _scai_rows[:20],
    "measurement_basis": "estimated_known_buffers_identity_deduplicated",
}))
'''
        result = await self._execute_unlocked(
            code, save_to_history=False, skip_safety_check=True
        )
        if not result.success:
            return {"top_offenders": []}
        for line in (result.output or "").splitlines():
            if not line.startswith("__SCAI_RUNTIME_METRICS__:"):
                continue
            try:
                payload = json.loads(line.split(":", 1)[1])
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict):
                return payload
        return {"top_offenders": []}

    def _record_globals_diff(
        self,
        before: set[str],
        after: set[str],
        declared: set[str] | None = None,
        loaded: set[str] | None = None,
    ) -> Dict[str, Any]:
        """Record a created/loaded/removed user-global diff.

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
        return self._last_execution_diff

    @staticmethod
    def _loaded_global_names_from_code(code: str) -> set[str]:
        """Best-effort prior globals read by a cell; dynamic reads must be declared.

        A plain ``ast.walk`` cannot distinguish a prior global from a name that
        the same cell rebinds before reading it.  That promoted scratch loop
        variables and context-manager targets (notably closed loompy/h5py
        handles) into required cross-cell state.  Follow module execution order
        for the binding forms commonly emitted by analysis code instead.
        """
        import ast

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()

        loaded: set[str] = set()

        def _target_names(node: ast.AST | None) -> set[str]:
            if isinstance(node, ast.Name):
                return {node.id}
            if isinstance(node, (ast.Tuple, ast.List)):
                return set().union(*(_target_names(item) for item in node.elts))
            return set()

        def _loads(node: ast.AST | None, bound: set[str]) -> None:
            if node is None:
                return
            loaded.update(
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id not in bound
            )

        def _block(statements: list[ast.stmt], incoming: set[str]) -> set[str]:
            bound = set(incoming)
            for statement in statements:
                if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    _loads(getattr(statement, "value", None), bound)
                    targets = (
                        statement.targets
                        if isinstance(statement, ast.Assign)
                        else [statement.target]
                    )
                    for target in targets:
                        _loads(target, bound)
                        bound.update(_target_names(target))
                elif isinstance(statement, ast.AugAssign):
                    if isinstance(statement.target, ast.Name):
                        if statement.target.id not in bound:
                            loaded.add(statement.target.id)
                    else:
                        _loads(statement.target, bound)
                    _loads(statement.value, bound)
                    bound.update(_target_names(statement.target))
                elif isinstance(statement, (ast.With, ast.AsyncWith)):
                    body_bound = set(bound)
                    for item in statement.items:
                        _loads(item.context_expr, body_bound)
                        body_bound.update(_target_names(item.optional_vars))
                    bound = _block(statement.body, body_bound)
                elif isinstance(statement, (ast.For, ast.AsyncFor)):
                    _loads(statement.iter, bound)
                    _block(
                        statement.body,
                        bound | _target_names(statement.target),
                    )
                    _block(statement.orelse, bound)
                    # A zero-iteration loop does not bind its target afterward.
                elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                    bound.update(
                        alias.asname or alias.name.split(".", 1)[0]
                        for alias in statement.names
                    )
                elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Defaults, decorators, and annotations are evaluated while
                    # defining the function. Its body runs in a separate local
                    # scope: parameters and same-cell bindings must not be
                    # mistaken for reads of an older kernel global.
                    for decorator in statement.decorator_list:
                        _loads(decorator, bound)
                    for default in (
                        list(statement.args.defaults)
                        + [value for value in statement.args.kw_defaults if value]
                    ):
                        _loads(default, bound)
                    for argument in (
                        list(statement.args.posonlyargs)
                        + list(statement.args.args)
                        + list(statement.args.kwonlyargs)
                    ):
                        _loads(argument.annotation, bound)
                    if statement.args.vararg:
                        _loads(statement.args.vararg.annotation, bound)
                    if statement.args.kwarg:
                        _loads(statement.args.kwarg.annotation, bound)
                    _loads(statement.returns, bound)
                    function_bound = set(bound)
                    function_bound.add(statement.name)
                    function_bound.update(
                        argument.arg
                        for argument in (
                            list(statement.args.posonlyargs)
                            + list(statement.args.args)
                            + list(statement.args.kwonlyargs)
                        )
                    )
                    if statement.args.vararg:
                        function_bound.add(statement.args.vararg.arg)
                    if statement.args.kwarg:
                        function_bound.add(statement.args.kwarg.arg)
                    _block(statement.body, function_bound)
                    bound.add(statement.name)
                elif isinstance(statement, ast.If):
                    _loads(statement.test, bound)
                    _block(statement.body, bound)
                    _block(statement.orelse, bound)
                elif isinstance(statement, ast.While):
                    _loads(statement.test, bound)
                    _block(statement.body, bound)
                    _block(statement.orelse, bound)
                elif isinstance(statement, (ast.Try, ast.TryStar)):
                    _block(statement.body, bound)
                    for handler in statement.handlers:
                        _loads(handler.type, bound)
                        handler_bound = set(bound)
                        if handler.name:
                            handler_bound.add(handler.name)
                        _block(handler.body, handler_bound)
                    _block(statement.orelse, bound)
                    _block(statement.finalbody, bound)
                else:
                    # Other nested scopes stay conservative: they may read a
                    # prior global depending on their eventual execution.
                    _loads(statement, bound)
            return bound

        _block(tree.body, set())
        return loaded
