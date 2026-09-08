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
import copy
import time
import hmac
import hashlib
import logging
import tempfile
from queue import Empty as QueueEmpty
from typing import Optional, Dict, Any, List, Tuple, Final
from datetime import datetime
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_output
from jupyter_client import KernelManager

from backend.utils.crypto import get_integrity_secret, sign_hmac_sha256
from backend.config._config import DATA_BASE_DIR

# Cap on a single iopub poll so the wall-clock deadline is re-checked at least
# this often even while the kernel keeps streaming output (tqdm/scanpy verbose).
IOPUB_POLL_CAP = 30.0
_REGULAR_SAVE_COPY_CHUNK_BYTES: Final[int] = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Session pkl HMAC integrity helpers (host-side only)
# ---------------------------------------------------------------------------

from .regular_save import (  # noqa: F401
    GENERATION_PAIR_DIRNAME,
    paired_generation_sources,
    _sig_path,
    _pkl_hmac,
    _pkl_hmac_stream,
    _write_pkl_signature,
    _sha256_file,
    _regular_save_journal_path,
    _regular_save_backup_path,
    _journal_body,
    _write_regular_save_journal,
    _read_regular_save_journal,
    _copy_durable,
    _remove_durable,
    _regular_component,
    _regular_component_paths,
    _preflight_regular_rollback,
    _restore_regular_components,
    _regular_metadata_matches,
    _cleanup_regular_save_transaction,
    recover_regular_save_transaction,
    _commit_pkl_transaction,
    _verify_pkl_integrity,
)


def _integrity_dir(sid: str) -> Path:
    return DATA_BASE_DIR / "integrity" / sid


def _path_is_within(path: str, root: str) -> bool:
    resolved = str(Path(path).resolve())
    root_resolved = str(Path(root).resolve()).rstrip(os.sep)
    return resolved == root_resolved or resolved.startswith(root_resolved + os.sep)


from .code_safety import (  # noqa: F401
    compute_remaining_timeout,
    _to_thread_drained,
    logger,
    _is_kernel_protocol_error,
    DANGEROUS_PATTERNS,
    _is_safe_os_system,
    _is_safe_subprocess,
    _is_safe_importlib,
    _literal_is_safe,
    _literal_path,
    _is_safe_file_removal,
    _is_safe_os_remove,
    _is_safe_os_unlink,
    _is_safe_os_rmdir,
    _is_safe_file_write,
    _EXCEPTION_CHECKS,
    _DANGEROUS_AST_FUNCS,
    _DANGEROUS_AST_ATTRS,
    _DANGEROUS_AST_MODULES,
    _SUBPROCESS_EXEC_FUNCS,
    _OS_PROCESS_EXEC_FUNCS,
    _CALL_SITE_PATTERN_KEYS,
    _mask_prose,
    _match_context,
    _validate_code_safety_ast,
    validate_code_safety,
    extract_code_from_markdown,
)
from .notebook_state import (  # noqa: F401
    _NOTEBOOK_INLINE_IMAGE_LIMIT_BYTES,
    _NOTEBOOK_TEXT_OUTPUT_LIMIT_CHARS,
    _NOTEBOOK_HTML_OUTPUT_LIMIT_CHARS,
    _NOTEBOOK_TRACEBACK_LIMIT_LINES,
    _NOTEBOOK_OUTPUT_OMISSION,
    _NOTEBOOK_HTML_OMISSION,
    _NOTEBOOK_TRACEBACK_OMISSION,
    CodeExecutionResult,
    NotebookState,
)


# Template for unwinding any stale SCAI no-op stubs from plt.close/plt.show
# and installing fresh ones.  Embedded into f-string code templates via
# {_TPL_NOOP_INSTALL}.  The string must NOT contain { or } to be f-string safe.
_TPL_NOOP_INSTALL = (
    "# Temporarily disable plt.close/plt.show to keep figures alive.\n"
    "# Unwind any leftover SCAI no-op stubs from prior runs before installing fresh ones.\n"
    "_unwrap_close_guard = 0\n"
    "while getattr(plt.close, '_scai_noop', False) and _unwrap_close_guard < 64:\n"
    "    plt.close = getattr(plt.close, '_scai_prev', plt.close)\n"
    "    _unwrap_close_guard += 1\n"
    "_unwrap_show_guard = 0\n"
    "while getattr(plt.show, '_scai_noop', False) and _unwrap_show_guard < 64:\n"
    "    plt.show = getattr(plt.show, '_scai_prev', plt.show)\n"
    "    _unwrap_show_guard += 1\n"
    "# Defensive baseline restore for legacy sessions with unmarked stubs.\n"
    "try:\n"
    "    if '_original_plt_close' in globals() and plt.close is not _original_plt_close \\\n"
    "            and not getattr(plt.close, '_scai_noop', False):\n"
    "        plt.close = _original_plt_close\n"
    "except Exception:\n"
    "    pass\n"
    "try:\n"
    "    if '_capturing_show' in globals() and plt.show is not _capturing_show \\\n"
    "            and not getattr(plt.show, '_scai_noop', False):\n"
    "        plt.show = _capturing_show\n"
    "except Exception:\n"
    "    pass\n"
    "_orig_close = plt.close\n"
    "_orig_show = plt.show\n"
    "def _noop_close(*a, **kw): return None\n"
    "_noop_close._scai_noop = True\n"
    "_noop_close._scai_prev = _orig_close\n"
    "def _noop_show(*a, **kw): return None\n"
    "_noop_show._scai_noop = True\n"
    "_noop_show._scai_prev = _orig_show\n"
    "plt.close = _noop_close\n"
    "plt.show = _noop_show\n"
)
