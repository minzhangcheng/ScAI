# Dill state persistence and file upload into the workspace.
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


class PersistenceMixin:
    """File upload into the workspace.

    The dill persistence this class was named for lives in
    ``sandbox_parts/runtime_save.py`` and commits through
    ``regular_save._commit_pkl_transaction``. What used to be here was a
    save-coalescing path -- ``save_state``, ``save_state_async``,
    ``_schedule_coalesced_save`` and ``await_pending_saves`` -- that no
    production code ever called. Its task set was therefore always empty, which
    made ``await_pending_saves`` a drain that drained nothing while reading, at
    its one call site in ``kernel.stop``, like an ordering guarantee. The
    guarantee is ``_exec_lock``, and it is stated there now. If coalescing is
    wanted again it should return together with the caller that wants it.
    """

    def upload_file(self, local_path: str, filename: Optional[str] = None) -> str:
        """Copy a file to the sandbox working directory"""
        import shutil
        src = Path(local_path)
        dst = self.working_dir / (filename or src.name)
        shutil.copy(src, dst)
        logger.info(f"Uploaded file to sandbox: {dst}")
        return str(dst)
