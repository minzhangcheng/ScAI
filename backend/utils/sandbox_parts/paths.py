# Workspace path accessors and restore-status flags.
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


class PathsMixin:
    """Workspace path accessors and restore-status flags."""

    def get_notebook_path(self) -> str:
        """Get path to the saved notebook"""
        return str(self.notebook_state.notebook_path)

    def get_session_path(self) -> str:
        """Get path to the session dump file"""
        return str(self.notebook_state.session_path)

    def get_all_state_files(self) -> List[str]:
        """Get all state-related files for export"""
        files = []
        if self.notebook_state.notebook_path.exists():
            files.append(str(self.notebook_state.notebook_path))
        if self.notebook_state.state_path.exists():
            files.append(str(self.notebook_state.state_path))
        if self.notebook_state.session_path.exists():
            files.append(str(self.notebook_state.session_path))
        return files

    def get_code_history(self) -> List[Dict[str, Any]]:
        """Get all executed code cells"""
        return self.notebook_state.cells

    def was_session_restored(self) -> bool:
        """Check if session was successfully restored from previous state"""
        return self._session_restored

    def did_session_restore_fail(self) -> bool:
        """Check if session restore was attempted but failed"""
        return self._session_restore_failed
