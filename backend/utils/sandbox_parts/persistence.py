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
    """Dill state persistence and file upload into the workspace."""

    async def _run_coalesced_save(self) -> bool:
        """Run the one shared save generation owned by this sandbox."""
        return await self.save_session()

    def _schedule_coalesced_save(self, loop: asyncio.AbstractEventLoop):
        """Return the active save task or schedule one for this revision."""
        active = getattr(self, "_pending_save_task", None)
        if active is not None and not active.done():
            return active

        task = loop.create_task(self._run_coalesced_save())
        self._pending_save_task = task
        self._pending_save_tasks.add(task)

        def _retire(completed):
            self._pending_save_tasks.discard(completed)
            if getattr(self, "_pending_save_task", None) is completed:
                self._pending_save_task = None

        task.add_done_callback(_retire)
        return task

    def save_state(self):
        """Save current state synchronously - only saves notebook, not session.

        Note: This method only reliably saves the notebook state. Session state
        (Python variables) may not be saved if called from sync context.
        Use save_state_async() for complete state persistence.
        """
        self.notebook_state.save()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - safe to create a new one
            try:
                asyncio.run(self.save_session())
            except Exception as e:
                logger.exception("Failed to save session state")
        else:
            self._schedule_coalesced_save(loop)

    async def save_state_async(self) -> None:
        """Save current state asynchronously through the shared generation."""
        self.notebook_state.save()
        task = self._schedule_coalesced_save(asyncio.get_running_loop())
        await asyncio.shield(task)

    async def await_pending_saves(self):
        """Wait for all pending save tasks to complete."""
        while self._pending_save_tasks:
            tasks = list(self._pending_save_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    def upload_file(self, local_path: str, filename: Optional[str] = None) -> str:
        """Copy a file to the sandbox working directory"""
        import shutil
        src = Path(local_path)
        dst = self.working_dir / (filename or src.name)
        shutil.copy(src, dst)
        logger.info(f"Uploaded file to sandbox: {dst}")
        return str(dst)
