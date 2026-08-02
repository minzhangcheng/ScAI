# Code execution against the kernel and figure collection.
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
    _to_thread_drained,
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


class ExecutionMixin:
    """Code execution against the kernel and figure collection."""

    def _allowed_write_roots(self) -> List[str]:
        """Return the list of write-root paths rendered into the kernel init code,
        so host-side tests can assert that a path is NOT writable by the sandbox."""
        return [
            str(self.working_dir.resolve()),
            str(self.notebook_state.state_dir.resolve()),
            str(self._session_tmp.resolve()),
        ]

    async def execute(
        self,
        code: str,
        save_to_history: bool = True,
        skip_safety_check: bool = False,
        declared_outputs: Optional[List[str]] = None,
        track_globals: bool = False,
    ) -> CodeExecutionResult:
        """Execute code in the kernel.

        Args:
            code: Python code to execute
            save_to_history: Whether to save successful execution to notebook history
            skip_safety_check: If True, skip dangerous-pattern validation.
                Only set this for trusted internal code (e.g. kernel init, session restore).
            declared_outputs: Optional persistent_outputs declared by the agent
                tool call. Every declared name must exist after successful code
                execution before it is recorded as a required runtime output.
            track_globals: When True (and save_to_history), record created/loaded
                user-global diffs into the runtime dependency tracker.
        """
        declared = set(declared_outputs or [])
        confirmed_declared: set[str] = set()
        before: set[str] = set()
        should_track = bool(track_globals or declared) and save_to_history
        async with self._exec_lock:
            if should_track:
                before = await self._snapshot_user_globals()
            result = await self._execute_unlocked(
                code, save_to_history=save_to_history, skip_safety_check=skip_safety_check
            )
            if should_track:
                after = await self._snapshot_user_globals()
                # A failed cell can still create or delete globals before the
                # exception. Always reconcile the tracker with the live
                # namespace so removed outputs cannot poison later saves.
                # Explicit outputs become required only after a successful
                # execution actually produced them.
                confirmed_declared = declared & after if result.success else set()
                self._record_globals_diff(
                    before,
                    after,
                    declared=confirmed_declared,
                    loaded=self._loaded_global_names_from_code(code),
                )
                missing_declared = declared - after if result.success else set()
                if missing_declared:
                    missing_text = ", ".join(sorted(missing_declared))
                    contract_error = (
                        "Declared persistent outputs were not created: "
                        f"{missing_text}. Ensure every persistent_outputs name "
                        "is assigned before the cell completes."
                    )
                    result.success = False
                    result.error = (
                        f"{result.error}\n{contract_error}"
                        if result.error else contract_error
                    )
        if result.success and save_to_history:
            self.runtime_revision += 1
            if confirmed_declared:
                self.record_declared_outputs(confirmed_declared)
                tracker = getattr(self, "runtime_dependency_tracker", None)
                if tracker is not None:
                    tracker.confirm_recomputed(
                        confirmed_declared,
                        reason=(
                            "Successful execution explicitly re-produced the "
                            "declared persistent output."
                        ),
                    )
            self.notebook_state.update_runtime_persistence(
                "live_dirty",
                current_revision=self.runtime_revision,
                saved_revision=self.saved_runtime_revision,
            )
        return result

    async def _execute_unlocked(
        self,
        code: str,
        save_to_history: bool = True,
        skip_safety_check: bool = False
    ) -> CodeExecutionResult:
        """Execute code in the kernel (unlocked — caller must hold _exec_lock).

        Args:
            code: Python code to execute
            save_to_history: Whether to save successful execution to notebook history
            skip_safety_check: If True, skip dangerous-pattern validation.
                Only set this for trusted internal code (e.g. kernel init, session restore).
        """
        # Check if kernel is dead (failed restart)
        if self._kernel_dead:
            return CodeExecutionResult(
                success=False,
                output="",
                error="Kernel is unavailable (restart failed). Please restart the analysis session.",
                figures=[]
            )

        # Validate code safety before execution (unless explicitly skipped for internal code)
        if not skip_safety_check:
            is_safe, error_msg = validate_code_safety(code)
            if not is_safe:
                logger.warning(f"Code blocked by safety check for session {self.session_id}: {error_msg}")
                return CodeExecutionResult(
                    success=False,
                    error=f"Code execution blocked: {error_msg}"
                )

        if self.kc is None:
            await self.start()

        # Guard against silent kernel death (e.g. OOM kill, libzmq abort).
        # km.is_alive() checks whether the kernel subprocess is still running.
        # Without this check, kc.execute() on a dead kernel triggers ZMQ channel
        # corruption that calls abort() inside libzmq, killing the entire process.
        if self.km is not None and not self.km.is_alive():
            logger.warning(
                f"[{self.session_id}] Kernel process is dead before execute; "
                f"attempting restart"
            )
            await self._restart_kernel()
            if self._kernel_dead:
                return CodeExecutionResult(
                    success=False,
                    output="",
                    error="Kernel died unexpectedly and restart failed. Please restart the analysis session.",
                    figures=[]
                )

        outputs = []
        figures = []
        output_text = []
        error_msg = None
        execution_count = 0
        
        try:
            # Send execute request
            msg_id = self.kc.execute(code)

            # Collect outputs — only process messages belonging to THIS execution (matched by msg_id).
            # Timeout is wall-clock: the deadline is fixed before the loop so a task that
            # keeps streaming output (tqdm/scanpy verbose) cannot reset it indefinitely.
            deadline = time.monotonic() + self.timeout
            while True:
                remaining = compute_remaining_timeout(deadline, time.monotonic())
                msg = None
                if remaining > 0:
                    try:
                        # Pass timeout directly to get_iopub_msg so the thread
                        # returns after the ZMQ-level poll expires.  Previously
                        # asyncio.wait_for(asyncio.to_thread(get_iopub_msg)) was
                        # used, but that only cancels the *coroutine* — the
                        # underlying thread stays blocked on ZMQ recv forever.
                        # Each timeout leaked an orphaned reader thread; multiple
                        # threads reading the same ZMQ socket is unsafe and
                        # eventually triggers libzmq abort (src/msg.cpp:387).
                        msg = await _to_thread_drained(
                            self.kc.get_iopub_msg,
                            timeout=min(remaining, IOPUB_POLL_CAP)
                        )
                    except QueueEmpty:
                        # ZMQ poll timed out — no message arrived within the
                        # poll cap.  Re-check the wall-clock deadline and
                        # kernel liveness before looping or declaring timeout.
                        if self.km and not self.km.is_alive():
                            logger.warning(
                                f"[{self.session_id}] Kernel died during execution; "
                                f"aborting output collection"
                            )
                            await self._restart_kernel()
                            if self._kernel_dead:
                                return CodeExecutionResult(
                                    success=False,
                                    output="",
                                    error="Kernel died during execution and restart failed.",
                                    figures=[]
                                )
                            # Kernel restarted successfully; the old execution is gone,
                            # return an error so the caller can retry.
                            return CodeExecutionResult(
                                success=False,
                                output="",
                                error="Kernel died during execution. Restarted successfully, but the execution was lost.",
                                figures=[]
                            )
                        if compute_remaining_timeout(deadline, time.monotonic()) > 0:
                            continue
                        remaining = 0  # fall through to timeout handling below

                if remaining <= 0:
                    error_msg = f"Execution timed out after {self.timeout} seconds"
                    # Interrupt the kernel to stop the running code
                    if self.km:
                        try:
                            self.km.interrupt_kernel()
                            logger.warning(f"Kernel interrupted after timeout for session {self.session_id}")
                            # Wait briefly for kernel to acknowledge interrupt
                            await asyncio.sleep(0.5)
                            # Drain any remaining messages to clear the queue
                            try:
                                while True:
                                    drain_msg = await _to_thread_drained(
                                        self.kc.get_iopub_msg, timeout=1.0
                                    )
                                    if drain_msg['msg_type'] == 'status' and drain_msg['content']['execution_state'] == 'idle':
                                        break
                            except QueueEmpty:
                                # Kernel may be in bad state, restart it
                                logger.warning(f"Kernel did not return to idle after interrupt for session {self.session_id}")
                                await self._restart_kernel()
                        except Exception as interrupt_error:
                            logger.error(f"Failed to interrupt kernel: {interrupt_error}")
                    break

                # Skip messages from other execution requests (e.g. leftover from session restore)
                parent_msg_id = msg.get('parent_header', {}).get('msg_id', '')
                if parent_msg_id and parent_msg_id != msg_id:
                    continue

                msg_type = msg['msg_type']
                content = msg['content']

                if msg_type == 'status':
                    if content['execution_state'] == 'idle':
                        break

                elif msg_type == 'execute_input':
                    execution_count = content.get('execution_count', 0)

                elif msg_type == 'stream':
                    text = content.get('text', '')
                    output_text.append(text)
                    outputs.append({
                        "type": "stream",
                        "name": content.get('name', 'stdout'),
                        "text": text
                    })

                elif msg_type == 'execute_result':
                    data = content.get('data', {})
                    if 'text/plain' in data:
                        output_text.append(data['text/plain'])
                    outputs.append({
                        "type": "execute_result",
                        "data": data
                    })

                elif msg_type == 'display_data':
                    data = content.get('data', {})
                    # Extract images
                    if 'image/png' in data:
                        figures.append(data['image/png'])
                    outputs.append({
                        "type": "display_data",
                        "data": data
                    })

                elif msg_type == 'error':
                    error_msg = '\n'.join(content.get('traceback', []))
                    outputs.append({
                        "type": "error",
                        "ename": content.get('ename', 'Error'),
                        "evalue": content.get('evalue', ''),
                        "traceback": content.get('traceback', [])
                    })
            
            # Save to history
            if save_to_history and not error_msg:
                self.notebook_state.add_cell(code, outputs, execution_count)
                self.notebook_state.save()
            
            return CodeExecutionResult(
                success=error_msg is None,
                output='\n'.join(output_text),
                error=error_msg,
                figures=figures,
                execution_count=execution_count
            )
            
        except Exception as e:
            logger.exception("Execution error")
            if _is_kernel_protocol_error(e):
                logger.warning(
                    f"Kernel protocol corruption detected for session {self.session_id}; "
                    f"restarting kernel"
                )
                await self._restart_kernel()
            return CodeExecutionResult(
                success=False,
                error=str(e)
            )

    async def execute_and_get_figures(self, code: str) -> Tuple[str, List[str]]:
        """Execute code and return output and figures"""
        result = await self.execute(code)
        return result.output, result.figures
