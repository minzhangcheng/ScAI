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
import signal
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


IOPUB_INTERRUPT_DRAIN_TIMEOUT = 5.0


_CORE_DUMPING_EXITS = frozenset(
    -int(member)
    for member in (
        getattr(signal, name, None)
        for name in ("SIGSEGV", "SIGABRT", "SIGBUS", "SIGFPE", "SIGILL", "SIGQUIT")
    )
    if member is not None
)


def describe_kernel_exit(kernel_manager: Any) -> str:
    """Return a safe, factual description of a terminated kernel process."""
    try:
        process = getattr(getattr(kernel_manager, "provisioner", None), "process", None)
        if process is None:
            return "exit status unavailable (no kernel process handle)"
        # jupyter_client 8 exposes provisioner.poll() as a coroutine. The
        # process handle is the synchronous source of the POSIX exit status.
        code = process.poll()
        if code is None:
            return "process still present, no exit status yet"
        if code < 0:
            try:
                name = signal.Signals(-code).name
            except ValueError:
                name = f"signal {-code}"
            dumped = (
                " (core dump suppressed by RLIMIT_CORE)"
                if code in _CORE_DUMPING_EXITS
                else ""
            )
            return f"killed by {name}{dumped}"
        return f"exited with status {code}"
    except Exception as exc:  # pragma: no cover - failure diagnostics must not raise
        return f"exit status unavailable ({exc!r})"


async def _drain_iopub_until_idle(
    kernel_client: Any,
    timeout_seconds: float = IOPUB_INTERRUPT_DRAIN_TIMEOUT,
    *,
    monotonic=time.monotonic,
) -> bool:
    """Drain an interrupted execution under one wall-clock deadline."""
    deadline = monotonic() + max(0.0, float(timeout_seconds))
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            return False
        try:
            message = await _to_thread_drained(
                kernel_client.get_iopub_msg,
                timeout=min(1.0, remaining),
            )
        except QueueEmpty:
            return False
        if (
            message.get("msg_type") == "status"
            and message.get("content", {}).get("execution_state") == "idle"
        ):
            return True


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

    def _note_execution_effect(
        self, *, mutates_namespace: bool, reached_kernel: bool
    ) -> None:
        """Advance the runtime revision for anything that could change globals.

        Tying this to success made a failed cell's side effects invisible to
        persistence: the kernel does not roll a cell back, so a statement that
        mutated ``adata`` before raising leaves it mutated -- but the revision
        stayed put, and a save whose revision equals the saved revision reuses
        the previous generation instead of writing one. The mutation was then
        in the live namespace and in no dump.

        Read-only probes opt out explicitly rather than being inferred from
        ``records_history``. Inferring was wrong even before that flag was
        renamed: several mutating internal cells stayed out of the notebook
        (``adata_val = adata.copy()`` among them), so "not recorded" never
        implied "changed nothing".
        """
        if not mutates_namespace:
            return
        if not reached_kernel:
            # Blocked by the safety check, the kernel was dead, or the send
            # itself raised: no statement ran, so nothing changed. Passed in
            # rather than read from the instance, because the probes between
            # the execution and this call use the same field.
            return
        # Bookkeeping, not correctness: a sandbox still being constructed has no
        # revision yet, and refusing to execute over a missing counter would be
        # a worse failure than starting it here.
        self.runtime_revision = int(getattr(self, "runtime_revision", 0) or 0) + 1

    async def execute(
        self,
        code: str,
        records_history: bool = True,
        skip_safety_check: bool = False,
        declared_outputs: Optional[List[str]] = None,
        track_globals: bool = False,
        mutates_namespace: bool = True,
    ) -> CodeExecutionResult:
        """Execute code in the kernel.

        Args:
            code: Python code to execute
            records_history: Whether this execution belongs in the notebook.
                Default True: the notebook is a record of what the kernel ran,
                failures included. Read-only probes opt out explicitly. The same
                execution boundary carries `mutates_namespace`; the two are
                orthogonal, and one boundary with two flags is legible where two
                boundaries drift.
            skip_safety_check: If True, skip dangerous-pattern validation.
                Only set this for trusted internal code (e.g. kernel init, session restore).
            declared_outputs: Optional persistent_outputs declared by the agent
                tool call. Every declared name must exist after successful code
                execution before it is recorded as a required runtime output.
            track_globals: When True (and records_history), record created/loaded
                user-global diffs into the runtime dependency tracker.
            mutates_namespace: Whether this execution could change user globals.
                Read-only probes opt out; everything else defaults to True,
                because a cell that raised part-way still changed what it had
                already changed.
        """
        declared_raw = [
            str(name).strip()
            for name in (declared_outputs or [])
            if str(name).strip()
        ]
        declared = {name for name in declared_raw if name.isidentifier()}
        declared_not_names = sorted(
            {name for name in declared_raw if not name.isidentifier()}
        )
        before: set[str] = set()
        should_track = bool(track_globals or declared) and records_history
        async with self._exec_lock:
            if should_track:
                before = await self._snapshot_user_globals()
            result = await self._execute_unlocked(
                code, records_history=records_history, skip_safety_check=skip_safety_check
            )
            # Captured here, before the declared-output probes below run: they
            # call _execute_unlocked too, and would otherwise overwrite this
            # execution's verdict with their own.
            reached_kernel = bool(
                getattr(self, "_last_execution_reached_kernel", False)
            )
            if declared_not_names:
                result.output = (result.output or "") + (
                    "\n[Kernel] Ignored persistent_outputs entries that are "
                    "not Python variable names: "
                    + ", ".join(declared_not_names)
                    + ". persistent_outputs lists user-global names this cell "
                    "assigns, not files it writes -- a name carrying a path "
                    "or an extension can never appear in the namespace. Files "
                    "written under the working directory are recorded by the "
                    "host automatically; declare the in-memory object instead "
                    "(e.g. 'adata', not 'adata.h5ad')."
                )
            if should_track:
                loaded_names = self._loaded_global_names_from_code(code)
                # Whether a value can be serialized is no longer asked here.
                # Spec 3.11 shows unserialisable values arise on every run, and
                # spec 4.4 makes the answer coverage rather than a verdict: the
                # save records what it could not write, and spec section 10
                # carries that to the agent.
                after = await self._snapshot_user_globals()
                # A failed cell can still create or delete globals before the
                # exception, so the diff is reconciled either way.
                self._record_globals_diff(
                    before,
                    after,
                    declared=declared & after if result.success else set(),
                    loaded=loaded_names,
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
        self._note_execution_effect(
            mutates_namespace=mutates_namespace, reached_kernel=reached_kernel
        )
        if result.success and records_history:
            self.notebook_state.update_runtime_persistence(
                "live_dirty",
                current_revision=self.runtime_revision,
                saved_revision=self.saved_runtime_revision,
            )
        return result

    async def _execute_unlocked(
        self,
        code: str,
        records_history: bool = True,
        skip_safety_check: bool = False,
        execution_timeout: Optional[float] = None,
    ) -> CodeExecutionResult:
        """Execute code in the kernel (unlocked — caller must hold _exec_lock).

        Args:
            code: Python code to execute
            records_history: Whether this execution belongs in the notebook.
                Default True: the notebook is a record of what the kernel ran,
                failures included. Read-only probes opt out explicitly. The same
                execution boundary carries `mutates_namespace`; the two are
                orthogonal, and one boundary with two flags is legible where two
                boundaries drift.
            skip_safety_check: If True, skip dangerous-pattern validation.
                Only set this for trusted internal code (e.g. kernel init, session restore).
            execution_timeout: Optional wall-clock limit for this execution.
                Internal operations such as runtime persistence use their own
                stricter deadline instead of inheriting the general sandbox timeout.
        """
        # Nothing has reached the kernel yet. Every early return below leaves
        # this False, so a request the kernel never saw cannot mark the runtime
        # dirty and force a multi-gigabyte save of an unchanged namespace.
        self._last_execution_reached_kernel = False

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
            exit_reason = describe_kernel_exit(self.km)
            logger.warning(
                f"[{self.session_id}] Kernel process is dead before execute "
                f"({exit_reason}); attempting restart"
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
        timed_out = False
        
        try:
            # Send execute request
            msg_id = self.kc.execute(code)
            # Only after the request is accepted. Setting it beforehand counted
            # a send that raised as code having run.
            self._last_execution_reached_kernel = True

            # Collect outputs — only process messages belonging to THIS execution (matched by msg_id).
            # Timeout is wall-clock: the deadline is fixed before the loop so a task that
            # keeps streaming output (tqdm/scanpy verbose) cannot reset it indefinitely.
            timeout_seconds = (
                float(execution_timeout)
                if execution_timeout is not None
                else float(self.timeout)
            )
            if timeout_seconds <= 0:
                raise ValueError("execution_timeout must be positive")
            deadline = time.monotonic() + timeout_seconds
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
                            exit_reason = describe_kernel_exit(self.km)
                            logger.warning(
                                f"[{self.session_id}] Kernel died during execution "
                                f"({exit_reason}); aborting output collection"
                            )
                            await self._restart_kernel()
                            if self._kernel_dead:
                                return CodeExecutionResult(
                                    success=False,
                                    output="",
                                    error=(
                                        "Kernel died during execution and restart "
                                        f"failed ({exit_reason})."
                                    ),
                                    figures=[]
                                )
                            # Kernel restarted successfully; the old execution is gone,
                            # return an error so the caller can retry.
                            return CodeExecutionResult(
                                success=False,
                                output="",
                                error=(
                                    "Kernel died during execution "
                                    f"({exit_reason}). Restarted successfully, "
                                    "but the execution was lost."
                                ),
                                figures=[]
                            )
                        if compute_remaining_timeout(deadline, time.monotonic()) > 0:
                            continue
                        remaining = 0  # fall through to timeout handling below

                if remaining <= 0:
                    timed_out = True
                    error_msg = f"Execution timed out after {timeout_seconds} seconds"
                    # Interrupt the kernel to stop the running code
                    if self.km:
                        try:
                            self.km.interrupt_kernel()
                            logger.warning(f"Kernel interrupted after timeout for session {self.session_id}")
                            # Wait briefly for kernel to acknowledge interrupt
                            await asyncio.sleep(0.5)
                            # Continuous output without an idle status is as
                            # unhealthy as a silent kernel, so the entire drain
                            # has one wall-clock budget.
                            returned_idle = await _drain_iopub_until_idle(self.kc)
                            if not returned_idle:
                                logger.warning(
                                    "Kernel did not return to idle within %.1fs "
                                    "after interrupt for session %s",
                                    IOPUB_INTERRUPT_DRAIN_TIMEOUT,
                                    self.session_id,
                                )
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
            
            # A cell that raised is part of what the kernel ran, so it is
            # part of the record (spec 3.10). `save_notebook` already builds the
            # `error` output type; the traceback was constructed above and then
            # thrown away here. A cell the safety check rejected is *not*
            # recorded -- it returned before reaching the kernel, so there is
            # nothing to replay.
            if records_history:
                self.notebook_state.add_cell(code, outputs, execution_count)
                self.notebook_state.save()
            
            return CodeExecutionResult(
                success=error_msg is None,
                output='\n'.join(output_text),
                error=error_msg,
                figures=figures,
                execution_count=execution_count,
                timed_out=timed_out,
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
