# Jupyter kernel lifecycle: start, readiness, restart, stop.
# Moved verbatim out of backend/utils/sandbox.py; composed into LocalJupyterSandbox.
"""
Local Jupyter Kernel Executor with State Persistence

Uses dill.dump_session/load_session to save and restore the complete Python environment.
Supports pause/resume by saving and restoring the full session state.
"""

import os
import re
import json
import sys
import base64
import asyncio
import time
import hmac
import hashlib
import logging
from importlib.metadata import PackageNotFoundError, version as package_version
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

from . import kernel_bootstrap as _bootstrap_owner

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


_KERNEL_PROCESS_START_TIMEOUT = 30.0
_KERNEL_MANAGER_RESTART_TIMEOUT = 30.0
_KERNEL_MANAGER_SHUTDOWN_TIMEOUT = 10.0


def _dask_query_planning_setting() -> str:
    """Select the dataframe backend supported by the installed Dask release."""
    try:
        release = package_version("dask")
    except PackageNotFoundError:
        return "false"
    match = re.match(r"^(\d+)\.(\d+)", str(release))
    if match is None:
        return "false"
    return "true" if tuple(map(int, match.groups())) >= (2025, 1) else "false"


async def _start_kernel_manager(
    manager: KernelManager,
    *,
    cwd: str,
    env: Dict[str, str],
) -> None:
    """Start a kernel through Jupyter's native coroutine when available."""
    native = getattr(manager, "_async_start_kernel", None)
    if callable(native):
        await native(cwd=cwd, env=env)
        return
    manager.start_kernel(cwd=cwd, env=env)


async def _restart_kernel_manager(manager: KernelManager) -> None:
    """Restart a manager without blocking the event loop."""
    native = getattr(manager, "_async_restart_kernel", None)
    if callable(native):
        await native()
        return
    await _to_thread_drained(manager.restart_kernel)


async def _shutdown_kernel_manager(manager: KernelManager) -> None:
    """Shut down a manager through its native coroutine when available."""
    native = getattr(manager, "_async_shutdown_kernel", None)
    if callable(native):
        await native(now=True)
        return
    manager.shutdown_kernel(now=True)



def _bootstrap_notebook_cell(code: str, result) -> dict:
    """One bootstrap execution, in the shape `set_bootstrap_cells` stores.

    `CodeExecutionResult` carries the flattened stdout rather than the raw iopub
    outputs, so the stream is reconstructed here. The bootstrap raises on
    failure before this is reached, so there is no error branch to build.
    """
    outputs = []
    if result.output:
        outputs.append({"type": "stream", "name": "stdout", "text": result.output})
    return {
        "code": code,
        "outputs": outputs,
        "execution_count": result.execution_count or 0,
    }


class KernelMixin:
    """Jupyter kernel lifecycle: start, readiness, restart, stop."""

    def _init_exec_lock(self):
        """Single lock serializing every coroutine that drives the kernel client
        sockets (execute + save_session). Two coroutines reading the same ZMQ
        socket concurrently triggers libzmq abort (see src/msg.cpp warning)."""
        self._exec_lock = asyncio.Lock()

    def _kernel_launch_environment(self) -> Dict[str, str]:
        """Build an isolated environment before the kernel process starts.

        ``sitecustomize`` applies the irreversible process-level boundary, so
        cache and runtime locations must already point at per-session roots.
        """
        return _bootstrap_owner.build_kernel_launch_environment(
            self, bindings=sys.modules[__name__]
        )

    def _install_isolated_kernel_launcher(self, manager: KernelManager) -> None:
        """Wrap the kernelspec with a trusted pre-IPython isolation stage."""
        return _bootstrap_owner.install_isolated_kernel_launcher(
            self, manager, bindings=sys.modules[__name__]
        )

    async def start(self) -> dict:
        """Start the Jupyter kernel

        Returns:
            dict with keys:
                - started: bool - whether kernel was started (False if already running)
                - session_restored: bool - whether previous session was restored
                - restore_failed: bool - whether restore was attempted but failed
        """
        # Snapshot V2: unresolved recovery blocks kernel startup entirely.
        from backend.services.session_store import assert_recovery_clear
        assert_recovery_clear(session_id=self.session_id)

        if self.km is not None:
            return {
                "started": False,
                "session_restored": self._session_restored,
                "restore_failed": self._session_restore_failed
            }

        logger.info(f"Starting Jupyter kernel for session {self.session_id}")

        # Manager, process, channels, bootstrap, and restore are one resource-
        # acquisition unit. Any failure must tear down whatever this call has
        # acquired, including a process whose client/channel setup failed.
        self._starting = True
        try:
            self.km = KernelManager(kernel_name='python3')
            self._install_isolated_kernel_launcher(self.km)
            try:
                await asyncio.wait_for(
                    _start_kernel_manager(
                        self.km,
                        cwd=str(self.working_dir),
                        env=self._kernel_launch_environment(),
                    ),
                    timeout=_KERNEL_PROCESS_START_TIMEOUT,
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    "Kernel process failed to start within timeout"
                ) from exc
            self.kc = self.km.client()
            self.kc.start_channels()

            try:
                await asyncio.wait_for(
                    self._wait_for_ready(),
                    timeout=30
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError("Kernel failed to start within timeout") from exc

            # Readiness proves this is a new, reachable namespace. Clear a dead
            # latch inherited from the previous process so trusted bootstrap
            # code can pass _execute_unlocked's availability guard. Any later
            # failure sets it again in the outer handler.
            self._kernel_dead = False

            # Bootstrap failures must not leave a live, partially initialized
            # kernel behind: callers would otherwise see ``km``/``kc`` and
            # assume the namespace is usable.
            await self._initialize_kernel()

            # A freshly started+initialized kernel is alive regardless of any
            # prior dead state. Bump the namespace generation before restore;
            # a failed restore discards this generation below.
            self._kernel_dead = False
            self.kernel_epoch = getattr(self, "kernel_epoch", 0) + 1
            epoch_callback = getattr(self, "runtime_epoch_callback", None)
            if callable(epoch_callback):
                epoch_callback(self.kernel_epoch)

            # `_restore_session` assigns into globals() frame by frame, so any
            # error or cancellation leaves a partial namespace. save=False is
            # mandatory: that namespace must never reach the persisted dump.
            await self._restore_session()
        except BaseException:
            self._kernel_dead = True
            try:
                await self._shutdown_kernel_unlocked()
            except Exception:
                logger.exception(
                    "Failed to stop partially started kernel for %s",
                    self.session_id,
                )
            raise
        finally:
            self._starting = False

        return {
            "started": True,
            "session_restored": self._session_restored,
            "restore_failed": self._session_restore_failed
        }

    async def _wait_for_ready(self):
        """Wait for kernel to be ready"""
        return await _bootstrap_owner.wait_for_ready(
            self, bindings=sys.modules[__name__]
        )

    async def _initialize_kernel(self):
        """Initialize kernel with common setup"""
        if self._initialized:
            return

        init_code = _bootstrap_owner.build_common_bootstrap_source(self)
        result = await self._execute_unlocked(
            init_code, records_history=False, skip_safety_check=True
        )
        if result.output:
            logger.debug("Kernel init stdout: %s", result.output[:1000])
        if not result.success:
            logger.error(
                "Kernel common initialization failed for %s: %s",
                self.session_id,
                result.error or "unknown error",
            )
            raise RuntimeError("Kernel common initialization failed")

        path_setup_code = _bootstrap_owner.build_path_setup_source(self)
        path_result = await self._execute_unlocked(
            path_setup_code, records_history=False, skip_safety_check=True
        )
        if path_result.output:
            logger.debug("Kernel path-setup stdout: %s", path_result.output[:1000])
        if not path_result.success:
            logger.error(
                "Kernel path/sandbox initialization failed for %s: %s",
                self.session_id,
                path_result.error or "unknown error",
            )
            raise RuntimeError("Kernel path/sandbox initialization failed")

        # The environment every later cell assumes, recorded as the notebook's
        # opening cells (spec 5.3.2). These two still opt out of the ordinary
        # record gate: it saves, and this runs before restore has read the
        # notebook off disk, so an append here would overwrite a resumed
        # session's history with a two-cell file.
        self.notebook_state.set_bootstrap_cells([
            _bootstrap_notebook_cell(init_code, result),
            _bootstrap_notebook_cell(path_setup_code, path_result),
        ])

        self._initialized = True
        logger.info("Kernel initialized")

    async def _restart_kernel(self) -> None:
        """Restart the kernel, then restore the session fail-closed.

        Restart/readiness/bootstrap failure marks the sandbox dead. Restore also
        mutates globals frame by frame, so restore failure or cancellation
        discards the partial namespace and marks the sandbox dead. Reentrancy is
        guarded because _initialize_kernel runs code through execute(), which
        can itself trigger a restart on a protocol error.
        """
        if self._restarting:
            logger.warning(
                f"[{self.session_id}] _restart_kernel reentered; skipping nested restart"
            )
            return
        self._restarting = True
        try:
            try:
                await asyncio.wait_for(
                    _restart_kernel_manager(self.km),
                    timeout=_KERNEL_MANAGER_RESTART_TIMEOUT,
                )

                # Rebuild client connection.  Brief sleep to let any
                # in-flight ZMQ reader threads finish their current poll
                # cycle before we close the channels underneath them.
                await asyncio.sleep(0.3)

                if self.kc is not None:
                    try:
                        self.kc.stop_channels()
                    except Exception as e:
                        logger.debug(f"Failed to stop kernel channels during recovery: {e}")
                self.kc = self.km.client()
                self.kc.start_channels()

                # Drain stale messages that may have arrived on the new
                # channels before the kernel was fully initialised (e.g.
                # kernel-info broadcasts).  These carry the new session key
                # so they won't trigger "Invalid Signature", but leaving
                # them in the queue can confuse the first execute().
                try:
                    for _ in range(10):
                        await _to_thread_drained(
                            self.kc.get_iopub_msg, timeout=0.1
                        )
                except QueueEmpty:
                    pass  # queue drained

                # Wait for kernel to be ready
                await asyncio.wait_for(self._wait_for_ready(), timeout=30)

                # The restarted process is reachable. Clear the old latch so
                # internal bootstrap can execute; failure paths below restore it.
                self._kernel_dead = False

                # Re-initialize the kernel
                self._initialized = False
                await self._initialize_kernel()
            except BaseException as restart_error:
                logger.error(f"Failed to restart kernel: {restart_error}")
                self._kernel_dead = True
                try:
                    await self._shutdown_kernel_unlocked()
                except Exception:
                    logger.exception(
                        "Failed to stop partially restarted kernel for %s",
                        self.session_id,
                    )
                if not isinstance(restart_error, Exception):
                    raise
                return

            # Kernel is healthy again — and it is a NEW namespace. Anything
            # mid-flight across several cells (a snapshot runtime capture)
            # must not keep going as if nothing happened: the restart
            # rehydrates from the last saved pkl, i.e. an older generation
            # than the one it already partially serialized.
            self._kernel_dead = False
            self.kernel_epoch = getattr(self, "kernel_epoch", 0) + 1
            epoch_callback = getattr(self, "runtime_epoch_callback", None)
            if callable(epoch_callback):
                epoch_callback(self.kernel_epoch)

            # Restore mutates globals frame by frame.  A failed restore leaves a
            # partial namespace, so it must never be handed back as usable.
            try:
                await self._restore_session()
            except BaseException as restore_error:
                logger.error(
                    f"[{self.session_id}] Session restore failed after kernel restart "
                    f"(discarding partial namespace): {restore_error}"
                )
                self._kernel_dead = True
                try:
                    await self._shutdown_kernel_unlocked()
                except Exception:
                    logger.exception(
                        "Failed to stop kernel after restore failure for %s",
                        self.session_id,
                    )
                if not isinstance(restore_error, Exception):
                    raise
                return

            logger.info(f"Kernel restarted for session {self.session_id}")
        finally:
            self._restarting = False

    async def stop(self, *, save: bool = True):
        """Stop the kernel after its requested final save has committed.

        ``save=False`` shuts the kernel down without persisting it. Used when a
        snapshot restore has already installed the authoritative pkl/notebook for
        this session: saving would overwrite them with the pre-restore kernel.

        With ``save=True`` the closing save is **one attempt** (spec 5.1). A
        failure keeps the previous committed generation, is recorded by the
        persistence layer, and does not stop the shutdown: holding the kernel
        alive for a person to answer is the gate this replaced. The state is
        then stale rather than current, and the UI says so.
        """
        # Pending saves themselves need this lock. Drain before acquiring it.
        if save:
            await self.await_pending_saves()
        async with self._exec_lock:
            if save:
                try:
                    saved = await self._save_session_locked()
                    if saved is not True:
                        raise RuntimeError(
                            "runtime persistence did not commit a current generation"
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # One attempt (spec 5.1). The previous committed generation is
                    # intact and the failure is already recorded; shutting down
                    # anyway is the point of removing the gate.
                    logger.exception(
                        "Final runtime persistence failed for %s; the previous "
                        "committed generation is kept and the kernel is closed",
                        self.session_id,
                    )
                # Independent of the dump, per spec 5.1: when the runtime could not
                # be persisted the notebook is the only remaining record of what
                # ran, which is exactly when losing it costs most.
                try:
                    self.notebook_state.save()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Notebook save failed during stop for %s", self.session_id,
                    )

            await self._shutdown_kernel_unlocked()

    async def _shutdown_kernel_unlocked(self):
        """Release resources; also used by failed startup inside execute's lock."""
        manager = self.km

        if self.kc is not None:
            try:
                self.kc.stop_channels()
            except Exception:
                logger.exception(
                    "Error stopping kernel channels for %s", self.session_id
                )
            self.kc = None
        self._initialized = False

        if manager is not None:
            try:
                await asyncio.wait_for(
                    _shutdown_kernel_manager(manager),
                    timeout=_KERNEL_MANAGER_SHUTDOWN_TIMEOUT,
                )
            except BaseException:
                self._kernel_dead = True
                self.km = manager
                raise
            self.km = None

        logger.info(f"Kernel stopped for session {self.session_id}")

    async def restart(self):
        """Restart kernel and restore state"""
        await self.stop()
        await self.start()
