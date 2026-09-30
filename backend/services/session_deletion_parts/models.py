"""Lease and pending-cleanup records for session deletion."""
from __future__ import annotations

import asyncio
import os
import threading
import weakref
from dataclasses import dataclass, field

class SessionProducerStopError(RuntimeError):
    """Deletion tombstoned a session but could not stop every producer."""


@dataclass
class SessionDeletionIdentity:
    """Owner and generation needed for deletion without analysis deserialization."""

    id: str
    user_id: str
    lifecycle_generation: int = 0


@dataclass
class _TombstoneCleanupLease:
    """Process-crash-safe ownership of one session's physical cleanup."""

    session_id: str
    fd: int
    pid: int
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        if os.getpid() != self.pid:
            # A forked child shares the inherited open-file-description. An
            # explicit LOCK_UN there would release the parent's live lease.
            os.close(self.fd)
            return
        try:
            import fcntl

            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(self.fd)
            finally:
                _release_local_cleanup_lease(self.session_id, self.pid)

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


@dataclass
class _PendingTombstoneCleanup:
    owner_id: str = ""
    generation: int = 0
    lease: _TombstoneCleanupLease | None = None
    tasks: list[asyncio.Task] = field(default_factory=list)
    workflow_task: asyncio.Task | None = None
    report_workflow_task: asyncio.Task | None = None
    chat_task: asyncio.Task | None = None
    workflow: object | None = None
    report_workflow: object | None = None
    chat_sandbox: object | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    drain_started: bool = False


# Tombstoned generations are never made readable again. This registry retains
# only producer handles needed for a later cleanup retry in the same process;
# after a restart the producers are gone and the disk sweep can proceed.
_pending_tombstone_cleanup: dict[str, _PendingTombstoneCleanup] = {}

_cleanup_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    weakref.WeakValueDictionary[str, asyncio.Lock],
] = weakref.WeakKeyDictionary()
_cleanup_locks_guard = threading.Lock()
_local_cleanup_leases: set[str] = set()
_local_cleanup_leases_guard = threading.Lock()
_local_cleanup_leases_pid = os.getpid()


def _claim_local_cleanup_lease(session_id: str) -> tuple[bool, int]:
    """Claim one process-local lease before relying on ``flock`` semantics."""
    global _local_cleanup_leases_pid
    pid = os.getpid()
    with _local_cleanup_leases_guard:
        # A post-fork child inherits Python memory but not the parent's logical
        # ownership. The kernel flock remains the cross-process authority.
        if _local_cleanup_leases_pid != pid:
            _local_cleanup_leases.clear()
            _local_cleanup_leases_pid = pid
        if session_id in _local_cleanup_leases:
            return False, pid
        _local_cleanup_leases.add(session_id)
        return True, pid


def _release_local_cleanup_lease(session_id: str, pid: int) -> None:
    global _local_cleanup_leases_pid
    with _local_cleanup_leases_guard:
        if _local_cleanup_leases_pid == pid == os.getpid():
            _local_cleanup_leases.discard(session_id)
