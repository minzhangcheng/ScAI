"""Safe-point manager for cooperative workflow pause during snapshot capture.

Provides a per-session ``asyncio.Event``-based mechanism: A capture
transaction calls ``acquire()`` to block the workflow, and the workflow
checks ``is_paused()`` / ``wait_for()`` at safe points.  When the capture
finishes (or fails) ``release()`` unblocks the workflow.

``acquire(session_id, workflow_state)`` stores a shallow descriptor copy
so capture can read fields such as ``kernel_present`` without relying on
the live sandbox alone.

Empty-complete eligibility only trusts descriptors produced by cooperative
acknowledgement or an explicit producer acquire
(``eligibility_trusted=True``).  Capture-side seeding via
``eligibility_trusted=False`` may still store graph metadata but must not
authorize empty-complete.

Process-wide access: use ``get_snapshot_safe_point_manager()`` so every
SnapshotService and restore coordinator share one coordinator keyed by
session_id.
"""

from __future__ import annotations

import asyncio
from typing import Any


class SafePointManager:
    """Manages per-session safe points that cooperatively pause workflows.

    Thread-safe for asyncio: all state mutations happen under a single lock.

    Usage::

        mgr = get_snapshot_safe_point_manager()

        # Capture transaction acquires the safe point (pauses workflow):
        ok = mgr.acquire("session-1", workflow_state)
        try:
            ...  # do capture work
        finally:
            mgr.release("session-1")

        # Workflow checks at safe points:
        if mgr.is_paused("session-1"):
            await mgr.wait_for("session-1")
    """

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}
        self._pending_events: dict[str, asyncio.Event] = {}
        self._descriptors: dict[str, dict[str, Any]] = {}
        # Descriptors may be stored for metadata without authorizing
        # empty-complete. Only cooperative ack / producer acquires set this.
        self._eligibility_trusted: dict[str, bool] = {}
        self._lock = asyncio.Lock()

    def request_safe_point(self, session_id: str) -> bool:
        """Signal the workflow to pause for an upcoming restore operation.

        The workflow checks ``has_pending_request()`` at its next safe
        point and calls ``acknowledge_safe_point_request()`` to confirm.

        Returns ``True`` if the request was newly created, ``False`` if
        one was already pending or the workflow is already paused.
        """
        if session_id in self._events:
            return False
        if session_id in self._pending_events:
            return False
        self._pending_events[session_id] = asyncio.Event()
        return True

    def cancel_safe_point_request(self, session_id: str) -> None:
        """Withdraw a pending safe-point request.

        If the workflow has already acknowledged the request (and thus
        paused), the safe point is released to unblock it.  No teardown
        is performed — the workflow simply continues.
        """
        event = self._pending_events.pop(session_id, None)
        if event is not None:
            event.set()  # unblock the job if it was waiting
        self.release(session_id)

    def has_pending_request(self, session_id: str) -> bool:
        """Return ``True`` if a safe-point request is pending."""
        return session_id in self._pending_events

    def acknowledge_safe_point_request(
        self,
        session_id: str,
        workflow_state: dict[str, Any] | None = None,
    ) -> bool:
        """Transition a pending request to an acquired safe point.

        Called by the workflow when it reaches a safe point and sees a
        pending request.  Optionally stores *workflow_state* as the
        safe-point descriptor (``kernel_present``, ``next_node``, etc.).
        The descriptor is eligibility-trusted for empty-complete.

        Returns ``True`` if the request was acknowledged, ``False`` if
        none was pending.
        """
        event = self._pending_events.pop(session_id, None)
        if event is None:
            return False
        self.acquire(session_id, workflow_state, eligibility_trusted=True)
        event.set()  # unblock the job
        return True

    def update_descriptor(
        self,
        session_id: str,
        workflow_state: dict[str, Any],
        *,
        eligibility_trusted: bool | None = None,
    ) -> bool:
        """Update the descriptor for an already-acquired safe point.

        When *eligibility_trusted* is provided, updates the trust flag;
        otherwise the existing trust state is preserved.

        Returns ``True`` if the session is paused and the descriptor was
        replaced, ``False`` otherwise.
        """
        if session_id not in self._events:
            return False
        self._descriptors[session_id] = dict(workflow_state)
        if eligibility_trusted is not None:
            self._eligibility_trusted[session_id] = bool(eligibility_trusted)
        return True

    def get_pending_event(self, session_id: str) -> asyncio.Event | None:
        """Return the asyncio.Event associated with a pending request.

        The background job awaits this event to learn when the workflow
        has acknowledged the safe point.  Returns ``None`` if no request
        is pending.
        """
        return self._pending_events.get(session_id)

    def acquire(
        self,
        session_id: str,
        workflow_state: dict[str, Any] | None = None,
        *,
        eligibility_trusted: bool = True,
    ) -> bool:
        """Request a cooperative pause for *session_id*.

        Stores a shallow copy of *workflow_state* as the safe-point
        descriptor (for ``kernel_present`` and related capture hints).

        *eligibility_trusted* (default True) marks the descriptor as
        authorized for empty-complete eligibility. Capture lifecycle
        seeding must pass ``eligibility_trusted=False`` so raw
        ``workflow_state`` cannot invent idle/no-kernel authority.

        Returns ``True`` if the safe point was newly acquired, ``False``
        if one is already held (idempotent guard — existing trust/descriptor
        are left unchanged).
        """
        if session_id in self._events:
            return False
        event = asyncio.Event()
        event.clear()
        self._events[session_id] = event
        if workflow_state is not None:
            self._descriptors[session_id] = dict(workflow_state)
            self._eligibility_trusted[session_id] = bool(eligibility_trusted)
        else:
            self._descriptors.pop(session_id, None)
            self._eligibility_trusted.pop(session_id, None)
        return True

    def get_descriptor(self, session_id: str) -> dict[str, Any] | None:
        """Return the workflow-state descriptor stored at acquire, if any."""
        desc = self._descriptors.get(session_id)
        return dict(desc) if desc is not None else None

    def is_eligibility_trusted(self, session_id: str) -> bool:
        """True when the held descriptor may authorize empty-complete."""
        return bool(self._eligibility_trusted.get(session_id, False))

    def get_eligibility_descriptor(self, session_id: str) -> dict[str, Any] | None:
        """Descriptor for empty-complete eligibility, or None if untrusted."""
        if not self.is_eligibility_trusted(session_id):
            return None
        return self.get_descriptor(session_id)

    def release(self, session_id: str) -> None:
        """Release the safe point for *session_id*, unblocking any waiter."""
        event = self._events.pop(session_id, None)
        self._descriptors.pop(session_id, None)
        self._eligibility_trusted.pop(session_id, None)
        if event is not None:
            event.set()

    def is_paused(self, session_id: str) -> bool:
        """Return ``True`` if a safe-point is active (workflow should pause)."""
        return session_id in self._events

    async def wait_for(self, session_id: str) -> None:
        """Block the current coroutine until the safe point is released.

        If no safe point is active for *session_id* this returns immediately.
        """
        event = self._events.get(session_id)
        if event is not None:
            await event.wait()

    @property
    def active_sessions(self) -> set[str]:
        """Return the set of session IDs with active safe points."""
        return set(self._events.keys())


_PROCESS_SAFE_POINT_MANAGER: SafePointManager | None = None


def get_snapshot_safe_point_manager() -> SafePointManager:
    """Return the process-wide SafePointManager (lazy singleton)."""
    global _PROCESS_SAFE_POINT_MANAGER
    if _PROCESS_SAFE_POINT_MANAGER is None:
        _PROCESS_SAFE_POINT_MANAGER = SafePointManager()
    return _PROCESS_SAFE_POINT_MANAGER
