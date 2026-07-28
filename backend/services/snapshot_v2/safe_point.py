"""Safe-point manager for cooperative workflow pause during snapshot capture.

Provides a per-session ``asyncio.Event``-based mechanism: A capture
transaction calls ``acquire()`` to block the workflow, and the workflow
checks ``is_paused()`` / ``wait_for()`` at safe points.  When the capture
finishes (or fails) ``release()`` unblocks the workflow.

``acquire(session_id, workflow_state, owner_token=...)`` stores a shallow
descriptor copy so capture can read fields such as ``kernel_present``
without relying on the live sandbox alone.

Empty-complete eligibility only trusts descriptors produced by cooperative
acknowledgement or an explicit producer acquire
(``eligibility_trusted=True``).  Capture-side seeding via
``eligibility_trusted=False`` may still store graph metadata but must not
authorize empty-complete.

Process-wide access: use ``get_snapshot_safe_point_manager()`` so every
SnapshotService and restore coordinator share one coordinator keyed by
session_id.

Ownership enforcement:
- ``SafePointRequest`` is a frozen dataclass with ``owner_token``,
  ``generation``, ``event``.
- Requester mutations (``request_safe_point`` / ``acquire`` /
  ``update_descriptor`` / ``cancel_safe_point_request`` / ``release``)
  require a non-empty ``owner_token`` and reject foreign owners.
- ``acknowledge_safe_point_request(session_id, generation, descriptor,
  role=...)`` checks generation for workflow acknowledgements and for a
  restore coordinator that has atomically proven the session quiescent
  (neither requires the requester token).
- Empty tokens remain allowed only via explicitly named test adapters.
- No filesystem I/O, workflow calls, or message emission is performed
  while holding ``SafePointManager._lock`` (the lock only protects the
  internal data structures).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SafePointRequest:
    """A pending safe-point request with ownership and generation handshake.

    Attributes:
        owner_token: Non-empty token identifying the requester.
        generation: Monotonic counter for acknowledge handshake.
        event: Set when the request is acknowledged by the workflow.
    """
    owner_token: str
    generation: int
    event: asyncio.Event


def _normalize_owner_token(owner_token: str | None) -> str:
    """Return a stripped owner token; empty/None become ``\"\"``."""
    if owner_token is None:
        return ""
    return str(owner_token).strip()


class SafePointManager:
    """Manages per-session safe points that cooperatively pause workflows.

    Thread-safe for asyncio: all state mutations happen under a single lock.

    Usage::

        mgr = get_snapshot_safe_point_manager()

        # Capture transaction acquires the safe point (pauses workflow):
        ok = await mgr.acquire("session-1", workflow_state, owner_token="capture:1")
        try:
            ...  # do capture work
        finally:
            await mgr.release("session-1", "capture:1")

        # Workflow checks at safe points:
        if mgr.is_paused("session-1"):
            await mgr.wait_for("session-1")
    """

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}
        self._pending_requests: dict[str, SafePointRequest] = {}
        self._descriptors: dict[str, dict[str, Any]] = {}
        # Descriptors may be stored for metadata without authorizing
        # empty-complete. Only cooperative ack / producer acquires set this.
        self._eligibility_trusted: dict[str, bool] = {}
        self._acquired_owners: dict[str, str] = {}
        self._generation_counters: dict[str, int] = {}
        self._superseded: set[str] = set()
        self._availability_waiters: dict[str, set[asyncio.Event]] = {}
        self._lock = asyncio.Lock()

    def _request_safe_point_unlocked(
        self, session_id: str, owner_token: str
    ) -> SafePointRequest | None:
        if session_id in self._events or session_id in self._pending_requests:
            return None
        generation = self._generation_counters.get(session_id, 0) + 1
        self._generation_counters[session_id] = generation
        request = SafePointRequest(
            owner_token=owner_token,
            generation=generation,
            event=asyncio.Event(),
        )
        self._pending_requests[session_id] = request
        return request

    def _notify_available_unlocked(self, session_id: str) -> None:
        if session_id in self._events or session_id in self._pending_requests:
            return
        for waiter in self._availability_waiters.pop(session_id, set()):
            waiter.set()

    def mark_superseded(self, session_id: str) -> None:
        """Record that a committed restore has invalidated the in-flight run.

        Session-scoped rather than workflow-scoped on purpose: the restore evicts
        the workflow object, so anything else frozen on this safe point — the
        bioinformatics step loop, a node-entry gate — could not reach a flag
        stored there. Cleared once the superseded run has unwound.
        """
        self._superseded.add(session_id)

    def is_superseded(self, session_id: str) -> bool:
        """True while a run predating a committed restore is still unwinding."""
        return session_id in self._superseded

    def clear_superseded(self, session_id: str) -> None:
        self._superseded.discard(session_id)

    async def request_safe_point(
        self, session_id: str, owner_token: str
    ) -> SafePointRequest | bool:
        """Signal the workflow to pause for an upcoming restore operation.

        The workflow checks ``has_pending_request()`` at its next safe
        point and calls ``acknowledge_safe_point_request()`` to confirm.

        *owner_token* must be non-empty. It is stored in the pending
        request and must match on subsequent ``cancel_safe_point_request``,
        ``update_descriptor``, or ``release`` calls.

        Returns the new ``SafePointRequest`` if created, ``False`` if
        one was already pending, the workflow is already paused, or the
        token is empty.
        """
        token = _normalize_owner_token(owner_token)
        if not token:
            return False
        async with self._lock:
            request = self._request_safe_point_unlocked(session_id, token)
            return request if request is not None else False

    async def request_safe_point_when_available(
        self,
        session_id: str,
        owner_token: str,
        *,
        timeout_seconds: float,
    ) -> SafePointRequest | None:
        token = _normalize_owner_token(owner_token)
        if not token or timeout_seconds < 0:
            return None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds

        while True:
            waiter = asyncio.Event()
            async with self._lock:
                request = self._request_safe_point_unlocked(session_id, token)
                if request is not None:
                    return request
                self._availability_waiters.setdefault(session_id, set()).add(waiter)

            remaining = deadline - loop.time()
            try:
                if remaining <= 0:
                    return None
                await asyncio.wait_for(waiter.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            finally:
                async with self._lock:
                    waiters = self._availability_waiters.get(session_id)
                    if waiters is not None:
                        waiters.discard(waiter)
                        if not waiters:
                            self._availability_waiters.pop(session_id, None)

    async def cancel_safe_point_request(
        self, session_id: str, owner_token: str,
    ) -> bool:
        """Withdraw a pending safe-point request.

        *owner_token* must be non-empty and must match the stored owner
        on the pending request or acquired hold; otherwise returns
        ``False``.

        If the workflow has already acknowledged the request (and thus
        paused), the safe point is released to unblock it.  No teardown
        is performed — the workflow simply continues.

        Returns ``True`` when the request was resolved (cancelled or
        already free under matching ownership), ``False`` on ownership
        mismatch or empty token.
        """
        token = _normalize_owner_token(owner_token)
        if not token:
            return False
        async with self._lock:
            req = self._pending_requests.get(session_id)
            if req is not None:
                if req.owner_token != token:
                    return False
                self._pending_requests.pop(session_id, None)
                req.event.set()  # unblock the job if it was waiting
            return self._release_unlocked(session_id, token)

    def has_pending_request(self, session_id: str) -> bool:
        """Return ``True`` if a safe-point request is pending."""
        return session_id in self._pending_requests

    def get_pending_generation(self, session_id: str) -> int | None:
        """Return the generation counter for a pending request, or ``None``."""
        req = self._pending_requests.get(session_id)
        if req is None:
            return None
        return req.generation

    def get_pending_observation(
        self, session_id: str,
    ) -> SafePointRequest | None:
        """Return the pending request for bioinformatics step boundaries."""
        return self._pending_requests.get(session_id)

    async def acknowledge_safe_point_request(
        self,
        session_id: str,
        generation: int,
        workflow_state: dict[str, Any] | None = None,
        role: str = "workflow",
    ) -> bool:
        """Transition a pending request to an acquired safe point.

        Called by the workflow when it reaches a safe point and sees a
        pending request.  *generation* must match the generation stored
        in the pending request.

        Workflow acknowledgements (``role="workflow"``), and coordinator
        acknowledgements after an authoritative quiescence check, never
        require the requester token.

        Optionally stores *workflow_state* as the safe-point descriptor
        (``kernel_present``, ``next_node``, etc.).  The descriptor is
        eligibility-trusted for empty-complete.

        Returns ``True`` if the request was acknowledged, ``False`` if
        none was pending or if the generation doesn't match.
        """
        async with self._lock:
            req = self._pending_requests.pop(session_id, None)
            if req is None:
                return False
            if req.generation != generation:
                self._pending_requests[session_id] = req
                return False
            acquired = self._acquire_unlocked(
                session_id, workflow_state,
                eligibility_trusted=True,
                owner_token=req.owner_token,
            )
            if not acquired:
                self._pending_requests[session_id] = req
                return False
            req.event.set()  # unblock the job
            return True

    async def update_descriptor(
        self,
        session_id: str,
        owner_token: str,
        workflow_state: dict[str, Any],
        *,
        eligibility_trusted: bool | None = None,
    ) -> bool:
        """Update the descriptor for an already-acquired safe point.

        *owner_token* must be non-empty and match the acquired owner.
        When *eligibility_trusted* is provided, updates the trust flag;
        otherwise the existing trust state is preserved.

        Returns ``True`` if the session is paused under matching ownership
        and the descriptor was replaced, ``False`` otherwise.
        """
        token = _normalize_owner_token(owner_token)
        if not token:
            return False
        async with self._lock:
            if session_id not in self._events:
                return False
            stored = self._acquired_owners.get(session_id)
            if stored is None or stored != token:
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
        req = self._pending_requests.get(session_id)
        if req is None:
            return None
        return req.event

    async def acquire(
        self,
        session_id: str,
        workflow_state: dict[str, Any] | None = None,
        *,
        eligibility_trusted: bool = True,
        owner_token: str,
    ) -> bool:
        """Request a cooperative pause for *session_id*.

        Stores a shallow copy of *workflow_state* as the safe-point
        descriptor (for ``kernel_present`` and related capture hints).

        *eligibility_trusted* (default True) marks the descriptor as
        authorized for empty-complete eligibility. Capture lifecycle
        seeding must pass ``eligibility_trusted=False`` so raw
        ``workflow_state`` cannot invent idle/no-kernel authority.

        *owner_token* must be non-empty. It is stored and must match on
        subsequent ``update_descriptor`` / ``release`` /
        ``cancel_safe_point_request`` calls.

        Returns ``True`` if the safe point was newly acquired, ``False``
        if one is already held, the token is empty, or acquisition is
        refused (idempotent guard — existing trust/descriptor are left
        unchanged when already held).
        """
        token = _normalize_owner_token(owner_token)
        if not token:
            return False
        async with self._lock:
            return self._acquire_unlocked(
                session_id,
                workflow_state,
                eligibility_trusted=eligibility_trusted,
                owner_token=token,
            )

    def _acquire_unlocked(
        self,
        session_id: str,
        workflow_state: dict[str, Any] | None = None,
        *,
        eligibility_trusted: bool = True,
        owner_token: str,
    ) -> bool:
        """Acquire without taking the lock (caller must hold ``_lock``)."""
        if session_id in self._events or session_id in self._pending_requests:
            return False
        token = _normalize_owner_token(owner_token)
        if not token:
            return False
        event = asyncio.Event()
        event.clear()
        self._events[session_id] = event
        self._acquired_owners[session_id] = token
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

    def is_owned_by(self, session_id: str, owner_token: str) -> bool:
        """Return whether *owner_token* owns the acquired session hold."""
        token = _normalize_owner_token(owner_token)
        return bool(token) and self._acquired_owners.get(session_id) == token

    def get_eligibility_descriptor(self, session_id: str) -> dict[str, Any] | None:
        """Descriptor for empty-complete eligibility, or None if untrusted."""
        if not self.is_eligibility_trusted(session_id):
            return None
        return self.get_descriptor(session_id)

    async def release(self, session_id: str, owner_token: str) -> bool:
        """Release the safe point for *session_id*, unblocking any waiter.

        *owner_token* must be non-empty and match the stored owner
        token or the operation returns ``False``.

        Returns ``True`` if the safe point was released (or already free
        with no conflicting owner), ``False`` on ownership mismatch or
        empty token.
        """
        token = _normalize_owner_token(owner_token)
        if not token:
            return False
        async with self._lock:
            return self._release_unlocked(session_id, token)

    def _release_unlocked(self, session_id: str, owner_token: str) -> bool:
        """Release without taking the lock (caller must hold ``_lock``)."""
        token = _normalize_owner_token(owner_token)
        if not token:
            return False
        stored = self._acquired_owners.get(session_id)
        if stored is not None and stored != token:
            return False
        pending = self._pending_requests.get(session_id)
        if (
            pending is not None
            and pending.owner_token
            and pending.owner_token != token
        ):
            return False
        pending = self._pending_requests.pop(session_id, None)
        if pending is not None:
            pending.event.set()
        event = self._events.pop(session_id, None)
        self._descriptors.pop(session_id, None)
        self._eligibility_trusted.pop(session_id, None)
        self._acquired_owners.pop(session_id, None)
        if event is not None:
            event.set()
        self._notify_available_unlocked(session_id)
        return True

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


class EmptyTokenSafePointAdapter:
    """Test-only adapter that allows empty-token mutations.

    Production callers must never use this. Tests that still seed or
    clear holds without an explicit owner may wrap a manager with this
    adapter.
    """

    def __init__(self, manager: SafePointManager) -> None:
        self._manager = manager

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)

    async def request_safe_point(
        self, session_id: str, owner_token: str = "test-empty-adapter",
    ) -> SafePointRequest | bool:
        token = _normalize_owner_token(owner_token) or "test-empty-adapter"
        return await self._manager.request_safe_point(session_id, token)

    async def cancel_safe_point_request(
        self, session_id: str, owner_token: str = "test-empty-adapter",
    ) -> bool:
        token = _normalize_owner_token(owner_token)
        if token:
            return await self._manager.cancel_safe_point_request(session_id, token)
        # Allow teardown by matching whatever owner is present (test only).
        async with self._manager._lock:
            pending = self._manager._pending_requests.get(session_id)
            acquired = self._manager._acquired_owners.get(session_id)
            owner = (
                pending.owner_token if pending is not None else None
            ) or acquired or "test-empty-adapter"
            return self._manager._release_unlocked(session_id, owner)

    async def acquire(
        self,
        session_id: str,
        workflow_state: dict[str, Any] | None = None,
        *,
        eligibility_trusted: bool = True,
        owner_token: str = "test-empty-adapter",
    ) -> bool:
        token = _normalize_owner_token(owner_token) or "test-empty-adapter"
        return await self._manager.acquire(
            session_id,
            workflow_state,
            eligibility_trusted=eligibility_trusted,
            owner_token=token,
        )

    async def update_descriptor(
        self,
        session_id: str,
        owner_token_or_state: str | dict[str, Any],
        workflow_state: dict[str, Any] | None = None,
        *,
        eligibility_trusted: bool | None = None,
    ) -> bool:
        # Support both new (token, state) and legacy (state-only) call shapes.
        if isinstance(owner_token_or_state, dict) and workflow_state is None:
            async with self._manager._lock:
                if session_id not in self._manager._events:
                    return False
                self._manager._descriptors[session_id] = dict(owner_token_or_state)
                if eligibility_trusted is not None:
                    self._manager._eligibility_trusted[session_id] = bool(
                        eligibility_trusted
                    )
                return True
        token = _normalize_owner_token(str(owner_token_or_state))
        if not token:
            async with self._manager._lock:
                if session_id not in self._manager._events:
                    return False
                self._manager._descriptors[session_id] = dict(workflow_state or {})
                if eligibility_trusted is not None:
                    self._manager._eligibility_trusted[session_id] = bool(
                        eligibility_trusted
                    )
                return True
        return await self._manager.update_descriptor(
            session_id,
            token,
            workflow_state or {},
            eligibility_trusted=eligibility_trusted,
        )

    async def release(
        self, session_id: str, owner_token: str = "test-empty-adapter",
    ) -> bool:
        token = _normalize_owner_token(owner_token)
        if token:
            return await self._manager.release(session_id, token)
        async with self._manager._lock:
            pending = self._manager._pending_requests.get(session_id)
            acquired = self._manager._acquired_owners.get(session_id)
            owner = (
                pending.owner_token if pending is not None else None
            ) or acquired or "test-empty-adapter"
            return self._manager._release_unlocked(session_id, owner)


_PROCESS_SAFE_POINT_MANAGER: SafePointManager | None = None


def get_snapshot_safe_point_manager() -> SafePointManager:
    """Return the process-wide SafePointManager (lazy singleton)."""
    global _PROCESS_SAFE_POINT_MANAGER
    if _PROCESS_SAFE_POINT_MANAGER is None:
        _PROCESS_SAFE_POINT_MANAGER = SafePointManager()
    return _PROCESS_SAFE_POINT_MANAGER
