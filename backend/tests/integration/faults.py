"""Fault injection helpers for Snapshot V2 lifecycle crash recovery tests.

Defines the exact fault points and a FaultBarrier that coordinates
injection at named transitions.
"""

import logging

logger = logging.getLogger(__name__)

RESTORE_POINTS = (
    "target_staged",
    "workspace_applied",
    "runtime_pkl_replaced",
    "runtime_signature_replaced",
    "notebook_applied",
    "state_persisted",
    "checkpoint_seeded",
    "journal_committed",
    "cleanup_started",
)

DELETE_POINTS = (
    "journal_written",
    "manifests_hidden",
    "lineage_saved",
    "gc_started",
)

IMPORT_POINTS = (
    "objects_verified",
    "manifest_published",
    "session_saved",
)

ALL_FAULT_POINTS = RESTORE_POINTS + DELETE_POINTS + IMPORT_POINTS


class FaultInjected(Exception):
    """Raised when a fault point is triggered during a lifecycle operation."""

    def __init__(self, point: str, message: str = "Fault injected"):
        self.point = point
        super().__init__(f"[{point}] {message}")


class FaultBarrier:
    """Coordinates fault injection at named lifecycle transitions.

    Usage::

        barrier = FaultBarrier("runtime_pkl_replaced")
        coord = RestoreCoordinator(fault_barrier=barrier)
        with pytest.raises(FaultInjected):
            await coord.apply(...)
    """

    def __init__(self, trigger_point: str | None = None):
        self._trigger = trigger_point
        self._hit = False

    def check(self, point: str) -> None:
        """Raise FaultInjected if *point* matches the configured trigger."""
        if point == self._trigger and not self._hit:
            self._hit = True
            logger.info("Fault injected at %s", point)
            raise FaultInjected(point)

    def check_and_resume(self, point: str) -> None:
        """Raise FaultInjected if *point* matches, then reset."""
        if point == self._trigger and not self._hit:
            self._hit = True
            logger.info("Fault injected at %s", point)
            raise FaultInjected(point)

    @property
    def was_hit(self) -> bool:
        return self._hit

    def reset(self) -> None:
        self._hit = False


def make_rollback_barrier(point: str) -> FaultBarrier:
    """Create a FaultBarrier that triggers at *point*.

    Recovery should roll back to the pre_restore state.
    """
    return FaultBarrier(point)


def make_forward_only_barrier(point: str) -> FaultBarrier:
    """Create a FaultBarrier that triggers at *point*.

    Recovery should forward-complete to the target state.
    """
    return FaultBarrier(point)
