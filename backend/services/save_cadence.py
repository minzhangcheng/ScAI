"""When a runtime save is attempted. Spec 5.1.

The decision is a pure function of three things -- which boundary is asking,
when the last *successful* commit happened, and now -- so it is testable without
a kernel, a sandbox or a clock.

Three boundaries ask:

* ``round_boundary`` -- the graph reached the point spec 6.4 names. Always
  saves: it is the boundary the exposure bound is stated in terms of.
* ``closing`` -- the analysis is ending. Always saves, exactly once.
* ``step_boundary`` -- a cell finished inside a round. Saves only once the
  wall-clock floor has passed, because a save is minutes of work and a run that
  saved after every step would spend most of its time persisting.
* ``decision_boundary`` -- a decision snapshot is about to be taken and the
  kernel has moved past its newest committed generation (Part B spec 4.5).
  Always saves: the snapshot copies a committed generation, so a boundary that
  waited out the floor would capture a dump older than the workspace beside it.
  The floor is not the instrument here -- the caller only asks at all when a
  cell has run since the last commit.

The clock runs from the last **successful** commit, never from the last
attempt. Spec 5.1's promise -- "at most one round or N minutes of work is at
risk" -- is a statement about the last generation that actually reached disk. A
failed attempt that reset the clock would let the real exposure grow without
bound while the number shown to the user stayed at N.
"""
from __future__ import annotations

from typing import Final, Literal

SaveReason = Literal[
    "round_boundary", "step_boundary", "closing", "decision_boundary",
]

#: Boundaries that persist unconditionally.
_UNCONDITIONAL: Final[frozenset[str]] = frozenset(
    {"round_boundary", "closing", "decision_boundary"}
)


def should_save(
    *,
    reason: SaveReason | str,
    last_committed_at: float | None,
    now: float,
    floor_minutes: int,
) -> bool:
    """Whether the boundary named by *reason* should attempt a runtime save.

    ``last_committed_at`` is a POSIX timestamp of the last successful commit,
    or ``None`` when nothing has ever committed. ``floor_minutes`` is the
    wall-clock floor between step-boundary saves.

    An unknown *reason* saves. A boundary this module has not been taught about
    is a caller that believes it is a persistence point; refusing would drop its
    generation silently, and the cost of an extra save is minutes, not data.
    """
    if reason in _UNCONDITIONAL:
        return True
    if last_committed_at is None:
        # Nothing has ever committed. A session should not wait out the floor
        # for its first generation -- until one exists there is nothing to fall
        # back to at all.
        return True
    if reason != "step_boundary":
        return True
    floor_seconds = max(0, int(floor_minutes)) * 60
    return (now - last_committed_at) >= floor_seconds


def last_committed_epoch(sandbox) -> float | None:
    """POSIX time of *sandbox*'s last successful runtime commit, or None.

    Read from ``runtime_persistence["last_saved_at"]``, which the notebook
    record writes only when a generation reaches status ``current`` -- i.e. only
    on success. A failed attempt updates ``last_attempt_*`` and leaves this
    alone, which is exactly the clock spec 5.1 asks for.

    Anything unreadable answers ``None``, which makes the next step boundary
    save. Erring toward one extra save costs minutes; erring the other way
    silently extends the exposure window.
    """
    from datetime import datetime

    notebook = getattr(sandbox, "notebook_state", None)
    record = getattr(notebook, "runtime_persistence", None)
    if not isinstance(record, dict):
        return None
    raw = record.get("last_saved_at")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        # No timezone branch on purpose. `fromisoformat` returns an aware value
        # for the offset-bearing strings written today, and `.timestamp()` is
        # then exact. For a naive string written by an older build it returns a
        # naive value that `.timestamp()` resolves in this process's zone --
        # which is the zone it was written in, since the writer was this same
        # container. Reinterpreting those as UTC would move the floor on every
        # deployment that does not run UTC.
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def runtime_save_due(sandbox, reason: SaveReason | str) -> bool:
    """Whether *sandbox* should attempt a runtime save at this boundary."""
    import time

    from backend.config import RUNTIME_SAVE_FLOOR_MINUTES

    return should_save(
        reason=reason,
        last_committed_at=last_committed_epoch(sandbox),
        now=time.time(),
        floor_minutes=RUNTIME_SAVE_FLOOR_MINUTES,
    )
