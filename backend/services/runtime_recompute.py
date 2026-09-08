"""What survives the runtime-persistence HITL gate.

Spec 5.1 replaces the gate with closing-save semantics: one attempt, no retry,
no human in the loop, the previous committed generation kept, and a persistent
non-blocking warning. Exactly one blocking case survives, because it is not a
question about *this* save at all:

    If no generation ever succeeded, the runtime is marked unavailable and
    re-run-from-data is offered.

That is ``RuntimeUnavailable``, and ``prepare_runtime_recompute`` is the
authorization the user gives to answer it. Everything else the gate did --
arming a dialog on a failed save, the retry/manage/terminate actions, the
degraded-recovery hold -- is gone, because a failed save leaves a *stale*
generation the analysis can continue from while the UI says so.
"""
from __future__ import annotations

from typing import Any

from backend.models.schemas import TaskStatus


class RuntimeUnavailable(RuntimeError):
    """No committed runtime generation exists and work has already happened.

    Raised in place of the gate's ``RuntimePersistenceBlocked`` for the one
    condition that still stops forward execution. It is answered by an explicit
    re-run, not by a dialog: there is nothing to retry, because nothing was ever
    saved.
    """


def prepare_runtime_recompute(state: Any, sandbox: Any | None = None) -> None:
    """Record the user's explicit authorization to re-run from the input data.

    A process restart leaves the notebook metadata as the authoritative
    failed record while the authorization lives in ``AnalysisState``, so this
    applies it to a freshly created sandbox before the executor loop reads it.
    """
    persistence = dict(getattr(state, "runtime_persistence", None) or {})
    persistence.update({
        "requires_recompute": False,
        "recommended_action": "recompute_or_reduce_runtime",
        "recompute_authorized": True,
    })
    notebook = getattr(sandbox, "notebook_state", None)
    if notebook is not None:
        notebook.runtime_persistence.update(persistence)
        notebook.save_metadata()
        persistence = dict(notebook.runtime_persistence)
    state.runtime_persistence = persistence
    state.runtime_recompute_pending = True
    state.status = TaskStatus.RUNNING
