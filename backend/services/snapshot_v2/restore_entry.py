"""RestoreEntryPlan and one-shot entry logic for Snapshot V2.

Determines the correct graph entry intent and target node based on
restore metadata (``restore_resume_kind``, ``restore_start_intent``,
``restore_resume_node``) and builds a verified entry plan that the
:meth:`~backend.workflow.core.SingleCellAnalysisWorkflow._restore_entry`
node uses to route the workflow.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, Field


class RestoreEntryPlan(BaseModel):
    """Describes how the workflow graph should enter after a restore.

    ``intent`` determines the graph entry strategy:

    * ``normal`` — fresh / uninteresting state; route through
      ``initialize_round`` as usual.
    * ``explicit_start`` — user clicked "Start" on an idle-restored
      session; route through ``initialize_round`` but preserve the
      current round (don't auto-increment).
    * ``graph_node`` — jump directly to ``target_node``, bypassing
      ``initialize_round``.
    * ``semantic_gate`` — waiting at a HITL gate; the existing
      ``resume_restored_wait_state`` handles this outside the graph.
    * ``terminal`` — snapshot was taken after completion; no graph
      entry needed.
    * ``idle`` — restored from a manual snapshot with no continuation
      target; do not start the graph.
    """

    intent: Literal["normal", "explicit_start", "graph_node",
                    "semantic_gate", "terminal", "idle"]
    target_node: str | None = None
    workflow_state: dict[str, Any] = Field(default_factory=dict)
    preserve_round_on_initialize: bool = False
    terminal_status: Literal["completed", "stopped", "failed"] | None = None


def build_restore_entry_plan(
    analysis_state: Any,
    *,
    restore_start_intent: str | None = None,
) -> RestoreEntryPlan:
    """Build a ``RestoreEntryPlan`` from analysis state metadata.

    This is a pure function — no side effects, no mutation of
    *analysis_state*.  The caller (``_restore_entry`` in
    ``core.py``) applies the plan.
    """
    resume_kind = getattr(analysis_state, "restore_resume_kind", None)
    start_intent = restore_start_intent or getattr(
        analysis_state, "restore_start_intent", None,
    )

    # explicit_start from lifecycle POST /run on idle-restored sessions
    if start_intent == "explicit_start":
        return RestoreEntryPlan(
            intent="explicit_start",
            target_node="initialize_round",
            preserve_round_on_initialize=True,
        )

    # Resume kind from dispatcher activation
    if resume_kind == "terminal":
        terminal_payload = dict(
            getattr(analysis_state, "restore_resume_workflow_state", None) or {}
        )
        terminal_status = terminal_payload.get("terminal_status")
        if terminal_status not in {"completed", "stopped", "failed"}:
            current = getattr(analysis_state, "status", None)
            terminal_status = getattr(current, "value", current)
        if terminal_status not in {"completed", "stopped", "failed"}:
            terminal_status = "completed"
        return RestoreEntryPlan(
            intent="terminal",
            terminal_status=terminal_status,
        )

    if resume_kind == "semantic_gate":
        return RestoreEntryPlan(
            intent="semantic_gate",
            target_node="wait_for_user_input",
        )

    if resume_kind == "graph_node":
        target = getattr(analysis_state, "restore_resume_node", None) or ""
        return RestoreEntryPlan(
            intent="graph_node",
            target_node=target,
            workflow_state=copy.deepcopy(
                getattr(analysis_state, "restore_resume_workflow_state", None) or {}
            ),
        )

    # Idle restore (branch_id set, no resume kind) — no auto-start
    branch_id = getattr(analysis_state, "restore_branch_id", None)
    if branch_id and not resume_kind:
        return RestoreEntryPlan(
            intent="idle",
        )

    return RestoreEntryPlan(intent="normal")
