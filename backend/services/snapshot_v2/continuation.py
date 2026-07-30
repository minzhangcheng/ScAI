"""Continuation capsule registry for snapshot V2.

Maps SnapshotPhase values to typed ContinuationCapsule variants and
serialises required workflow state into the object store.
"""

from __future__ import annotations

import json
from typing import Any

from backend.constants import WORKFLOW_GRAPH_NODES, WaitInputReason
from backend.models.schemas import SnapshotPhase
from backend.services.snapshot_v2.errors import SnapshotCaptureError
from backend.services.snapshot_v2.models import (
    ContinuationCapsule,
    GraphNodeCapsule,
    IdleCapsule,
    ObjectRef,
    SemanticGateCapsule,
    TerminalCapsule,
)
from backend.services.snapshot_v2.object_store import ObjectStore

GRAPH_VERSION = "2.0"

# ── Phase-to-capsule-type mapping ─────────────────────────────────────────────

_GRAPH_NODE_PHASES: frozenset[str] = frozenset({
    "pre_execution_stage1",
    "pre_execution_stage2",
    "pre_execution_full",
})

_SEMANTIC_GATE_PHASES: frozenset[str] = frozenset({
    "pre_annotation",
    "post_annotation",
    "pre_direction",
    "post_direction",
    "pre_round_wait",
    "pre_integration_gate",
    "pre_major_revision",  # real plan-revision gate
})

_TERMINAL_PHASES: frozenset[str] = frozenset({
    "post_report",
})

# Unconditional idle only — post_chat / pre_restore derive from live safe point
_IDLE_PHASES: frozenset[str] = frozenset({
    "manual",
})

# Dynamic phases: capsule derived from acknowledged live safe-point descriptor
_DYNAMIC_PHASES: frozenset[str] = frozenset({
    "post_chat",
    "pre_restore",
})

_ALL_KNOWN_PHASES = (
    _GRAPH_NODE_PHASES
    | _SEMANTIC_GATE_PHASES
    | _TERMINAL_PHASES
    | _IDLE_PHASES
    | _DYNAMIC_PHASES
)

# ── Phase → next-node mapping for graph_node capsules ─────────────────────────
# Align with backend/workflow/restore_targets.py. initialize_round is not a
# V2 restore target for this phase.

_PHASE_TO_NEXT_NODE: dict[str, str] = {
    "pre_execution_stage1": "bioinformatics_analysis",
    "pre_execution_stage2": "bioinformatics_analysis",
    "pre_execution_full": "bioinformatics_analysis",
}

_PHASE_TO_PLANNING_STAGE: dict[str, str] = {
    "pre_execution_stage1": "stage1",
    "pre_execution_stage2": "stage2",
    "pre_execution_full": "full",
}

# ── Phase → gate mapping for semantic_gate capsules ───────────────────────────

_PHASE_TO_GATE: dict[str, str] = {
    "pre_annotation": "annotation_confirmation",
    "post_annotation": "annotation_confirmation",
    "pre_direction": "direction_selection",
    "post_direction": "direction_selection",
    "pre_round_wait": "round_wait",
    "pre_integration_gate": "integration_selection",
    "pre_major_revision": "plan_revision",
}

# ── Known graph node names (from authoritative WORKFLOW_GRAPH_NODES) ────────

# Capsule targets exclude the restore_entry node (internal graph entry point).
_KNOWN_GRAPH_NODES: frozenset[str] = frozenset(
    n for n in WORKFLOW_GRAPH_NODES if n != "restore_entry"
)


# Reverse of _PHASE_TO_GATE for the dynamic path, which knows the gate but not
# the capture phase that would normally imply it.
_GATE_TO_PHASE: dict[str, str] = {
    gate: phase for phase, gate in _PHASE_TO_GATE.items()
}

# A live HITL gate is recorded on the session's status, not in the safe-point
# descriptor. Only statuses whose gate is unambiguous appear here.
_STATUS_TO_GATE: dict[str, str] = {
    "waiting_annotation": "annotation_confirmation",
    "waiting_plan_review": "plan_review",
    "waiting_input": "round_wait",
}

# ``waiting_input`` is the one entry above that is not self-describing. The
# round gate, direction selection and integration selection all block in
# ``_wait_for_user_input`` and so resume at the same physical checkpoint, which
# is why mapping them all to round_wait is sound. An intervention gate driven by
# ``_drive_hitl_gate`` also parks on WAITING_INPUT, but it blocks *inside* the
# node it interrupted and names itself only on ``wait_input_reason``. Freezing
# one of those as round_wait is not a harmless relabel: the restore dispatcher
# falls back to the capsule's gate when the payload carries no wait reason, so
# the reason is overwritten with "round_wait" while the intervention's own info
# survives — the reconnect resend no longer recognizes it, and round_wait is a
# semantic wait gate, so /resume answers "restored wait continuation" and drives
# the graph forward from a gate the session was never at. Idle is the honest
# capsule for a mid-node park: the true position needs the checkpointer's
# ``next``, which is not available here.
_NON_ROUND_WAIT_INPUT_REASONS: frozenset[str] = frozenset({
    WaitInputReason.LLM_FAILURE.value,
    WaitInputReason.AGENT_DISABLED.value,
    WaitInputReason.MARKER_DB_MISSING.value,
    WaitInputReason.SNAPSHOT_FAILURE.value,
})


def _gate_from_status(analysis_state: Any) -> str | None:
    """Return the semantic gate a parked session is sitting at, or None."""
    if analysis_state is None:
        return None
    status = getattr(analysis_state, "status", None)
    status = getattr(status, "value", status)
    if status is None:
        return None
    status = str(status).lower()
    gate = _STATUS_TO_GATE.get(status)
    if gate is None:
        return None
    if status == "waiting_input":
        reason = getattr(analysis_state, "wait_input_reason", None)
        reason = getattr(reason, "value", reason)
        if reason is not None and str(reason) in _NON_ROUND_WAIT_INPUT_REASONS:
            return None
    return gate


class ContinuationRegistry:
    """Maps a SnapshotPhase to a ContinuationCapsule, serialising state on the fly."""

    # Expose mappings for tests / restore_targets alignment checks
    _PHASE_TO_NEXT_NODE = _PHASE_TO_NEXT_NODE
    _PHASE_TO_GATE = _PHASE_TO_GATE

    def __init__(self, object_store: ObjectStore):
        self._object_store = object_store

    def freeze(
        self,
        phase: SnapshotPhase | str,
        analysis_state: Any,
        workflow_state: dict,
        *,
        live_safe_point: dict[str, Any] | None = None,
    ) -> ContinuationCapsule:
        if phase not in _ALL_KNOWN_PHASES:
            raise SnapshotCaptureError(
                "UNKNOWN_PHASE",
                f"Unknown continuation phase: {phase}",
            )

        if phase in _DYNAMIC_PHASES:
            return self.freeze_dynamic(
                live_safe_point or workflow_state or {},
                analysis_state=analysis_state,
                workflow_state=workflow_state,
            )

        if phase in _GRAPH_NODE_PHASES:
            return self._freeze_graph_node(phase, workflow_state)
        if phase in _SEMANTIC_GATE_PHASES:
            return self._freeze_semantic_gate(phase, analysis_state, workflow_state)
        if phase in _TERMINAL_PHASES:
            return TerminalCapsule(graph_version=GRAPH_VERSION, status="completed")
        return IdleCapsule(graph_version=GRAPH_VERSION)

    def freeze_dynamic(
        self,
        live_safe_point: dict[str, Any] | None,
        analysis_state: Any = None,
        workflow_state: dict | None = None,
    ) -> ContinuationCapsule:
        """Derive capsule from the acknowledged live safe-point descriptor.

        Mapping rules:
        - active semantic gate → SemanticGateCapsule
        - active graph next node → GraphNodeCapsule
        - completed / terminal → TerminalCapsule
        - truly inactive → IdleCapsule
        """
        desc = dict(live_safe_point or {})

        # Explicit kind on descriptor wins
        kind = desc.get("continuation_kind") or desc.get("kind")
        if kind == "terminal" or desc.get("status") == "completed":
            return TerminalCapsule(graph_version=GRAPH_VERSION, status="completed")
        if kind == "idle":
            return IdleCapsule(graph_version=GRAPH_VERSION)

        # `inactive` means "nothing is executing / no kernel". That is orthogonal
        # to "parked at a resumable HITL gate", and the two coincide constantly:
        # the quiescent-coordinator descriptor (jobs.py) hard-codes
        # ``inactive: True`` for every non-terminal status, including the
        # ``waiting*`` statuses whose whole point is that the session IS parked
        # somewhere resumable. Short-circuiting to Idle here made the
        # status-derived gate below unreachable for exactly the producer it was
        # written for — pre_restore, the snapshot a user reaches for to undo a
        # restore — so the rollback point silently lost the gate. It still
        # suppresses the mid-node `next_node` branch, where "nothing is
        # executing" really does contradict a live graph position.
        inactive = desc.get("inactive") is True

        gate = desc.get("semantic_gate") or desc.get("gate")
        if gate:
            payload = {
                "gate": gate,
                "source": "live_safe_point",
                "current_round": desc.get("current_round", 0),
            }
            put_result = self._object_store.put_bytes(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(),
                "application/json",
            )
            ref = put_result.ref
            return SemanticGateCapsule(
                graph_version=GRAPH_VERSION,
                gate=str(gate),
                wait_reason=desc.get("wait_reason"),
                payload_ref=ref,
            )

        next_node = None if inactive else (
            desc.get("graph_node") or desc.get("next_node")
        )
        if next_node and next_node in _KNOWN_GRAPH_NODES:
            # Validate planning_stage before put_bytes so invalid stages fail
            # closed without writing objects (symmetric with restore-side check).
            raw_stage = desc.get("planning_stage", "full")
            if raw_stage not in ("stage1", "stage2", "full"):
                raise SnapshotCaptureError(
                    "INVALID_PLANNING_STAGE",
                    f"Live descriptor has invalid planning_stage: {raw_stage!r}",
                    details={
                        "planning_stage": raw_stage,
                        "next_node": str(next_node),
                    },
                )
            serialized = self._serialize_workflow_state(desc)
            put_result = self._object_store.put_bytes(
                json.dumps(serialized, sort_keys=True, ensure_ascii=False).encode(),
                "application/json",
            )
            ref = put_result.ref
            return GraphNodeCapsule(
                graph_version=GRAPH_VERSION,
                next_node=str(next_node),
                workflow_state_ref=ref,
                planning_stage=raw_stage,
            )


        # Truly inactive / unknown → idle
        # The descriptor-driven branches above read keys
        # (`gate`/`semantic_gate`/`graph_node`/`next_node`) that no producer in
        # the tree ever writes — WorkflowState declares none of them and the
        # three descriptor builders emit only kernel_present / current_round /
        # inactive / status. So every dynamic capture of a *running* session
        # fell through to Idle, including pre_restore: the one snapshot a user
        # reaches for to undo a restore recorded no position at all.
        #
        # The session's own status does record a live HITL gate, and a session
        # parked at a gate is exactly the case where "idle" is wrong and the
        # resume target is unambiguous. Anything else still yields Idle —
        # recovering a mid-node graph position needs the checkpointer's `next`,
        # which is not available here.
        gate = _gate_from_status(analysis_state)
        if gate is not None:
            return self._freeze_semantic_gate(
                _GATE_TO_PHASE.get(gate, gate),
                analysis_state,
                workflow_state or {},
            )

        return IdleCapsule(graph_version=GRAPH_VERSION)

    # ── private helpers ─────────────────────────────────────────────────────

    def _freeze_graph_node(self, phase: str, workflow_state: dict) -> GraphNodeCapsule:
        next_node = _PHASE_TO_NEXT_NODE.get(phase)
        if next_node not in _KNOWN_GRAPH_NODES:
            raise SnapshotCaptureError(
                "INVALID_NEXT_NODE",
                f"Phase {phase} maps to unknown graph node: {next_node}",
            )

        planning_stage = _PHASE_TO_PLANNING_STAGE.get(phase, "full")
        serialized = self._serialize_workflow_state(workflow_state)
        put_result = self._object_store.put_bytes(
            json.dumps(serialized, sort_keys=True, ensure_ascii=False).encode(),
            "application/json",
        )
        ref = put_result.ref

        return GraphNodeCapsule(
            graph_version=GRAPH_VERSION,
            next_node=next_node,
            workflow_state_ref=ref,
            planning_stage=planning_stage,
        )

    def _freeze_semantic_gate(
        self, phase: str, analysis_state: Any, workflow_state: dict
    ) -> SemanticGateCapsule:
        gate = _PHASE_TO_GATE.get(phase, phase)
        payload = {
            "phase": phase,
            "gate": gate,
            "current_round": workflow_state.get("current_round", 0),
            "planning_stage": workflow_state.get("planning_stage", "full"),
            "plan_approved": workflow_state.get("plan_approved", False),
            "failed_node": workflow_state.get("failed_node"),
            "skip_nodes": list(workflow_state.get("skip_nodes", [])),
            "integration_next": workflow_state.get("integration_next"),
        }

        put_result = self._object_store.put_bytes(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(),
            "application/json",
        )
        ref = put_result.ref

        return SemanticGateCapsule(
            graph_version=GRAPH_VERSION,
            gate=gate,
            wait_reason=None,
            payload_ref=ref,
        )

    @staticmethod
    def _serialize_workflow_state(workflow_state: dict) -> dict:
        carried = (
            "current_round",
            "pi_instructions",
            "planning_stage",
            "plan_approved",
            "planning_result",
            "planning_context",
            "user_plan_feedback",
            "plan_revision_needed",
            "plan_revision_context",
            "plan_revision_resume",
            "report_blocked",
            "integration_next",
            "failed_node",
            "skip_nodes",
            "node_retry_count",
            "node_errors",
            "adjusted_instructions",
        )
        result = {}
        for key in carried:
            if key in workflow_state:
                result[key] = workflow_state[key]
        return result
