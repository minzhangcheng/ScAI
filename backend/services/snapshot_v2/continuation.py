"""Continuation capsule registry for snapshot V2.

Maps SnapshotPhase values to typed ContinuationCapsule variants and
serialises required workflow state into the object store.
"""

from __future__ import annotations

import json
from typing import Any

from backend.constants import WORKFLOW_GRAPH_NODES
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
            return self.freeze_dynamic(live_safe_point or workflow_state or {})

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
        if kind == "idle" or desc.get("inactive") is True:
            return IdleCapsule(graph_version=GRAPH_VERSION)

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

        next_node = desc.get("graph_node") or desc.get("next_node")
        if next_node and next_node in _KNOWN_GRAPH_NODES:
            serialized = self._serialize_workflow_state(desc)
            put_result = self._object_store.put_bytes(
                json.dumps(serialized, sort_keys=True, ensure_ascii=False).encode(),
                "application/json",
            )
            ref = put_result.ref
            # Validate planning_stage before constructing the capsule so an
            # invalid live-descriptor value surfaces as a typed capture error
            # (symmetric with the restore-side check) instead of a raw
            # pydantic ValidationError.
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
            return GraphNodeCapsule(
                graph_version=GRAPH_VERSION,
                next_node=str(next_node),
                workflow_state_ref=ref,
                planning_stage=raw_stage,
            )

        # Truly inactive / unknown → idle
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
