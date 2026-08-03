"""Continuation dispatcher for snapshot V2 — translates frozen capsules into
resume-ready AnalysisState metadata with drift detection and branch identity.

Provides :class:`ContinuationDispatcher` with two public methods:

    * ``stage_checkpoint(capsule, state, analysis_state, …) → StagedCheckpoint``
    * ``activate(analysis_state, staged, workspace_path=...) → None``

The caller (e.g. restore coordinator or snapshot route) stages the checkpoint
against the current WorkflowState, then activates it to mutate AnalysisState.
"""

import copy
import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from backend.constants import GATE_ANNOTATION_CONFIRMATION, WORKFLOW_GRAPH_NODES
from backend.services.snapshot_v2.errors import SnapshotCaptureError
from backend.services.snapshot_v2.object_store import ObjectStore

logger = logging.getLogger(__name__)


# ── Known graph node names (must match WORKFLOW_GRAPH_NODES) ──────────────────

# Capsule targets exclude the restore_entry node (internal graph entry point).
_KNOWN_GRAPH_NODES: frozenset[str] = frozenset(
    n for n in WORKFLOW_GRAPH_NODES if n != "restore_entry"
)

# ── Known semantic gate names (must match constants.py) ───────────────────────

_KNOWN_GATES: frozenset[str] = frozenset({
    "round_wait",
    "direction_selection",
    "integration_selection",
    "annotation_confirmation",
    "plan_review",
    "plan_revision",
})


# Gate → the TaskStatus its live counterpart announces. Each HITL endpoint
# validates the one status it owns, so a restored session parked on the wrong
# one cannot be acted on at all.
#
#   annotation_confirmation → workflow/annotation.py announces
#       WAITING_ANNOTATION; routes/interaction.py::confirm_annotations rejects
#       anything else.
#   plan_review            → workflow/planning.py announces
#       WAITING_PLAN_REVIEW; routes/interaction.py::plan_review rejects
#       anything else.
#
# round_wait / direction_selection / integration_selection are announced as
# WAITING_INPUT by workflow/synthesis.py and workflow/integration.py, and
# `plan_revision` is an internal routing decision with no dedicated status, so
# all four keep the WAITING_INPUT default.
_GATE_TO_STATUS_NAMES: dict[str, str] = {
    GATE_ANNOTATION_CONFIRMATION: "WAITING_ANNOTATION",
    "plan_review": "WAITING_PLAN_REVIEW",
}



# ── StagedCheckpoint model ────────────────────────────────────────────────────


class StagedCheckpoint(BaseModel):
    """Validated checkpoint metadata ready to be activated on AnalysisState.

    Created by ``ContinuationDispatcher.stage_checkpoint()`` after drift
    validation.  The caller passes it to ``activate()`` to mutate the live
    ``AnalysisState``.
    """

    kind: str  # "graph_node" | "semantic_gate" | "terminal" | "idle"

    # Graph node fields
    next_node: Optional[str] = None
    planning_stage: Optional[str] = None
    workflow_state: Optional[dict] = None  # deserialised payload from ObjectRef

    # Semantic gate fields
    gate: Optional[str] = None
    payload: Optional[dict] = None  # deserialised payload from ObjectRef

    # Terminal fields
    terminal_status: Optional[str] = None

    # Branch identity (always set)
    branch_id: str
    branch_label: str
    parent_snapshot_id: str
    transaction_id: str


# ── ContinuationDispatcher ────────────────────────────────────────────────────


class ContinuationDispatcher:
    """Validates a ContinuationCapsule and produces resume-ready metadata.

    Usage::

        dispatcher = ContinuationDispatcher(object_store)
        staged = dispatcher.stage_checkpoint(capsule, workflow_state, analysis_state,
                                              snapshot_id="snap-1", transaction_id="tx-1")
        dispatcher.activate(analysis_state, staged)
    """

    def __init__(self, object_store: ObjectStore):
        self._object_store = object_store

    # ── Public API ─────────────────────────────────────────────────────

    def stage_checkpoint(
        self,
        capsule: Any,
        workflow_state: dict[str, Any],
        analysis_state: Any,
        *,
        snapshot_id: str,
        transaction_id: str,
    ) -> StagedCheckpoint:
        """Validate the capsule's target against the workflow state, produce
        a ``StagedCheckpoint``.

        Raises ``SnapshotCaptureError`` (via ``_check_drift``) if the capsule
        contains an unknown node or gate, or if required payloads can't be
        loaded from the object store.

        The returned ``StagedCheckpoint`` carries branch identity derived from
        *transaction_id* and *snapshot_id*.
        """
        kind = getattr(capsule, "kind", None)

        branch_id = f"restore-{transaction_id}"
        label = f"branch-{snapshot_id[:8]}"

        if kind == "graph_node":
            return self._stage_graph_node(
                capsule, workflow_state, snapshot_id, transaction_id, branch_id, label,
            )
        elif kind == "semantic_gate":
            return self._stage_semantic_gate(
                capsule, snapshot_id, transaction_id, branch_id, label,
            )
        elif kind == "terminal":
            return self._stage_terminal(
                capsule, snapshot_id, transaction_id, branch_id, label,
            )
        elif kind == "idle":
            return self._stage_idle(
                snapshot_id, transaction_id, branch_id, label,
            )
        else:
            raise SnapshotCaptureError(
                "UNKNOWN_CAPSULE_KIND",
                f"Capsule kind '{kind}' is not a recognised continuation type",
            )

    def activate(
        self,
        analysis_state: Any,
        staged: StagedCheckpoint,
        *,
        workspace_path: Any = None,
    ) -> None:
        """Atomically set resume metadata fields on *analysis_state*.

        Branch activation spans several independently fallible operations
        (round-result cloning, figure archival, and provenance seeding). Build
        the complete result on a detached state first, then replace the live
        model storage in one step. An exception therefore leaves the caller's
        original AnalysisState untouched.
        """
        candidate = (
            analysis_state.model_copy(deep=True)
            if isinstance(analysis_state, BaseModel)
            else copy.deepcopy(analysis_state)
        )
        self._activate_in_place(
            candidate,
            staged,
            workspace_path=workspace_path,
        )
        self._swap_state(analysis_state, candidate)

    def _activate_in_place(
        self,
        analysis_state: Any,
        staged: StagedCheckpoint,
        *,
        workspace_path: Any = None,
    ) -> None:
        """Build one fully activated state on a detached candidate.

        Exact continuation only — no fresh-run fallback:
        - ``semantic_gate``: rebuild exact waiting payload
        - ``graph_node``: seed exact next-node checkpoint
        - ``terminal``: do not start a workflow
        - ``idle``: load content without starting analysis

        All four kinds set branch identity and invalidate report paths.
        """
        if staged.kind == "graph_node":
            analysis_state.restore_resume_kind = "graph_node"
            analysis_state.restore_resume_node = staged.next_node
            analysis_state.restore_resume_gate = None
            analysis_state.restore_resume_workflow_state = (
                copy.deepcopy(staged.workflow_state) if staged.workflow_state else {}
            )
            self._clear_gate_payload(analysis_state)
            if staged.planning_stage is not None and hasattr(analysis_state, "sandbox_state"):
                ss = dict(getattr(analysis_state, "sandbox_state", None) or {})
                ss["planning_stage"] = staged.planning_stage
                analysis_state.sandbox_state = ss
        elif staged.kind == "semantic_gate":
            analysis_state.restore_resume_kind = "semantic_gate"
            analysis_state.restore_resume_gate = staged.gate
            analysis_state.restore_resume_node = (
                staged.payload.get("next_node")
                or staged.payload.get("node_name")
            ) if staged.payload else None
            # Rebuild exact waiting payload from capsule.
            # Fail closed if capsule declared payload_ref but stage lost it.
            if staged.payload is None:
                raise SnapshotCaptureError(
                    "MISSING_GATE_PAYLOAD",
                    "semantic_gate staged checkpoint requires payload when "
                    "capsule declares payload_ref",
                )
            workflow_fields = {
                key: copy.deepcopy(staged.payload[key])
                for key in (
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
                if key in staged.payload
            }
            analysis_state.restore_resume_workflow_state = workflow_fields
            wait_reason = (
                staged.payload.get("wait_reason")
                or staged.payload.get("wait_input_reason")
                or getattr(staged, "wait_reason", None)
                or staged.gate
            )
            analysis_state.wait_input_reason = wait_reason
            from backend.models.schemas import TaskStatus
            # The gate - not the capsule kind - decides the status; see
            # _GATE_TO_STATUS_NAMES above for the per-gate rationale. A gate that
            # is restored onto the wrong status commits cleanly (live_verify only
            # compares the gate itself) and is then unreachable: the restore has
            # already evicted the workflow object, so the route's no-workflow
            # branch — gated on exactly the status that was not written — is the
            # only path left.
            gate_status = getattr(
                TaskStatus,
                _GATE_TO_STATUS_NAMES.get(staged.gate or "", "WAITING_INPUT"),
                TaskStatus.WAITING_INPUT,
            )
            try:
                analysis_state.status = gate_status
            except Exception:
                pass
            # Rebuild gate payload fields present on AnalysisState.
            if "current_round" in staged.payload and hasattr(
                analysis_state, "current_round",
            ):
                try:
                    analysis_state.current_round = int(staged.payload["current_round"])
                except (TypeError, ValueError):
                    pass
            for field in (
                "research_direction_candidates",
                "raw_direction_candidates",
                "direction_selection_summary",
                "cluster_annotations",
                "pending_user_action",
                "integration_next",
            ):
                if field in staged.payload and hasattr(analysis_state, field):
                    try:
                        setattr(analysis_state, field, staged.payload[field])
                    except Exception:
                        pass
            exploration_report = getattr(analysis_state, "exploration_report", None)
            if exploration_report is not None:
                for field in (
                    "research_direction_candidates",
                    "raw_direction_candidates",
                    "direction_selection_summary",
                ):
                    if field in staged.payload:
                        setattr(
                            exploration_report,
                            field,
                            copy.deepcopy(staged.payload[field]),
                        )
            if "annotation_state" in staged.payload:
                from backend.models.schemas import AnnotationConfirmationState

                raw_annotation = staged.payload.get("annotation_state")
                if isinstance(raw_annotation, dict):
                    analysis_state.annotation_state = (
                        AnnotationConfirmationState.model_validate(raw_annotation)
                    )
            # Stash full payload for resume consumers.
            if hasattr(analysis_state, "sandbox_state"):
                ss = dict(getattr(analysis_state, "sandbox_state", None) or {})
                ss["restore_gate_payload"] = dict(staged.payload)
                if "planning_stage" in staged.payload:
                    ss["planning_stage"] = staged.payload["planning_stage"]
                # skip_nodes is the one frozen field with a real consumer:
                # resume() reads sandbox_state["skip_nodes"] and injects it into
                # the graph state so @with_retry honours a Skip the user already
                # chose (core.py). Capture froze it, but activate only stashed it
                # under restore_gate_payload — a key nothing reads — so every
                # restore silently discarded those decisions and re-entered the
                # node the user had skipped past.
                if staged.payload.get("skip_nodes"):
                    ss["skip_nodes"] = list(staged.payload["skip_nodes"])
                analysis_state.sandbox_state = ss
        elif staged.kind == "terminal":
            analysis_state.restore_resume_kind = "terminal"
            analysis_state.restore_resume_node = None
            analysis_state.restore_resume_gate = None
            analysis_state.restore_resume_workflow_state = None
            self._clear_gate_payload(analysis_state)
            from backend.models.schemas import TaskStatus
            try:
                status = staged.terminal_status or "completed"
                analysis_state.status = (
                    TaskStatus.COMPLETED if status == "completed" else TaskStatus.STOPPED
                )
            except Exception:
                pass
        elif staged.kind == "idle":
            # Idle restores must not inherit a one-shot continuation from the
            # pre-restore live session.
            analysis_state.restore_resume_kind = None
            analysis_state.restore_resume_node = None
            analysis_state.restore_resume_gate = None
            analysis_state.restore_resume_workflow_state = None
            self._clear_gate_payload(analysis_state)
            from backend.models.schemas import TaskStatus
            try:
                if getattr(analysis_state, "status", None) == TaskStatus.RUNNING:
                    analysis_state.status = TaskStatus.PAUSED
            except Exception:
                pass
        else:
            raise SnapshotCaptureError(
                "UNKNOWN_CAPSULE_KIND",
                f"Cannot activate unknown capsule kind: {staged.kind!r}",
            )

        # Always set branch identity. Preserve source-lineage evidence and add a
        # branch-local inherited view, mirroring the round-result lineage model.
        source_lineage_id = (
            getattr(analysis_state, "restore_branch_id", None) or "main"
        )
        target_lineage_id = staged.branch_id or "main"
        if target_lineage_id != source_lineage_id:
            from backend.workflow.direction_coverage import (
                clone_round_results_for_branch,
            )
            from backend.services.report.lineage import (
                clone_publication_figures_for_branch,
            )

            analysis_state.round_results = clone_round_results_for_branch(
                list(getattr(analysis_state, "round_results", None) or []),
                branch_id=target_lineage_id,
                branch_label=staged.branch_label,
                source_snapshot_id=staged.parent_snapshot_id or "",
                source_lineage_id=source_lineage_id,
            )
            analysis_state.publication_figures = clone_publication_figures_for_branch(
                getattr(analysis_state, "publication_figures", None) or [],
                branch_id=target_lineage_id,
                source_lineage_id=source_lineage_id,
                workspace_path=workspace_path,
            )
        analysis_state.restore_branch_id = staged.branch_id
        analysis_state.restore_branch_label = staged.branch_label
        analysis_state.last_restored_snapshot_id = staged.parent_snapshot_id
        from backend.workflow.provenance import seed_restore_provenance
        seed_restore_provenance(analysis_state)

        # Snapshot objects do not capture reports/. Keeping any pre-restore path
        # would bind restored business state to uncaptured report bytes from a
        # different generation. Terminal restores can be regenerated manually;
        # an explicit unavailable report is safer than a plausible stale one.
        self._invalidate_report_paths(analysis_state)

    @staticmethod
    def _swap_state(target: Any, candidate: Any) -> None:
        """Publish a completed activation without field-by-field exposure."""
        # AnalysisState is a Pydantic model whose data live in ``__dict__``;
        # assigning the prepared mapping is the copy-and-swap publication
        # point. Keep Pydantic bookkeeping consistent for exclude-unset and
        # private/extra-field behaviour.
        target.__dict__ = candidate.__dict__
        for name in (
            "__pydantic_fields_set__",
            "__pydantic_extra__",
            "__pydantic_private__",
        ):
            if hasattr(candidate, name):
                object.__setattr__(target, name, getattr(candidate, name))

    @staticmethod
    def _clear_gate_payload(analysis_state: Any) -> None:
        if not hasattr(analysis_state, "sandbox_state"):
            return
        sandbox_state = dict(getattr(analysis_state, "sandbox_state", None) or {})
        sandbox_state.pop("restore_gate_payload", None)
        analysis_state.sandbox_state = sandbox_state

    # ── Stage helpers (private) ────────────────────────────────────────

    def _stage_graph_node(
        self,
        capsule: Any,
        workflow_state: dict[str, Any],
        snapshot_id: str,
        transaction_id: str,
        branch_id: str,
        label: str,
    ) -> StagedCheckpoint:
        next_node = getattr(capsule, "next_node", None)
        self._check_known_node(next_node)

        planning_stage = getattr(capsule, "planning_stage", "full")

        # Deserialise the workflow_state_ref if present
        ref = getattr(capsule, "workflow_state_ref", None)
        loaded_state: Optional[dict] = None
        if ref is not None:
            try:
                raw = self._object_store.get_bytes(ref.sha256)
                loaded_state = json.loads(raw.decode())
            except Exception as exc:
                raise SnapshotCaptureError(
                    "WORKFLOW_STATE_LOAD_FAILED",
                    f"Failed to load workflow state ref {ref.sha256}: {exc}",
                )

        return StagedCheckpoint(
            kind="graph_node",
            next_node=next_node,
            planning_stage=planning_stage,
            workflow_state=loaded_state,
            branch_id=branch_id,
            branch_label=label,
            parent_snapshot_id=snapshot_id,
            transaction_id=transaction_id,
        )

    def _stage_semantic_gate(
        self,
        capsule: Any,
        snapshot_id: str,
        transaction_id: str,
        branch_id: str,
        label: str,
    ) -> StagedCheckpoint:
        gate = getattr(capsule, "gate", None)
        self._check_known_gate(gate)

        # Deserialise the payload_ref — fail closed when declared but missing.
        ref = getattr(capsule, "payload_ref", None)
        loaded_payload: Optional[dict] = None
        if ref is None:
            raise SnapshotCaptureError(
                "MISSING_PAYLOAD_REF",
                "semantic_gate capsule declares no payload_ref; cannot rebuild gate",
            )
        try:
            raw = self._object_store.get_bytes(ref.sha256)
            loaded_payload = json.loads(raw.decode())
        except SnapshotCaptureError:
            raise
        except Exception as exc:
            raise SnapshotCaptureError(
                "PAYLOAD_LOAD_FAILED",
                f"Failed to load payload ref {ref.sha256}: {exc}",
            )
        if not isinstance(loaded_payload, dict):
            raise SnapshotCaptureError(
                "MISSING_GATE_PAYLOAD",
                "semantic_gate payload_ref did not yield a payload object",
            )

        return StagedCheckpoint(
            kind="semantic_gate",
            gate=gate,
            payload=loaded_payload,
            branch_id=branch_id,
            branch_label=label,
            parent_snapshot_id=snapshot_id,
            transaction_id=transaction_id,
        )

    def _stage_terminal(
        self,
        capsule: Any,
        snapshot_id: str,
        transaction_id: str,
        branch_id: str,
        label: str,
    ) -> StagedCheckpoint:
        status = getattr(capsule, "status", "completed")
        return StagedCheckpoint(
            kind="terminal",
            terminal_status=status,
            branch_id=branch_id,
            branch_label=label,
            parent_snapshot_id=snapshot_id,
            transaction_id=transaction_id,
        )

    def _stage_idle(
        self,
        snapshot_id: str,
        transaction_id: str,
        branch_id: str,
        label: str,
    ) -> StagedCheckpoint:
        return StagedCheckpoint(
            kind="idle",
            branch_id=branch_id,
            branch_label=label,
            parent_snapshot_id=snapshot_id,
            transaction_id=transaction_id,
        )

    # ── Drift detection ────────────────────────────────────────────────

    def _check_known_node(self, node: Optional[str]) -> None:
        if node is None:
            raise SnapshotCaptureError(
                "MISSING_NEXT_NODE",
                "Graph-node capsule has no next_node",
            )
        if node not in _KNOWN_GRAPH_NODES:
            raise SnapshotCaptureError(
                "UNKNOWN_NEXT_NODE",
                f"Graph-node capsule targets unknown graph node: {node}",
            )

    def _check_known_gate(self, gate: Optional[str]) -> None:
        if gate is None:
            raise SnapshotCaptureError(
                "MISSING_GATE",
                "Semantic-gate capsule has no gate",
            )
        if gate not in _KNOWN_GATES:
            raise SnapshotCaptureError(
                "UNKNOWN_GATE",
                f"Semantic-gate capsule targets unknown gate: {gate}",
            )

    # ── Report path invalidation ───────────────────────────────────────

    @staticmethod
    def _invalidate_report_paths(analysis_state: Any) -> None:
        """Clear generated report paths so a restore triggers fresh generation."""
        analysis_state.exploration_report_html_path = None
        analysis_state.detailed_report_html_path = None
        analysis_state.annotation_report_html_path = None
        analysis_state.integration_report_html_path = None
