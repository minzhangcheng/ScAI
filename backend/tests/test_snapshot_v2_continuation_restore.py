"""Tests for ContinuationDispatcher: exact restore targets, branch identity, drift detection."""
import pytest
from datetime import datetime
from typing import Any, Dict

from pydantic import TypeAdapter


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_analysis_state(**overrides):
    from backend.models.schemas import AnalysisState, TaskStatus, AnalysisPhase
    defaults = dict(
        id="test-session",
        user_id="u1",
        name="test",
        initial_instruction="analyze",
        status=TaskStatus.PAUSED,
        analysis_phase=AnalysisPhase.INNOVATION,
        current_round=2,
        max_rounds=4,
    )
    defaults.update(overrides)
    state = AnalysisState(**defaults)
    return state


def _make_workflow_state(**overrides):
    return {
        "analysis_state": _make_analysis_state(),
        "current_round": 2,
        "pi_instructions": {"direction": "test-dir"},
        "pathology_result": None,
        "bioinformatics_result": None,
        "biochemistry_result": None,
        "literature_result": None,
        "round_result": None,
        "should_continue": True,
        "error": None,
        "messages": [],
        "plan_approved": True,
        "planning_stage": "stage1",
        "planning_result": {"plan": "test-plan"},
        "planning_context": {"genes": ["G1", "G2"]},
        "user_plan_feedback": None,
        "plan_revision_needed": False,
        "plan_revision_context": None,
        "plan_revision_resume": False,
        "report_blocked": False,
        "integration_next": None,
        "failed_node": None,
        "skip_nodes": [],
        "node_retry_count": {},
        "node_errors": {},
        "human_guidance": None,
        "human_stop_requested": False,
        "human_guidance_processed": False,
        "user_action": None,
        "new_data_files": [],
        "adjusted_instructions": None,
        **overrides,
    }


def _make_graph_node_capsule(next_node="bioinformatics_analysis", planning_stage="stage1", *, object_store=None, workflow_state=None):
    from backend.services.snapshot_v2.models import GraphNodeCapsule, ObjectRef
    import json
    if object_store is not None and workflow_state is not None:
        # Extract only serializable fields (same as ContinuationRegistry._serialize_workflow_state)
        carried = (
            "current_round", "pi_instructions", "planning_stage", "plan_approved",
            "planning_result", "planning_context", "user_plan_feedback",
            "plan_revision_needed", "plan_revision_context", "plan_revision_resume",
            "report_blocked", "integration_next", "failed_node", "skip_nodes",
            "node_retry_count", "node_errors", "adjusted_instructions",
        )
        serialized = {}
        for key in carried:
            if key in workflow_state:
                serialized[key] = workflow_state[key]
        data = json.dumps(serialized, sort_keys=True, ensure_ascii=False).encode()
        ref = object_store.put_bytes(data, "application/json")
    else:
        ref = ObjectRef(
            sha256="ab" * 32,
            size=100,
            media_type="application/json",
        )
    return GraphNodeCapsule(
        graph_version="2.0",
        next_node=next_node,
        workflow_state_ref=ref,
        planning_stage=planning_stage,
    )


def _make_semantic_gate_capsule(gate="round_wait", *, object_store=None, payload=None):
    from backend.services.snapshot_v2.models import SemanticGateCapsule, ObjectRef
    import json
    if object_store is not None and payload is not None:
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ref = object_store.put_bytes(data, "application/json")
    else:
        ref = ObjectRef(
            sha256="cd" * 32,
            size=100,
            media_type="application/json",
        )
    return SemanticGateCapsule(
        graph_version="2.0",
        gate=gate,
        wait_reason=None,
        payload_ref=ref,
    )


def _make_terminal_capsule(status="completed"):
    from backend.services.snapshot_v2.models import TerminalCapsule
    return TerminalCapsule(
        graph_version="2.0",
        status=status,
    )


def _make_idle_capsule():
    from backend.services.snapshot_v2.models import IdleCapsule
    return IdleCapsule(graph_version="2.0")


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def v2_paths(tmp_path):
    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for d in [self.root, self.objects, self.staging, self.journals, self.integrity, self.committed]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


# ── Phase matrix tests ─────────────────────────────────────────────────────────

class TestPhaseMatrix:
    """Every fixed phase → exact next node/gate/terminal/idle behavior."""

    def test_graph_node_phase_pre_execution_stage1(self, v2_paths):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state(planning_stage="stage1")

        capsule = registry.freeze("pre_execution_stage1", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-1", transaction_id="tx-1",
        )

        assert staged.kind == "graph_node"
        assert staged.next_node == "bioinformatics_analysis"
        assert staged.planning_stage == "stage1"
        assert staged.workflow_state is not None
        assert staged.workflow_state["current_round"] == 2
        assert staged.workflow_state["plan_approved"] is True

    def test_graph_node_phase_pre_execution_stage2(self, v2_paths):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state(planning_stage="stage2")

        capsule = registry.freeze("pre_execution_stage2", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-2", transaction_id="tx-2",
        )

        assert staged.kind == "graph_node"
        assert staged.next_node == "bioinformatics_analysis"
        assert staged.planning_stage == "stage2"

    def test_graph_node_phase_pre_execution_full(self, v2_paths):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state(planning_stage="full")

        capsule = registry.freeze("pre_execution_full", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-3", transaction_id="tx-3",
        )

        assert staged.kind == "graph_node"
        assert staged.next_node == "bioinformatics_analysis"
        assert staged.planning_stage == "full"


    def test_graph_node_phase_pre_major_revision(self, v2_paths):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state(planning_stage="full")

        capsule = registry.freeze("pre_major_revision", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-4", transaction_id="tx-4",
        )

        assert staged.kind == "semantic_gate"
        assert staged.gate == "plan_revision"
        assert staged.payload is not None
        assert staged.payload.get("gate") == "plan_revision"


    def test_semantic_gate_phase_pre_annotation(self, v2_paths):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state()

        capsule = registry.freeze("pre_annotation", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-5", transaction_id="tx-5",
        )

        assert staged.kind == "semantic_gate"
        assert staged.gate == "annotation_confirmation"

    def test_semantic_gate_phase_pre_direction(self, v2_paths):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state()

        capsule = registry.freeze("pre_direction", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-6", transaction_id="tx-6",
        )

        assert staged.kind == "semantic_gate"
        assert staged.gate == "direction_selection"

    def test_semantic_gate_phase_pre_round_wait(self, v2_paths):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state()

        capsule = registry.freeze("pre_round_wait", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-7", transaction_id="tx-7",
        )

        assert staged.kind == "semantic_gate"
        assert staged.gate == "round_wait"
        # payload should contain current_round, planning_stage etc.
        assert staged.payload is not None
        assert staged.payload["current_round"] == 2

    def test_semantic_gate_phase_pre_integration_gate(self, v2_paths):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state()

        capsule = registry.freeze("pre_integration_gate", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-8", transaction_id="tx-8",
        )

        assert staged.kind == "semantic_gate"
        assert staged.gate == "integration_selection"

    def test_terminal_phase_post_report(self, v2_paths):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state()

        capsule = registry.freeze("post_report", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-9", transaction_id="tx-9",
        )

        assert staged.kind == "terminal"
        assert staged.terminal_status == "completed"
        # Terminal should not have next_node, gate, or planning_stage
        assert staged.next_node is None
        assert staged.gate is None

    def test_idle_phase_manual(self, v2_paths):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state()

        capsule = registry.freeze("manual", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-10", transaction_id="tx-10",
        )

        assert staged.kind == "idle"
        assert staged.next_node is None
        assert staged.gate is None

    def test_idle_phase_post_chat(self, v2_paths):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state()

        capsule = registry.freeze("post_chat", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-11", transaction_id="tx-11",
        )

        assert staged.kind == "idle"

    def test_idle_phase_pre_restore(self, v2_paths):
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state()

        capsule = registry.freeze("pre_restore", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-12", transaction_id="tx-12",
        )

        assert staged.kind == "idle"


# ── Activate tests ─────────────────────────────────────────────────────────────

class TestActivate:
    """Test that activate() correctly sets restore_resume_* fields."""

    def test_activate_graph_node_sets_fields(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        ws = _make_workflow_state()
        capsule = _make_graph_node_capsule(
            next_node="bioinformatics_analysis", planning_stage="stage1",
            object_store=store, workflow_state=ws,
        )
        analysis_state = _make_analysis_state()

        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-1", transaction_id="tx-abc",
        )
        dispatcher.activate(analysis_state, staged)

        assert analysis_state.restore_resume_kind == "graph_node"
        assert analysis_state.restore_resume_node == "bioinformatics_analysis"
        assert analysis_state.restore_branch_id == "restore-tx-abc"
        assert analysis_state.restore_branch_label == "branch-snap-1"

    def test_activate_semantic_gate_sets_fields(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher
        from backend.models.schemas import TaskStatus

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        capsule = _make_semantic_gate_capsule(
            gate="round_wait",
            object_store=store,
            payload={
                "current_round": 2,
                "planning_stage": "full",
                "wait_reason": "round_wait",
            },
        )
        analysis_state = _make_analysis_state(status="running")
        ws = _make_workflow_state()

        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-2", transaction_id="tx-def",
        )
        dispatcher.activate(analysis_state, staged)

        assert analysis_state.restore_resume_kind == "semantic_gate"
        assert analysis_state.restore_resume_gate == "round_wait"
        assert analysis_state.restore_branch_id == "restore-tx-def"
        assert analysis_state.wait_input_reason == "round_wait"
        assert analysis_state.status == TaskStatus.WAITING_INPUT
        assert analysis_state.current_round == 2
        ss = analysis_state.sandbox_state or {}
        assert ss.get("restore_gate_payload", {}).get("planning_stage") == "full"

    def test_activate_semantic_gate_fails_closed_without_payload(self, v2_paths):
        """Missing payload when capsule has payload_ref must fail closed."""
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import (
            ContinuationDispatcher, StagedCheckpoint,
        )
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state(status="running")
        staged = StagedCheckpoint(
            kind="semantic_gate",
            gate="round_wait",
            payload=None,
            branch_id="restore-tx-x",
            branch_label="branch-snap-x",
            parent_snapshot_id="snap-x",
            transaction_id="tx-x",
        )
        with pytest.raises(SnapshotCaptureError) as exc:
            dispatcher.activate(analysis_state, staged)
        assert exc.value.code == "MISSING_GATE_PAYLOAD"

    def test_activate_terminal_clears_workflow_fields(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        capsule = _make_terminal_capsule(status="completed")
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state()

        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-3", transaction_id="tx-ghi",
        )
        dispatcher.activate(analysis_state, staged)

        assert analysis_state.restore_resume_kind == "terminal"
        assert analysis_state.restore_resume_node is None
        assert analysis_state.restore_resume_gate is None

    def test_activate_idle_does_not_set_resume_fields(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        capsule = _make_idle_capsule()
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state()

        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-4", transaction_id="tx-jkl",
        )
        assert analysis_state.restore_resume_kind is None
        dispatcher.activate(analysis_state, staged)

        assert analysis_state.restore_resume_kind is None

    def test_activate_invalidates_report_paths(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        ws = _make_workflow_state()
        capsule = _make_graph_node_capsule(
            next_node="bioinformatics_analysis",
            object_store=store, workflow_state=ws,
        )
        analysis_state = _make_analysis_state()
        analysis_state.exploration_report_html_path = "/some/path/report.html"
        analysis_state.detailed_report_html_path = "/some/path/detailed.html"

        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-5", transaction_id="tx-mno",
        )
        dispatcher.activate(analysis_state, staged)

        assert analysis_state.exploration_report_html_path is None
        assert analysis_state.detailed_report_html_path is None

    def test_activate_sets_last_restored_snapshot_id(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        ws = _make_workflow_state()
        capsule = _make_graph_node_capsule(
            object_store=store, workflow_state=ws,
        )
        analysis_state = _make_analysis_state()

        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-parent-1", transaction_id="tx-pqr",
        )
        dispatcher.activate(analysis_state, staged)

        assert analysis_state.last_restored_snapshot_id == "snap-parent-1"


# ── Branch identity tests ─────────────────────────────────────────────────────

class TestBranchIdentity:
    """Branch identity uses transaction_id, sets parent, invalidates reports."""

    def test_branch_id_includes_transaction_id(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        ws = _make_workflow_state()
        capsule = _make_graph_node_capsule(
            object_store=store, workflow_state=ws,
        )
        analysis_state = _make_analysis_state()

        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-1", transaction_id="unique-tx-42",
        )
        assert staged.transaction_id == "unique-tx-42"
        assert staged.branch_id == "restore-unique-tx-42"

    def test_different_transactions_get_different_branch_ids(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        ws = _make_workflow_state()
        capsule = _make_graph_node_capsule(
            object_store=store, workflow_state=ws,
        )

        staged_a = dispatcher.stage_checkpoint(
            capsule, ws, _make_analysis_state(),
            snapshot_id="snap-1", transaction_id="tx-a",
        )
        staged_b = dispatcher.stage_checkpoint(
            capsule, ws, _make_analysis_state(),
            snapshot_id="snap-1", transaction_id="tx-b",
        )

        assert staged_a.branch_id != staged_b.branch_id


# ── Drift detection tests ─────────────────────────────────────────────────────

class TestDriftDetection:
    """Drifted targets fail before mutating state."""

    def test_unknown_next_node_in_graph_node_capsule_raises(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        ws = _make_workflow_state()
        capsule = _make_graph_node_capsule(
            next_node="nonexistent_node",
            object_store=store, workflow_state=ws,
        )
        analysis_state = _make_analysis_state()

        with pytest.raises(SnapshotCaptureError) as exc:
            dispatcher.stage_checkpoint(
                capsule, ws, analysis_state,
                snapshot_id="snap-1", transaction_id="tx-err",
            )
        assert exc.value.code == "UNKNOWN_NEXT_NODE"

    def test_unknown_gate_in_semantic_gate_capsule_raises(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        ws = _make_workflow_state()
        capsule = _make_semantic_gate_capsule(
            gate="nonexistent_gate",
            object_store=store, payload={"current_round": 2},
        )
        analysis_state = _make_analysis_state()

        with pytest.raises(SnapshotCaptureError) as exc:
            dispatcher.stage_checkpoint(
                capsule, ws, analysis_state,
                snapshot_id="snap-1", transaction_id="tx-err2",
            )
        assert exc.value.code == "UNKNOWN_GATE"

    def test_analysis_state_unchanged_after_drift_failure(self, v2_paths):
        """Drift must NOT mutate AnalysisState."""
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)

        ws = _make_workflow_state()
        capsule = _make_graph_node_capsule(
            next_node="nonexistent_node",
            object_store=store, workflow_state=ws,
        )
        analysis_state = _make_analysis_state()
        original_branch = analysis_state.restore_branch_id

        try:
            dispatcher.stage_checkpoint(
                capsule, ws, analysis_state,
                snapshot_id="snap-1", transaction_id="tx-safe",
            )
        except SnapshotCaptureError:
            pass

        assert analysis_state.restore_branch_id == original_branch
        assert analysis_state.restore_resume_kind is None


# ── WorkflowState match tests ─────────────────────────────────────────────────

class TestWorkflowStateMatch:
    """Payload/round/plan/direction/retry state must match."""

    def test_staged_checkpoint_carries_workflow_state(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher
        from backend.services.snapshot_v2.continuation import ContinuationRegistry

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state(
            current_round=3,
            plan_approved=True,
            planning_stage="stage2",
            planning_result={"key": "value"},
            failed_node="bioinformatics_analysis",
            skip_nodes=["pathology_analysis"],
            node_retry_count={"bioinformatics_analysis": 2},
            node_errors={"bioinformatics_analysis": "LLM timeout"},
            adjusted_instructions={"use_scanpy": True},
        )

        capsule = registry.freeze("pre_execution_stage2", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-1", transaction_id="tx-1",
        )

        assert staged.workflow_state is not None
        assert staged.workflow_state["current_round"] == 3
        assert staged.workflow_state["plan_approved"] is True
        assert staged.workflow_state["planning_stage"] == "stage2"
        assert staged.workflow_state["planning_result"]["key"] == "value"
        assert staged.workflow_state["failed_node"] == "bioinformatics_analysis"
        assert staged.workflow_state["skip_nodes"] == ["pathology_analysis"]
        assert staged.workflow_state["node_retry_count"]["bioinformatics_analysis"] == 2
        assert staged.workflow_state["node_errors"]["bioinformatics_analysis"] == "LLM timeout"
        assert staged.workflow_state["adjusted_instructions"]["use_scanpy"] is True

    def test_staged_semantic_gate_payload_matches_expected_fields(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher
        from backend.services.snapshot_v2.continuation import ContinuationRegistry

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        dispatcher = ContinuationDispatcher(store)
        analysis_state = _make_analysis_state()
        ws = _make_workflow_state(
            current_round=2,
            planning_stage="full",
            plan_approved=True,
            failed_node=None,
            skip_nodes=[],
            integration_next=None,
        )

        capsule = registry.freeze("pre_round_wait", analysis_state, ws)
        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-1", transaction_id="tx-1",
        )

        assert staged.payload is not None
        assert staged.payload["current_round"] == 2
        assert staged.payload["planning_stage"] == "full"
        assert staged.payload["plan_approved"] is True
        assert staged.payload["integration_next"] is None


# ── ID dispatching tests ──────────────────────────────────────────────────────

class TestIdDispatching:
    """Direct capsule → dispatcher tests (bypassing ContinuationRegistry)."""

    def test_dispatch_graph_node_directly(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        ws = _make_workflow_state()
        capsule = _make_graph_node_capsule(
            next_node="bioinformatics_analysis", planning_stage="stage1",
            object_store=store, workflow_state=ws,
        )

        staged = dispatcher.stage_checkpoint(
            capsule, ws, _make_analysis_state(),
            snapshot_id="snap-1", transaction_id="tx-1",
        )
        assert staged.kind == "graph_node"
        assert staged.next_node == "bioinformatics_analysis"

    def test_dispatch_semantic_gate_directly(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        ws = _make_workflow_state()
        capsule = _make_semantic_gate_capsule(
            gate="annotation_confirmation",
            object_store=store, payload={"current_round": 2},
        )

        staged = dispatcher.stage_checkpoint(
            capsule, ws, _make_analysis_state(),
            snapshot_id="snap-1", transaction_id="tx-1",
        )
        assert staged.kind == "semantic_gate"
        assert staged.gate == "annotation_confirmation"

    def test_dispatch_terminal_directly(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        capsule = _make_terminal_capsule(status="stopped")
        ws = _make_workflow_state()

        staged = dispatcher.stage_checkpoint(
            capsule, ws, _make_analysis_state(),
            snapshot_id="snap-1", transaction_id="tx-1",
        )
        assert staged.kind == "terminal"
        assert staged.terminal_status == "stopped"

    def test_dispatch_idle_directly(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher

        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)
        capsule = _make_idle_capsule()
        ws = _make_workflow_state()

        staged = dispatcher.stage_checkpoint(
            capsule, ws, _make_analysis_state(),
            snapshot_id="snap-1", transaction_id="tx-1",
        )
        assert staged.kind == "idle"


# ── Module-level existence guard ──────────────────────────────────────────────

def test_continuation_dispatcher_importable():
    from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher
    assert hasattr(ContinuationDispatcher, "stage_checkpoint")
    assert hasattr(ContinuationDispatcher, "activate")


def test_staged_checkpoint_model_importable():
    from backend.services.snapshot_v2.dispatcher import StagedCheckpoint
    from pydantic import BaseModel
    assert issubclass(StagedCheckpoint, BaseModel)


# ── Continuation with workflow core dispatch (integration-level) ─────────────

class TestWorkflowCoreDispatch:
    """After restore, core routes correctly based on restore_resume_kind."""

    @pytest.mark.asyncio
    async def test_graph_node_resume_routes_to_workflow(self):
        """GraphNode kind should start the graph workflow."""
        from unittest.mock import AsyncMock, MagicMock
        from backend.models.schemas import AnalysisState, TaskStatus, AnalysisPhase
        from backend.workflow.core import SingleCellAnalysisWorkflow

        state = AnalysisState(
            id="s1", user_id="u1", initial_instruction="test",
            status=TaskStatus.RUNNING,
            current_round=2, max_rounds=4,
            restore_resume_kind="graph_node",
            restore_resume_node="bioinformatics_analysis",
            restore_branch_id="restore-tx-1",
        )

        workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
        workflow.session_id = "s1"
        workflow._state = state
        workflow._send_message = AsyncMock()
        workflow._send_state_sync = AsyncMock()
        workflow.initialize = AsyncMock()
        workflow.compiled_graph = MagicMock()
        workflow.compiled_graph.ainvoke = AsyncMock(return_value={
            "analysis_state": state,
        })
        workflow._set_status = MagicMock()

        result = await workflow.run(state, "s1")
        workflow.compiled_graph.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_terminal_resume_does_not_create_workflow(self):
        """Terminal kind should skip graph creation entirely."""
        from unittest.mock import AsyncMock, MagicMock
        from backend.models.schemas import AnalysisState, TaskStatus, AnalysisPhase
        from backend.workflow.core import SingleCellAnalysisWorkflow

        state = AnalysisState(
            id="s2", user_id="u1", initial_instruction="test",
            status=TaskStatus.RUNNING,
            current_round=2, max_rounds=4,
            restore_resume_kind="terminal",
            restore_branch_id="restore-tx-2",
        )

        workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
        workflow.session_id = "s2"
        workflow._state = state
        workflow._send_message = AsyncMock()
        workflow._send_state_sync = AsyncMock()
        workflow.initialize = AsyncMock()
        workflow.compiled_graph = None
        # Use the real _set_status so state is actually modified
        set_status_mock = MagicMock(wraps=SingleCellAnalysisWorkflow._set_status)
        workflow._set_status = set_status_mock

        result = await workflow.run(state, "s2")
        assert result.status == TaskStatus.COMPLETED
        workflow.initialize.assert_not_awaited()
        set_status_mock.assert_called_once_with(state, TaskStatus.COMPLETED)

    @pytest.mark.asyncio
    async def test_idle_resume_does_not_auto_start(self):
        """Idle kind should not auto-start workflow."""
        from unittest.mock import AsyncMock, MagicMock
        from backend.models.schemas import AnalysisState, TaskStatus, AnalysisPhase
        from backend.workflow.core import SingleCellAnalysisWorkflow

        state = AnalysisState(
            id="s3", user_id="u1", initial_instruction="test",
            status=TaskStatus.RUNNING,
            current_round=2, max_rounds=4,
            restore_resume_kind=None,
            restore_branch_id="restore-tx-3",
        )

        workflow = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
        workflow.session_id = "s3"
        workflow._state = state
        workflow._send_message = AsyncMock()
        workflow._send_state_sync = AsyncMock()
        workflow.initialize = AsyncMock()
        workflow.compiled_graph = None
        workflow._set_status = MagicMock()

        result = await workflow.run(state, "s3")
        workflow.initialize.assert_not_awaited()
