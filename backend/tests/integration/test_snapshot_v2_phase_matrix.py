"""Integration tests: exact continuation phase matrix for Snapshot V2.

Parameterizes all fixed phases and verifies capsule type, next node/gate,
round/phase identity, and prior-node counters.
"""

import pytest

from backend.services.snapshot_v2.continuation import (
    GRAPH_VERSION,
    ContinuationRegistry,
    _GRAPH_NODE_PHASES,
    _SEMANTIC_GATE_PHASES,
    _TERMINAL_PHASES,
    _IDLE_PHASES,
    _PHASE_TO_NEXT_NODE,
    _PHASE_TO_PLANNING_STAGE,
    _PHASE_TO_GATE,
)


def _make_minimal_analysis_state():
    from backend.models.schemas import AnalysisState
    return AnalysisState(id="phase-test", name="test", user_id="u1", initial_instruction="analyze")


def _make_minimal_workflow_state(analysis_state):
    return {
        "analysis_state": analysis_state,
        "current_round": 2,
        "pi_instructions": {"key": "val"},
        "plan_approved": True,
        "planning_stage": "stage1",
        "planning_result": {"plan": "test"},
        "failed_node": None,
        "skip_nodes": [],
        "node_retry_count": {},
        "node_errors": {},
        "integration_next": None,
        "report_blocked": False,
    }


class TestFixedPhaseContinuation:
    @pytest.fixture
    def object_store(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        return ObjectStore(v2_paths)

    @pytest.fixture
    def registry(self, object_store):
        return ContinuationRegistry(object_store)

    def test_all_graph_node_phases_produce_graph_node_capsule(self, registry, v2_paths):
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        for phase in sorted(_GRAPH_NODE_PHASES):
            capsule = registry.freeze(phase, analysis_state, ws)
            assert capsule.kind == "graph_node", f"{phase} should produce graph_node"
            assert capsule.graph_version == GRAPH_VERSION
            assert capsule.next_node == _PHASE_TO_NEXT_NODE[phase]
            assert capsule.planning_stage == _PHASE_TO_PLANNING_STAGE[phase]
            assert capsule.workflow_state_ref.sha256 != "0" * 64

    def test_all_semantic_gate_phases_produce_semantic_gate_capsule(self, registry, v2_paths):
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        for phase in sorted(_SEMANTIC_GATE_PHASES):
            capsule = registry.freeze(phase, analysis_state, ws)
            assert capsule.kind == "semantic_gate", f"{phase} should produce semantic_gate"
            assert capsule.graph_version == GRAPH_VERSION
            assert capsule.gate == _PHASE_TO_GATE[phase]
            assert capsule.payload_ref.sha256 != "0" * 64

    def test_post_report_produces_terminal(self, registry, v2_paths):
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        capsule = registry.freeze("post_report", analysis_state, ws)
        assert capsule.kind == "terminal"
        assert capsule.graph_version == GRAPH_VERSION
        assert capsule.status == "completed"

    def test_idle_phases_produce_idle_capsule(self, registry, v2_paths):
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        for phase in sorted(_IDLE_PHASES):
            capsule = registry.freeze(phase, analysis_state, ws)
            assert capsule.kind == "idle", f"{phase} should produce idle"
            assert capsule.graph_version == GRAPH_VERSION

    def test_unknown_phase_raises(self, registry, v2_paths):
        from backend.services.snapshot_v2.errors import SnapshotCaptureError
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        with pytest.raises(SnapshotCaptureError) as exc:
            registry.freeze("unknown_phase", analysis_state, ws)
        assert "UNKNOWN_PHASE" in str(exc.value.code) or "UNKNOWN_PHASE" in str(exc.value)

    def test_workflow_state_serialization_preserves_fields(self, registry, v2_paths):
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)
        ws["failed_node"] = "pathology_analysis"
        ws["skip_nodes"] = ["biochemistry_analysis"]
        ws["node_retry_count"] = {"pathology_analysis": 2}
        ws["node_errors"] = {"pathology_analysis": "some error"}

        capsule = registry.freeze("pre_execution_stage1", analysis_state, ws)

        from backend.services.snapshot_v2.object_store import ObjectStore
        store = ObjectStore(v2_paths)
        raw = store.get_bytes(capsule.workflow_state_ref.sha256)
        import json
        loaded = json.loads(raw.decode())

        assert loaded.get("current_round") == 2
        assert loaded.get("failed_node") == "pathology_analysis"
        assert loaded.get("skip_nodes") == ["biochemistry_analysis"]
        assert loaded.get("node_retry_count", {}).get("pathology_analysis") == 2
        assert loaded.get("node_errors", {}).get("pathology_analysis") == "some error"

    def test_semantic_gate_payload_preserves_fields(self, registry, v2_paths):
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)
        ws["integration_next"] = "gate"

        capsule = registry.freeze("pre_integration_gate", analysis_state, ws)
        assert capsule.gate == "integration_selection"

        from backend.services.snapshot_v2.object_store import ObjectStore
        store = ObjectStore(v2_paths)
        raw = store.get_bytes(capsule.payload_ref.sha256)
        import json
        payload = json.loads(raw.decode())

        assert payload.get("current_round") == 2
        assert payload.get("integration_next") == "gate"
        assert payload.get("phase") == "pre_integration_gate"

    def test_idempotent_post_report_restore_creates_no_workflow(self, registry, v2_paths):
        from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        capsule = registry.freeze("post_report", analysis_state, ws)
        assert capsule.kind == "terminal"

        from backend.services.snapshot_v2.object_store import ObjectStore
        store = ObjectStore(v2_paths)
        dispatcher = ContinuationDispatcher(store)

        staged = dispatcher.stage_checkpoint(
            capsule, ws, analysis_state,
            snapshot_id="snap-terminal",
            transaction_id="tx-terminal",
        )
        assert staged.kind == "terminal"
        assert staged.terminal_status == "completed"

        dispatcher.activate(analysis_state, staged)
        assert analysis_state.restore_resume_kind == "terminal"
        assert analysis_state.restore_resume_node is None
