import pytest
from pydantic import TypeAdapter
from typing import Any, Dict


def _make_minimal_analysis_state():
    from backend.models.schemas import AnalysisState
    return AnalysisState(id="test-state", name="test", user_id="u1", initial_instruction="analyze")


def _make_minimal_workflow_state(analysis_state):
    return {
        "analysis_state": analysis_state,
        "current_round": 1,
        "pi_instructions": {"key": "val"},
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
        "planning_result": {"plan": "test"},
        "planning_context": {"context": "yes"},
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
    }


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


class TestContinuationRegistryPhases:
    def test_graph_node_phases_return_graph_node_capsule(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.workspace import WorkspaceCapture
        from backend.services.snapshot_v2.continuation import ContinuationRegistry

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        for phase in ["pre_execution_stage1", "pre_execution_stage2", "pre_execution_full"]:
            capsule = registry.freeze(phase, analysis_state, ws)
            assert capsule.kind == "graph_node"
            assert capsule.next_node == "bioinformatics_analysis"
            assert len(capsule.workflow_state_ref.sha256) == 64
            assert capsule.workflow_state_ref.size > 0

    def test_semantic_gate_phases_return_semantic_gate_capsule(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.continuation import ContinuationRegistry

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        for phase in ["pre_annotation", "post_annotation", "pre_direction", "post_direction",
                       "pre_round_wait", "pre_integration_gate", "pre_major_revision"]:
            capsule = registry.freeze(phase, analysis_state, ws)
            assert capsule.kind == "semantic_gate"
            assert capsule.gate
            assert len(capsule.payload_ref.sha256) == 64
        # pre_major_revision is the plan_revision gate
        rev = registry.freeze("pre_major_revision", analysis_state, ws)
        assert rev.gate == "plan_revision"

    def test_terminal_phase_returns_terminal_capsule(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.continuation import ContinuationRegistry

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        capsule = registry.freeze("post_report", analysis_state, ws)
        assert capsule.kind == "terminal"
        assert capsule.status == "completed"

    def test_idle_phase_returns_idle_capsule(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.continuation import ContinuationRegistry

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        capsule = registry.freeze("manual", analysis_state, ws)
        assert capsule.kind == "idle"

    def test_dynamic_post_chat_phase(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.continuation import ContinuationRegistry

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        capsule = registry.freeze("post_chat", analysis_state, ws)
        assert capsule.kind == "idle"

    def test_dynamic_pre_restore_phase(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.continuation import ContinuationRegistry

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        capsule = registry.freeze("pre_restore", analysis_state, ws)
        assert capsule.kind == "idle"

    def test_unknown_phase_rejected(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.errors import SnapshotCaptureError

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        with pytest.raises(SnapshotCaptureError):
            registry.freeze("nonexistent_phase", analysis_state, ws)

    def test_capsule_validates_as_continuation_capsule(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.continuation import ContinuationRegistry
        from backend.services.snapshot_v2.models import ContinuationCapsule

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        capsule = registry.freeze("pre_execution_stage1", analysis_state, ws)
        adapter = TypeAdapter(ContinuationCapsule)
        roundtrip = adapter.validate_python(capsule.model_dump(mode="json"))
        assert roundtrip.kind == "graph_node"
        assert roundtrip.next_node == capsule.next_node

    def test_graph_node_workflow_state_is_persisted(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.continuation import ContinuationRegistry

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        capsule = registry.freeze("pre_execution_stage1", analysis_state, ws)
        obj_path = v2_paths.objects / capsule.workflow_state_ref.sha256
        assert obj_path.exists()
        raw = obj_path.read_bytes()
        import json
        data = json.loads(raw)
        assert data["current_round"] == 1
        assert data["plan_approved"] is True
        assert data["planning_stage"] == "stage1"

    def test_semantic_gate_payload_is_persisted(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.continuation import ContinuationRegistry

        store = ObjectStore(v2_paths)
        registry = ContinuationRegistry(store)
        analysis_state = _make_minimal_analysis_state()
        ws = _make_minimal_workflow_state(analysis_state)

        capsule = registry.freeze("pre_annotation", analysis_state, ws)
        obj_path = v2_paths.objects / capsule.payload_ref.sha256
        assert obj_path.exists()
        raw = obj_path.read_bytes()
        import json
        data = json.loads(raw)
        assert "phase" in data
        assert "gate" in data
        assert data["phase"] == "pre_annotation"


class RecordingObjectStore:
    def __init__(self):
        self.put_calls = 0

    def put_bytes(self, payload, media_type):
        self.put_calls += 1
        raise AssertionError("put_bytes must not run for invalid planning_stage")


def test_freeze_dynamic_validates_planning_stage_before_writing():
    from backend.services.snapshot_v2.continuation import ContinuationRegistry
    from backend.services.snapshot_v2.errors import SnapshotCaptureError

    store = RecordingObjectStore()
    registry = ContinuationRegistry(store)
    live_safe_point = {
        "next_node": "bioinformatics_analysis",
        "planning_stage": "invalid",
    }

    with pytest.raises(SnapshotCaptureError, match="invalid planning_stage"):
        registry.freeze_dynamic(live_safe_point)

    assert store.put_calls == 0
