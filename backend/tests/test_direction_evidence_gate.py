from types import SimpleNamespace
import pytest

from backend.models.schemas import AnalysisState, ExplorationReport
from backend.tests.split_facade import source_for_path

SYNTHESIS = source_for_path("backend/workflow/synthesis.py")
PLANNING = source_for_path("backend/workflow/planning.py")


def test_all_exploration_report_paths_use_evidence_gate():
    assert SYNTHESIS.count("generate_exploration_report(") == 0
    assert SYNTHESIS.count("generate_raw_direction_candidates") >= 1
    assert SYNTHESIS.count("evaluate_direction_novelty_candidates") >= 1
    assert SYNTHESIS.count("score_directions_with_evidence") >= 2


def test_planning_consumes_direction_selection_decision_shape():
    assert "selected_directions" in PLANNING
    assert "direction_selection_summary" in PLANNING


def test_planning_decision_normalizer_accepts_new_and_legacy_shapes():
    from backend.workflow.planning import _selected_directions_from_decision

    selected = [{"direction": "A", "notes": "n"}]
    assert _selected_directions_from_decision({"selected_directions": selected}) == selected
    assert _selected_directions_from_decision(selected) == selected
    assert _selected_directions_from_decision({"selected_directions": "bad"}) == []
    assert _selected_directions_from_decision([]) == []
    assert _selected_directions_from_decision({}) == []


@pytest.mark.asyncio
async def test_evidence_gate_uses_skipped_evidence_when_literature_was_skipped():
    from backend.workflow.synthesis import SynthesisMixin

    state = AnalysisState(user_id="u1", name="s", initial_instruction="x")
    round_result = SimpleNamespace()
    raw = [{"direction_id": "dir_001", "direction": "IFN network"}]
    captured = {}

    class PI:
        async def generate_raw_direction_candidates(self, analysis_state, rr):
            return raw

        async def score_directions_with_evidence(self, analysis_state, rr, raw_candidates, direction_evidence):
            captured["evidence"] = direction_evidence
            return ExplorationReport(
                raw_direction_candidates=raw_candidates,
                direction_literature_evidence=direction_evidence,
                research_direction_candidates=[{"direction_id": "dir_001", "direction": "IFN network"}],
            )

        def _sanitize_direction_dependency_graph(self, report):
            return []

    class Literature:
        async def evaluate_direction_novelty_candidates(self, *args, **kwargs):
            raise AssertionError("Literature Agent must not be called after literature_research was skipped")

    wf = SynthesisMixin()
    wf.session_id = "s"
    wf.pi_agent = PI()
    wf.literature_agent = Literature()

    report = await wf._generate_evidence_gated_exploration_report(
        {"analysis_state": state, "literature_result": {"status": "skipped"}},
        state,
        round_result,
    )

    assert report.direction_literature_evidence[0]["status"] == "skipped"
    assert report.direction_literature_evidence[0]["literature_novelty"] == "unknown"
    assert captured["evidence"][0]["confidence"] == "failed"


@pytest.mark.asyncio
async def test_evidence_gate_sanitizes_dependency_graph_before_returning_report():
    from backend.workflow.synthesis import SynthesisMixin

    state = AnalysisState(user_id="u1", name="s", initial_instruction="x")
    round_result = SimpleNamespace()
    validated = {}

    class PI:
        async def generate_raw_direction_candidates(self, analysis_state, rr):
            return [{"direction_id": "dir_001", "direction": "A"}]

        async def score_directions_with_evidence(self, analysis_state, rr, raw_candidates, direction_evidence):
            return ExplorationReport(
                direction_dependency_graph={"dir_001": ["dir_002"]},
                research_direction_candidates=[{"direction_id": "dir_001", "direction": "A"}],
            )

        def _sanitize_direction_dependency_graph(self, report):
            validated["graph"] = report.direction_dependency_graph
            return []

    class Literature:
        async def evaluate_direction_novelty_candidates(self, *args, **kwargs):
            return [{"direction_id": "dir_001", "direction": "A", "status": "completed"}]

    wf = SynthesisMixin()
    wf.session_id = "s"
    wf.pi_agent = PI()
    wf.literature_agent = Literature()

    await wf._generate_evidence_gated_exploration_report({"analysis_state": state}, state, round_result)

    assert validated["graph"] == {"dir_001": ["dir_002"]}
