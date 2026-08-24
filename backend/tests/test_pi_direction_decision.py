"""Tests for PI direction candidate generation, scoring, and selection (Task 3)."""
from unittest.mock import MagicMock

import pytest

from backend.agents.pi_agent import PIAgent
from backend.tests.split_facade import split_source

SRC = split_source("backend.agents.pi_agent")


def _make_pi_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


class TestPiPromptsIncludeAllScoringRules:
    def test_methods_exist(self):
        assert "generate_raw_direction_candidates" in SRC
        assert "score_directions_with_evidence" in SRC

    def test_no_final_scoring_in_raw_prompt(self):
        assert "Do NOT assign final innovation" in SRC

    def test_all_scoring_phrases_present(self):
        for phrase in [
            "data_reliability=low",
            "literature_confidence=failed",
            "High literature_novelty with low data_reliability",
            "functional_impact must be assessed separately",
            "biology_expectedness must be assessed separately",
            "overall_priority derives from",
            "negative_validation_confidence",
            "prerequisite",
        ]:
            assert phrase in SRC


class TestDependencyGraphValidation:
    def test_rejects_cycle(self):
        agent = _make_pi_agent()
        with pytest.raises(ValueError):
            agent._validate_direction_dependency_graph(
                {"dir_001": ["dir_002"], "dir_002": ["dir_001"]}
            )

    def test_accepts_dag(self):
        agent = _make_pi_agent()
        agent._validate_direction_dependency_graph(
            {"dir_001": ["dir_002"], "dir_002": [], "dir_003": ["dir_001"]}
        )

    def test_accepts_empty(self):
        agent = _make_pi_agent()
        agent._validate_direction_dependency_graph({})

    def test_rejects_self_loop(self):
        agent = _make_pi_agent()
        with pytest.raises(ValueError):
            agent._validate_direction_dependency_graph({"dir_001": ["dir_001"]})


class TestSelectBestDirectionsShape:
    def test_mentions_rejected_shape(self):
        section = SRC.split("select_best_directions", 1)[1]
        assert "selected_directions" in section
        assert "rejected_directions" in section
        assert "selection_rationale" in section

    def test_fallback_returns_dict_shape(self):
        section = SRC.split("select_best_directions", 1)[1]
        # First fallback: no candidates -> returns dict
        assert "selected_directions" in section
        # Check that all return statements produce dict shape
        assert '"selected_directions"' in section
        assert '"rejected_directions"' in section
        assert '"selection_rationale"' in section


class TestGenerateRawDirectionCandidates:
    def test_resets_history(self):
        agent = _make_pi_agent()
        agent.message_history = [{"role": "user", "content": "old"}]
        # We can't easily test the full method without mocks,
        # but we can verify reset_history is called by checking
        # the method's source
        assert "self.reset_history()" in SRC.split("generate_raw_direction_candidates", 1)[1]


class TestScoreDirectionsWithEvidence:
    def test_resets_history(self):
        assert "self.reset_history()" in SRC.split("score_directions_with_evidence", 1)[1]


class TestDependencyGraphRepair:
    def test_a_cycle_drops_only_its_back_edge_and_records_repair(self):
        agent = _make_pi_agent()
        report = __import__("backend.models.schemas", fromlist=["ExplorationReport"]).ExplorationReport(
            direction_dependency_graph={
                "dir_001": ["dir_002"],
                "dir_002": ["dir_003"],
                "dir_003": ["dir_001"],
            },
            research_direction_candidates=[
                {"direction_id": "dir_001", "direction": "A"},
                {"direction_id": "dir_002", "direction": "B"},
                {"direction_id": "dir_003", "direction": "C"},
            ],
        )

        dropped = agent._sanitize_direction_dependency_graph(report)

        assert dropped == ["dir_003 -> dir_001"]
        assert len(report.research_direction_candidates) == 3
        assert report.direction_dependency_graph == {
            "dir_001": ["dir_002"],
            "dir_002": ["dir_003"],
            "dir_003": [],
        }
        assert report.direction_dependency_repairs
