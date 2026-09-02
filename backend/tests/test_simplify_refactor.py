"""Tests for the simplify/cleanup refactor: shared model methods and helpers."""
import inspect
import json
import uuid as _uuid
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from pydantic import BaseModel, Field
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schemas import (
    AgentType,
    AnalysisPlan,
    CrossModalMechanismCandidate,
    CrossModalEvidenceItem,
    DataDrivenMechanismProgram,
    PlanStep,
    tolerant_from_llm_dict,
)


class TestCrossModalMechanismCandidateToSummaryDict:
    def test_returns_five_fields(self):
        c = CrossModalMechanismCandidate(
            theme="Ferroptosis",
            mechanism_statement="Iron-dependent cell death via lipid peroxidation",
            supporting_evidence=[
                CrossModalEvidenceItem(source="bioinformatics", evidence="upreg", strength="strong")
            ],
            evidence_consistency="high",
            confidence="high",
            novelty_relative_to_current_direction="medium",
            relation_to_current_direction="central",
            missing_evidence=["in vivo validation"],
            report_recommendation="headline",
        )
        d = c.to_summary_dict()
        assert set(d.keys()) == {
            "theme", "mechanism_statement", "confidence",
            "relation_to_current_direction", "report_recommendation",
        }

    def test_defaults_are_populated(self):
        c = CrossModalMechanismCandidate(
            theme="X", mechanism_statement="Y",
        )
        d = c.to_summary_dict()
        assert d["confidence"] == "low"
        assert d["relation_to_current_direction"] == "uncovered"
        assert d["report_recommendation"] == "future_direction"

    def test_values_match_model(self):
        c = CrossModalMechanismCandidate(
            theme="Apoptosis",
            mechanism_statement="Caspase cascade",
            confidence="medium",
            relation_to_current_direction="complementary",
            report_recommendation="secondary",
        )
        d = c.to_summary_dict()
        assert d["theme"] == "Apoptosis"
        assert d["mechanism_statement"] == "Caspase cascade"
        assert d["confidence"] == "medium"
        assert d["relation_to_current_direction"] == "complementary"
        assert d["report_recommendation"] == "secondary"

    def test_exact_equivalence_with_manual_dict(self):
        """Golden test: to_summary_dict() must produce the identical dict
        that the original inline code built in analysis_log.py."""
        c = CrossModalMechanismCandidate(
            theme="T", mechanism_statement="M",
            confidence="high",
            relation_to_current_direction="central",
            report_recommendation="headline",
        )
        expected = {
            "theme": c.theme,
            "mechanism_statement": c.mechanism_statement,
            "confidence": c.confidence,
            "relation_to_current_direction": c.relation_to_current_direction,
            "report_recommendation": c.report_recommendation,
        }
        assert c.to_summary_dict() == expected


class TestDataDrivenMechanismProgramToPromptText:
    def _make_program(self, **overrides):
        defaults = dict(
            label="Ferroptosis signature",
            cell_types=["Macrophages", "T cells"],
            evidence_strength="moderate",
            specificity="medium",
            bulk_support="supported",
            triggering_observations=["Iron accumulation", "Lipid peroxidation markers elevated", "Third obs"],
            supporting_genes=["GPX4", "SLC7A11", "FTH1", "ACSL4"],
            supporting_pathways=["Ferroptosis pathway"],
            alternative_explanations=["Oxidative stress"],
            missing_evidence=["In vivo validation", "Lipidomics data", "Third missing"],
        )
        defaults.update(overrides)
        return DataDrivenMechanismProgram(**defaults)

    def test_compact_golden(self):
        p = self._make_program()
        text = p.to_prompt_text(style="compact", max_observations=2, max_genes=8,
                                max_pathways=8, max_alternatives=2, max_missing=2)
        assert "Ferroptosis signature" in text
        assert "moderate" in text
        assert "medium" in text
        assert "supported" in text
        assert "Iron accumulation" in text
        assert "Lipid peroxidation markers elevated" in text
        assert "Third obs" not in text
        assert "GPX4" in text
        assert "Oxidative stress" in text
        assert "In vivo validation" in text
        assert "Third missing" not in text

    def test_direction_golden(self):
        p = self._make_program()
        text = p.to_prompt_text(style="direction", max_observations=2, max_genes=12, max_missing=2)
        assert "[candidate mechanism]" in text
        assert "Ferroptosis signature" in text
        assert "moderate" in text
        assert "supported" in text
        assert "Macrophages" in text
        assert "T cells" in text
        assert "Iron accumulation" in text
        assert "GPX4" in text
        assert "Ferroptosis pathway" in text
        assert "In vivo validation" in text

    def test_direction_no_cell_types_graceful(self):
        p = self._make_program(cell_types=[])
        text = p.to_prompt_text(style="direction")
        assert "Ferroptosis signature" in text

    def test_direction_exploratory_tag(self):
        p = self._make_program(evidence_strength="weak", triggering_observations=["A"])
        text = p.to_prompt_text(style="direction")
        assert "[exploratory signal]" in text

    def test_detailed_golden(self):
        p = self._make_program()
        text = p.to_prompt_text(style="detailed", max_observations=3, max_genes=15,
                                max_pathways=8, max_alternatives=3, max_missing=3)
        lines = text.split("\n")
        assert lines[0] == "- Program: Ferroptosis signature"
        assert any("Cell types:" in l for l in lines)
        assert any("Observations:" in l for l in lines)
        assert any("Genes:" in l for l in lines)
        assert any("Pathways:" in l for l in lines)
        assert any("Bulk support:" in l for l in lines)
        assert any("Alternatives:" in l for l in lines)
        assert any("Missing evidence:" in l for l in lines)

    def test_empty_fields_graceful(self):
        p = DataDrivenMechanismProgram(label="Empty program")
        for style in ("compact", "direction", "detailed"):
            text = p.to_prompt_text(style=style)
            assert "Empty program" in text

    def test_truncation_obs(self):
        p = self._make_program()
        text = p.to_prompt_text(style="compact", max_observations=1, max_genes=8)
        assert "Iron accumulation" in text
        assert "Lipid peroxidation markers elevated" not in text

    def test_truncation_genes(self):
        p = self._make_program()
        text = p.to_prompt_text(style="compact", max_observations=2, max_genes=2)
        assert "GPX4" in text
        assert "ACSL4" not in text


class TestBioinformaticsParseRetry:
    """Test the extracted _retry_on_parse_failure method (async)."""

    @pytest.fixture
    def agent(self):
        with patch("backend.agents.bioinformatics_agent.BioinformaticsAgent.__init__", lambda self: None):
            a = BioinformaticsAgent.__new__(BioinformaticsAgent)
            # __init__ is bypassed, so the settings provider BaseAgent injects
            # has to be supplied here: _call_planning_llm reads llm_client and
            # model to label the replay-safe retry wrapper, and both resolve
            # through _provider.
            a._provider = MagicMock()
            a.agent_type = AgentType.BIOINFORMATICS
            a._safe_parse_json = MagicMock()
            # The retry deliberately goes through chat_without_history: the retry prompt
            # already carries the full original request and must not inherit unrelated
            # planning turns from BaseAgent history.
            a.chat_without_history = AsyncMock()
            # Under test here is the retry control flow, not plan validation. Passing
            # the payload straight through keeps the minimal fixtures below meaningful;
            # real _prepare_plan_data would reject them on missing PlanStep fields.
            a._prepare_plan_data = lambda data, **_kwargs: data
            a.session_id = "test"
            return a

    @pytest.mark.asyncio
    async def test_no_retry_when_parse_succeeds(self, agent):
        plan_data = {"steps": [{"id": 1, "title": "Step 1"}]}
        result = await agent._retry_on_parse_failure("prompt", plan_data)
        assert result == plan_data
        agent.chat_without_history.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_succeeds(self, agent):
        bad_data = {"parse_error": True}
        good_data = {"steps": [{"id": 1}]}
        agent.chat_without_history.return_value = "good json"
        agent._safe_parse_json.return_value = good_data
        result = await agent._retry_on_parse_failure("prompt", bad_data)
        assert "parse_error" not in result
        agent.chat_without_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_fails_returns_original(self, agent):
        bad_data = {"parse_error": True, "steps": []}
        agent.chat_without_history.return_value = "still bad"
        agent._safe_parse_json.return_value = {"parse_error": True}
        result = await agent._retry_on_parse_failure("prompt", bad_data, warn_on_double_failure=False)
        assert result == bad_data


class TestModelCopyIdSemantics:
    """Verify that model_copy + update preserves the 'new id on revision' semantic."""

    def test_retained_plan_gets_new_id(self):
        """model_copy(deep=True) alone would keep old id, making id == revision_of
        (self-referential). We must generate a new id."""
        original = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="test",
            steps=[PlanStep(id=1, title="S1", method="m", input="i", expected_output="o", rationale="r", depends_on=[])],
            status="approved",
            approved_at=datetime(2025, 1, 1),
        )
        original_id = original.id

        retained = original.model_copy(deep=True, update={
            "status": "draft",
            "revision_of": original.id,
            "revision_reason": "Retained previous plan",
            "id": str(_uuid.uuid4()),
            "approved_at": None,
        })

        assert retained.revision_of == original_id
        assert retained.id != original_id  # new id, not self-referential
        assert retained.status == "draft"
        assert retained.approved_at is None  # reset to match draft status
        assert len(retained.steps) == 1
        # steps are deep-copied — mutating one doesn't affect the other
        retained.steps[0].title = "modified"
        assert original.steps[0].title == "S1"


class TestNoOutputTokenCap:
    """The reasoning-model output floor was removed: a cap on a thinking model
    is a cap on thinking, and running out returns an empty answer channel."""

    def test_sentinel_is_none(self):
        from backend.constants import NO_OUTPUT_TOKEN_CAP
        assert NO_OUTPUT_TOKEN_CAP is None

    def test_reasoning_floor_is_gone(self):
        import backend.constants as constants
        assert not hasattr(constants, "REASONING_MODEL_MIN_TOKENS")


class TestSelectFallbackPlanFromHistory:
    """Test PlanningMixin._select_fallback_plan_from_history — the extracted
    helper that replaces the inline fallback loop in _planning_dialogue.

    This tests the REAL production code path, not a logic copy.
    """

    def _make_mixin(self):
        from backend.workflow.planning import PlanningMixin
        mixin = PlanningMixin()
        mixin.session_id = "test_session"
        return mixin

    def test_model_validate_failure_skips_entry(self):
        """When model_validate fails on a dialogue entry, the entry is skipped
        and a later valid entry is used as fallback instead.

        Uses naturally-invalid data (string id where int is expected) rather
        than mocking so the test exercises the real model_validate path."""
        from backend.models.schemas import AnalysisPlan, PlanStep

        mixin = self._make_mixin()

        good_plan = AnalysisPlan(
            phase="exploration", stage="full", overview="good plan",
            steps=[PlanStep(id=1, title="S1", method="m", input="i",
                            expected_output="o", rationale="r", depends_on=[])],
        )
        # Invalid: id is a string, not an int
        bad_plan_dict = {
            "phase": "exploration", "stage": "full", "overview": "bad",
            "steps": [{"id": "not_a_number", "title": "S"}],
            "key_decision_points": [], "expected_runtime": "~1 steps",
        }
        good_plan_dict = good_plan.model_dump(mode="json")

        # bad entry encountered first in reversed order (round 2), should be
        # skipped due to model_validate failure, then good entry from round 1 used
        dialogue_history = [
            {"pi_decision": "approve", "plan": good_plan_dict, "round": 1},
            {"pi_decision": "approve", "plan": bad_plan_dict, "round": 2},
        ]

        result = mixin._select_fallback_plan_from_history(dialogue_history)

        assert result is not None
        assert result.overview == "good plan"
        assert len(result.steps) == 1

    def test_all_entries_invalid_returns_none(self):
        """If all dialogue entries fail model_validate, returns None."""
        from pydantic import ValidationError
        from unittest.mock import patch
        from backend.models.schemas import AnalysisPlan

        mixin = self._make_mixin()

        bad_plan_dict = {
            "phase": "exploration", "stage": "full", "overview": "bad",
            "steps": [{"id": "not_a_number", "title": "S"}],
        }
        dialogue_history = [
            {"pi_decision": "approve", "plan": bad_plan_dict, "round": 1},
            {"pi_decision": "approve", "plan": bad_plan_dict, "round": 2},
        ]

        with patch.object(AnalysisPlan, "model_validate", side_effect=ValidationError.from_exception_data("AnalysisPlan", [])):
            result = mixin._select_fallback_plan_from_history(dialogue_history)

        assert result is None

    def test_reject_entries_skipped(self):
        """Entries with pi_decision == 'reject' are skipped."""
        from backend.models.schemas import AnalysisPlan, PlanStep

        mixin = self._make_mixin()

        good_plan = AnalysisPlan(
            phase="exploration", stage="full", overview="approved plan",
            steps=[PlanStep(id=1, title="S1", method="m", input="i",
                            expected_output="o", rationale="r", depends_on=[])],
        )
        dialogue_history = [
            {"pi_decision": "approve", "plan": good_plan.model_dump(mode="json"), "round": 2},
            {"pi_decision": "reject", "plan": good_plan.model_dump(mode="json"), "round": 1},
        ]

        result = mixin._select_fallback_plan_from_history(dialogue_history)
        assert result is not None
        assert result.overview == "approved plan"

    def test_no_steps_entry_skipped(self):
        """Entries with empty steps are skipped."""
        from backend.models.schemas import AnalysisPlan

        mixin = self._make_mixin()

        empty_plan_dict = {
            "phase": "exploration", "stage": "full", "overview": "empty",
            "steps": [],
        }
        dialogue_history = [
            {"pi_decision": "approve", "plan": empty_plan_dict, "round": 1},
        ]

        result = mixin._select_fallback_plan_from_history(dialogue_history)
        assert result is None


def test_plan_dialogue_html_facade_delegates_to_extracted_helper():
    """Exercise the production facade instead of a gate stub."""
    from backend.workflow.planning import PlanningMixin

    html = PlanningMixin._build_plan_dialogue_html([
        {
            "round": 1,
            "pi_decision": "approved",
            "pi_feedback": "Proceed",
            "plan": {
                "steps": [
                    {
                        "id": 1,
                        "title": "Quality control",
                        "method": "standard QC",
                        "rationale": "Establish data quality",
                    }
                ]
            },
        }
    ])

    assert "Round 1" in html
    assert "Quality control" in html


def test_split_prompt_examples_keep_nested_indentation():
    from backend.agents.pathology_parts import image_analysis
    from backend.services.report import planning as report_planning

    pathology_source = inspect.getsource(image_analysis.analyze_image)
    report_source = inspect.getsource(report_planning)

    assert '\n    "overall_structure": "Overall structure description"' in pathology_source
    assert '\n    "findings": ["finding 1", "finding 2", ...]' in pathology_source
    assert '\n        {{\n            "id": "unique_id"' in report_source
    assert '\n        - Introduces the purpose of this analysis step' in report_source


class _SampleModel(BaseModel):
    name: str = ""
    count: int = 0
    tags: list = Field(default_factory=list)


class TestTolerantFromLlmDict:
    def test_valid_data_passes_through(self):
        data = {"name": "test", "count": 5, "tags": ["a"]}
        result = tolerant_from_llm_dict(_SampleModel, data)
        assert result.name == "test"
        assert result.count == 5
        assert result.tags == ["a"]

    def test_bad_field_dropped(self):
        data = {"name": "test", "count": "not_a_number", "tags": ["a"]}
        result = tolerant_from_llm_dict(_SampleModel, data)
        assert result.name == "test"
        assert result.count == 0  # default
        assert result.tags == ["a"]

    def test_unknown_field_ignored(self):
        data = {"name": "test", "bogus": True}
        result = tolerant_from_llm_dict(_SampleModel, data)
        assert result.name == "test"

    def test_all_bad_returns_defaults(self):
        data = {"name": 123, "count": "bad"}
        result = tolerant_from_llm_dict(_SampleModel, data)
        assert result.name == ""
        assert result.count == 0

    def test_exploration_report_delegates(self):
        """ExplorationReport.from_llm_dict now delegates to tolerant_from_llm_dict."""
        from backend.models.schemas import ExplorationReport
        data = {"cluster_overview": "test", "bogus_field": True}
        result = ExplorationReport.from_llm_dict(data)
        assert result.cluster_overview == "test"


class TestLiteratureMechanismSignalBound:
    """Verify that _build_literature_audit_context only reads the last 8
    literature reports for mechanism signals (behavior change from unbounded).
    References were already bounded to [-4:]; mechanism signals now to [-8:].
    """

    def test_oldest_signals_excluded_beyond_8(self):
        """With >8 literature reports, the oldest report's mechanism signals
        should NOT appear in the output of _build_literature_audit_context."""
        from backend.workflow.synthesis import SynthesisMixin
        from backend.models.schemas import AnalysisState, LiteratureReport, LiteratureMechanismSignal

        # Build 10 literature reports; the oldest (index 0) has a unique marker signal.
        unique_mechanism = "UNIQUE_OLD_MECHANISM_MARKER"
        reports = []
        # Oldest report with unique signal
        reports.append(LiteratureReport(
            topic="oldest topic",
            report="oldest report body",
            mechanism_signals=[LiteratureMechanismSignal(
                mechanism=unique_mechanism,
                supporting_pmids=["123"],
                evidence_type="experimental",
                relevance_to_current_study="high",
            )],
        ))
        # 9 more recent reports with generic signals
        for i in range(1, 10):
            reports.append(LiteratureReport(
                topic=f"topic {i}",
                report=f"report body {i}",
                mechanism_signals=[LiteratureMechanismSignal(
                    mechanism=f"recent mechanism {i}",
                    supporting_pmids=[],
                    evidence_type="correlational",
                    relevance_to_current_study="medium",
                )],
            ))

        analysis_state = AnalysisState(name="test", user_id="test_user")
        analysis_state.literature_reports = reports

        state = {
            "analysis_state": analysis_state,
            "literature_result": {},
        }

        mixin = SynthesisMixin()
        mixin.session_id = "test"
        context = mixin._build_literature_audit_context(state)

        # The oldest report's unique signal should NOT appear (only last 8 are read)
        assert unique_mechanism not in context
        # A recent report's signal (from the last 8) should appear
        assert "recent mechanism 9" in context

    def test_all_signals_included_when_8_or_fewer(self):
        """With ≤8 literature reports, all signals should appear."""
        from backend.workflow.synthesis import SynthesisMixin
        from backend.models.schemas import AnalysisState, LiteratureReport, LiteratureMechanismSignal

        reports = []
        for i in range(8):
            reports.append(LiteratureReport(
                topic=f"topic {i}",
                report=f"report body {i}",
                mechanism_signals=[LiteratureMechanismSignal(
                    mechanism=f"mechanism {i}",
                    supporting_pmids=[],
                    evidence_type="experimental",
                    relevance_to_current_study="high",
                )],
            ))

        analysis_state = AnalysisState(name="test", user_id="test_user")
        analysis_state.literature_reports = reports

        state = {
            "analysis_state": analysis_state,
            "literature_result": {},
        }

        mixin = SynthesisMixin()
        mixin.session_id = "test"
        context = mixin._build_literature_audit_context(state)

        # All 8 signals should appear
        for i in range(8):
            assert f"mechanism {i}" in context
