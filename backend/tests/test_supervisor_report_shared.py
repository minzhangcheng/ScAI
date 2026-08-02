"""Supervisor's live report path must reuse the shared report modules
(score_section / literature_files / sections.resolve_literature_model).

Regression: 0.26.3/0.26.4 updated services/report/* but the workflow runs
supervisor_agent.generate_detailed_report when a supervisor exists, so none of
those changes reached the actual report.html.
"""
import pytest

from backend.agents.supervisor_agent import SupervisorAgent
from backend.config._config import SystemConfig
from backend.utils.settings_provider import AgentSettingsProvider
from backend.models.schemas import (
    AnalysisState,
    DirectionSelectionRecord,
    LiteratureReport,
    ScoreDefinition,
)


def _make_supervisor(state):
    provider = AgentSettingsProvider(state=state, config=SystemConfig(), global_client=None)
    sup = SupervisorAgent.__new__(SupervisorAgent)
    sup._provider = provider
    sup.session_id = None
    return sup


def test_score_definitions_html_uses_shared_enhanced_format():
    report = {
        "score_definitions": [
            {
                "score_name": "ISG_score",
                "method": "sc.tl.score_genes",
                "description": "interferon score",
                "parameters": {"ctrl_size": 50},
                "gene_list": ["A", "B", "C"],
                "gene_list_source": "curated",
            }
        ]
    }
    html = SupervisorAgent._build_score_definitions_html(report)
    # New shared format: plain-language computation column + CSV download link.
    assert "Computation Method" in html
    assert "Download CSV" in html
    assert "score_definitions.csv" in html
    # Mean-expression aggregation language from friendly_method.
    assert "Mean expression" in html


def test_appendix_resolves_literature_deep_research_label():
    state = AnalysisState(id="s1", name="t", initial_instruction="x", user_id="u1")
    state.literature_actual_model = "Google Deep Research"
    sup = _make_supervisor(state)
    appendix = sup._build_appendix(state, stats={})
    assert appendix["agent_models"]["literature"] == "Google Deep Research"


def test_appendix_includes_literature_report_link_when_reports_exist():
    state = AnalysisState(id="s2", name="t", initial_instruction="x", user_id="u1")
    state.literature_reports = [
        LiteratureReport(topic="T", report="body", references=[])
    ]
    sup = _make_supervisor(state)
    appendix = sup._build_appendix(state, stats={})
    # Canonical schema: literature_report is a top-level appendix key (matches
    # generator.py / regenerator._build_appendix_from_report), not nested.
    assert appendix.get("literature_report") == "literature_report.html"

    html = sup._build_appendix_html_section({"appendix": appendix})
    assert "literature_report.html" in html
    assert "Literature Research Report" in html


def test_appendix_no_literature_link_without_reports():
    state = AnalysisState(id="s3", name="t", initial_instruction="x", user_id="u1")
    sup = _make_supervisor(state)
    appendix = sup._build_appendix(state, stats={})
    assert not appendix.get("literature_report")


def test_live_appendix_includes_complete_direction_selection_history():
    state = AnalysisState(id="s4", name="t", initial_instruction="x", user_id="u1")
    state.user_chosen_directions = [{"direction": "Last direction", "notes": "last"}]
    state.direction_selection_records = [
        DirectionSelectionRecord(
            round_number=1,
            phase="exploration",
            event_type="pi_decide",
            pi_selected=[
                {"direction": "First direction", "notes": "first rationale"},
                {"direction": "Second direction", "notes": "second rationale"},
            ],
        ),
        DirectionSelectionRecord(
            round_number=4,
            phase="integration",
            integration_cycle=1,
            event_type="pi_decide",
            pi_selected=[{"direction": "Third direction", "notes": "third rationale"}],
        ),
    ]
    sup = _make_supervisor(state)

    appendix = sup._build_appendix(state, stats={})
    rows = appendix["user_interactions"]["direction_selection"]
    html = sup._build_appendix_html_section({"appendix": appendix})

    assert len(rows) == 3
    assert [row["direction"] for row in rows] == [
        "First direction", "Second direction", "Third direction",
    ]
    assert "Selection Event" in html
    assert "PI-selected" in html
    assert "Integration Cycle 1" in html


def test_supervisor_html_renders_partial_and_quality_banners(tmp_path):
    state = AnalysisState(id="s5", name="Partial", initial_instruction="x", user_id="u1")
    sup = _make_supervisor(state)
    report = {
        "title": "Partial report",
        "status": "partial",
        "partial_reason": "approved_partial",
        "partial_reasons": ["approved_partial"],
        "quality_disclosures": {
            "schema_version": 1,
            "status": "qualified",
            "rounds": [],
            "post_execution_reviews": [],
            "annotation": {
                "confirmation": {"status": "not_applicable"},
                "validation": {"status": "not_run"},
                "cell_counts": {"known_clusters": 0, "unknown_clusters": 0},
            },
            "blocking_issues": [],
            "cautions": ["Qualification retained."],
        },
        "appendix": {},
        "score_definitions": [],
        "_analysis_state": state,
    }

    path = sup._generate_html_report(
        report,
        sections=[],
        abstract="Abstract",
        conclusions="Conclusions",
        report_figures=[],
        figure_map={},
        stats={},
        language="en",
        output_path=tmp_path,
        workspace_path=None,
    )
    html = path.read_text(encoding="utf-8")

    assert "Partial report" in html
    assert "explicitly approved" in html
    assert "Analysis completed with quality qualifications" in html
    assert "Qualification retained" in html
