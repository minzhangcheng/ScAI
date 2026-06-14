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
from backend.models.schemas import AnalysisState, ScoreDefinition, LiteratureReport


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
