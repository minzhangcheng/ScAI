import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.models.schemas import AnalysisState, LiteratureReport, LiteratureMechanismSignal
from backend.workflow.agents import AgentNodesMixin


class _Harness(AgentNodesMixin):
    """Minimal host to call the persistence helper in isolation."""
    def __init__(self):
        self.session_id = "s"


def _signal():
    return LiteratureMechanismSignal(mechanism="m", supporting_pmids=["1"])


@pytest.mark.asyncio
async def test_persist_signals_attaches_to_all_current_round_reports():
    # A round can produce multiple LiteratureReports (multi-topic / PubMed gap-fill).
    # The round-level aggregate signals must be attached to EVERY current-round report,
    # because _build_literature_audit_context (Task 7) iterates ALL reports for signals.
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    rep1 = LiteratureReport(topic="t1", report="body1", references=[{"title": "x"}])
    rep2 = LiteratureReport(topic="t2", report="body2", references=[{"title": "y"}])
    state.literature_reports.extend([rep1, rep2])

    agent = MagicMock()
    agent.extract_mechanism_signals = AsyncMock(return_value=[_signal()])

    h = _Harness()
    await h._persist_literature_mechanism_signals(
        state, agent, report_ids=[rep1.id, rep2.id], observed_context="ctx",
    )
    for rep in (rep1, rep2):
        assert rep.mechanism_signals is not None
        assert rep.mechanism_signals[0].mechanism == "m"


@pytest.mark.asyncio
async def test_persist_skipped_when_no_reports():
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    agent = MagicMock()
    agent.extract_mechanism_signals = AsyncMock(return_value=[_signal()])
    h = _Harness()
    await h._persist_literature_mechanism_signals(state, agent, report_ids=[], observed_context="ctx")
    agent.extract_mechanism_signals.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_extraction_failure_leaves_none():
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    rep = LiteratureReport(topic="t", report="body")
    state.literature_reports.append(rep)
    agent = MagicMock()
    agent.extract_mechanism_signals = AsyncMock(return_value=None)
    h = _Harness()
    await h._persist_literature_mechanism_signals(state, agent, report_ids=[rep.id], observed_context="ctx")
    assert state.literature_reports[-1].mechanism_signals is None


def test_observed_context_prefers_current_workflow_state():
    # Gap-0 intent: literature extraction must align with the CURRENT round's data,
    # which lives in the WorkflowState dict at literature-research time, not yet in
    # analysis_state.round_results.
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    workflow_state = {
        "analysis_state": state,
        "biochemistry_result": {"findings": ["CUR_BIOCHEM"]},
        "bioinformatics_result": {"findings": ["CUR_BIOINFO"]},
        "pathology_result": {"findings": []},
    }
    h = _Harness()
    ctx = h._build_observed_context_for_literature(workflow_state)
    assert "CUR_BIOCHEM" in ctx
    assert "CUR_BIOINFO" in ctx


def test_observed_context_falls_back_to_prior_round_and_indicators():
    from backend.models.schemas import BiochemistryIndicator, RoundResult
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    state.round_results.append(RoundResult(round_number=1, biochemistry_findings=["PRIOR_BIOCHEM"]))
    state.biochemistry_indicators.append(
        BiochemistryIndicator(indicator="Iron", value=2.2, unit="x", reference_range="1.0")
    )
    workflow_state = {"analysis_state": state}  # no current-round result dicts
    h = _Harness()
    ctx = h._build_observed_context_for_literature(workflow_state)
    assert "PRIOR_BIOCHEM" in ctx
    assert "Iron" in ctx
