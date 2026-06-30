import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.models.schemas import (
    AnalysisState, LiteratureReport, LiteratureMechanismSignal, BiochemistryIndicator,
    AnalysisPhase, RoundResult, CrossModalSynthesis, CrossModalMechanismCandidate,
)
from backend.workflow.synthesis import SynthesisMixin
from backend.workflow import SingleCellAnalysisWorkflow


class _Host(SynthesisMixin):
    def __init__(self):
        self.session_id = "s"


def test_literature_audit_context_capped_and_includes_signals():
    host = _Host()
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    # References live on the persisted LiteratureReport (NOT on literature_result).
    rep = LiteratureReport(
        topic="t", report="X" * 100000,
        references=[{"title": "REF_MARKER", "pmid": "1"}],
    )
    rep.mechanism_signals = [LiteratureMechanismSignal(mechanism="SIGNAL_MARKER", supporting_pmids=["1"])]
    state.literature_reports.append(rep)
    # literature_result carries only report bodies under "reports" — no "references" key.
    workflow_state = {"analysis_state": state, "literature_result": {"reports": ["BODY_MARKER body text"]}}

    ctx = host._build_literature_audit_context(workflow_state)
    assert "SIGNAL_MARKER" in ctx       # (1) mechanism signals
    assert "REF_MARKER" in ctx          # (2) references from persisted report
    assert "BODY_MARKER" in ctx         # (3) report body excerpt
    # Hard cap ~32K chars
    assert len(ctx) <= 32000


def test_biochemistry_raw_context_includes_indicators():
    host = _Host()
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    # BiochemistryIndicator.value is a float (verified) — pass a number, not a string.
    state.biochemistry_indicators.append(
        BiochemistryIndicator(indicator="Iron", value=2.2, unit="x", reference_range="1.0")
    )
    workflow_state = {"analysis_state": state, "biochemistry_result": {"findings": ["Iron up 2.2x"]}}
    ctx = host._build_biochemistry_raw_context(workflow_state)
    assert "Iron" in ctx


def _candidate():
    return CrossModalMechanismCandidate(
        theme="iron injury", mechanism_statement="m", confidence="medium",
        relation_to_current_direction="uncovered", report_recommendation="secondary",
    )


def _make_wf_for_synthesis(state):
    """Build a bare SingleCellAnalysisWorkflow with the FULL set of collaborators
    that _pi_synthesis touches on the innovation path, all mocked.
    """
    import asyncio
    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "s"
    wf.config = MagicMock(max_rounds=5)
    wf.state_callback = None
    wf.message_callback = None
    wf._send_message = AsyncMock()
    wf._require_core_agent = AsyncMock(return_value=None)
    wf._create_auto_snapshot = AsyncMock()
    wf._build_and_save_detailed_segment = AsyncMock()  # innovation segment path
    wf._continue_event = asyncio.Event()
    wf._continue_event.set()
    wf._set_status = MagicMock()
    wf._send_state_sync = AsyncMock()
    return wf


@pytest.mark.asyncio
async def test_pi_synthesis_stores_cross_modal_for_innovation():
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    state.analysis_phase = AnalysisPhase.INNOVATION
    state.current_round_direction = "copper collapse"
    state.current_round = 2

    wf = _make_wf_for_synthesis(state)
    pi = MagicMock()
    pi.synthesize_round = AsyncMock(return_value=RoundResult(round_number=2, pi_summary="s"))
    pi.build_cross_modal_synthesis = AsyncMock(
        return_value=CrossModalSynthesis(matrix_markdown="| m |", candidates=[_candidate()])
    )
    pi.identify_data_needs = AsyncMock(return_value=[])
    wf.pi_agent = pi

    graph_state = {
        "analysis_state": state,
        "biochemistry_result": {"findings": ["Fe up"]},
        "pathology_result": {"findings": []},
        "bioinformatics_result": {"findings": ["Hmox1 up"]},
        "literature_result": {"findings": [], "reports": []},
    }

    out = await wf._pi_synthesis(graph_state)
    last = out["analysis_state"].round_results[-1]
    assert last.cross_modal_synthesis is not None
    assert last.cross_modal_synthesis.candidates[0].theme == "iron injury"
    # Audit matrix was passed into synthesize_round
    _, kwargs = pi.synthesize_round.call_args
    assert kwargs.get("cross_modal_context") == "| m |"
    # round_summary WS message carries simplified candidates, never matrix_markdown
    summary_call = next(c for c in wf._send_message.call_args_list if c.args[0] == "round_summary")
    payload = summary_call.args[1]
    # O4: matrix_markdown must NOT appear anywhere in the WS payload — top level...
    assert "matrix_markdown" not in payload
    # ...nor inside any candidate
    assert "cross_modal_candidates" in payload
    assert payload["cross_modal_candidates"][0]["theme"] == "iron injury"
    assert "matrix_markdown" not in payload["cross_modal_candidates"][0]


@pytest.mark.asyncio
async def test_pi_synthesis_skips_audit_for_exploration():
    # O3: exploration rounds must NOT run the cross-modal audit.
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    state.analysis_phase = AnalysisPhase.EXPLORATION
    state.current_round = 1

    wf = _make_wf_for_synthesis(state)
    # Make the exploration branch a no-op so the test stays focused on the audit gate.
    wf._exploration_round_succeeded = MagicMock(return_value=False)

    pi = MagicMock()
    pi.synthesize_round = AsyncMock(return_value=RoundResult(round_number=1, pi_summary="s"))
    pi.build_cross_modal_synthesis = AsyncMock()
    pi.identify_data_needs = AsyncMock(return_value=[])
    wf.pi_agent = pi

    graph_state = {
        "analysis_state": state,
        "biochemistry_result": {"findings": []},
        "pathology_result": {"findings": []},
        "bioinformatics_result": {"findings": []},
        "literature_result": {"findings": [], "reports": []},
    }

    out = await wf._pi_synthesis(graph_state)
    pi.build_cross_modal_synthesis.assert_not_awaited()
    assert out["analysis_state"].round_results[-1].cross_modal_synthesis is None
