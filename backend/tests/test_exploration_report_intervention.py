"""Exploration report failures retain their step until the user resolves them."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from backend.constants import LlmFailureAction
from backend.models.schemas import AnalysisState, AnalysisPhase, ExplorationReport
from backend.utils.llm_client import UpstreamExhaustedError
from backend.workflow import synthesis
from backend.workflow.helpers import WorkflowTerminatedError
from backend.workflow.synthesis import SynthesisMixin
from backend.workflow.synthesis_parts import phases


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_step", ["raw", "literature", "score"])
@pytest.mark.parametrize("action", list(LlmFailureAction))
async def test_report_intervention_holds_boundary_and_retries_only_failed_step(
    monkeypatch, failed_step, action,
):
    state = AnalysisState(user_id="u", name="s", initial_instruction="x", round_wait_seconds=0)
    state.current_round = 1
    graph = {"analysis_state": state, "bioinformatics_result": {"status": "completed"}}
    rr = SimpleNamespace(round_number=1, pi_summary="summary", exploration_report=None, report_chapter=None)
    ctx = {"state": graph, "analysis_state": state, "round_result": rr,
           "bioinformatics_result": graph["bioinformatics_result"]}
    raw = [{"direction_id": "dir_001", "direction": "A"}]
    evidence = [{"direction_id": "dir_001", "status": "completed"}]
    report = ExplorationReport(research_direction_candidates=raw)
    calls = {"raw": AsyncMock(return_value=raw),
             "literature": AsyncMock(return_value=evidence),
             "score": AsyncMock(return_value=report)}
    calls[failed_step].side_effect = [
        UpstreamExhaustedError("600s deadline", agent_name="pi", model="test"),
        {"raw": raw, "literature": evidence, "score": report}[failed_step],
    ]
    wf = SynthesisMixin()
    wf.session_id = "s"
    wf.pi_agent = SimpleNamespace(
        generate_raw_direction_candidates=calls["raw"],
        score_directions_with_evidence=calls["score"],
        plan_direction_literature_search=AsyncMock(),
        review_direction_literature_search=AsyncMock(),
        _sanitize_direction_dependency_graph=Mock(),
    )
    wf.literature_agent = SimpleNamespace(evaluate_direction_literature_search=calls["literature"])
    wf._send_message = AsyncMock()
    wf._build_and_save_detailed_segment = AsyncMock()
    entered, release = asyncio.Event(), asyncio.Event()

    async def intervention(error, node, analysis_state, **kwargs):
        assert node == "pi_synthesis"
        assert analysis_state is state
        assert kwargs == {}
        entered.set()
        await release.wait()
        return action

    wf._wait_for_llm_intervention = intervention
    phases.bind_facade(synthesis)
    monkeypatch.setattr(phases, "synthesize_agent_results", AsyncMock(return_value=ctx))
    persist, finalize = AsyncMock(), AsyncMock(return_value=graph)
    monkeypatch.setattr(phases, "persist_round_result", persist)
    monkeypatch.setattr(phases, "finalize_synthesis", finalize)
    task = asyncio.create_task(phases.run_pi_synthesis(wf, graph))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not task.done()
        persist.assert_not_awaited()
        finalize.assert_not_awaited()
        assert state.analysis_phase == AnalysisPhase.EXPLORATION
        assert state.exploration_report is None
        assert state.current_round == 1
        assert not any(c.args[0] == "exploration_report" for c in wf._send_message.call_args_list)
        release.set()
        if action == LlmFailureAction.TERMINATE:
            with pytest.raises(WorkflowTerminatedError):
                await task
            persist.assert_not_awaited()
            finalize.assert_not_awaited()
        else:
            await task
            persist.assert_awaited_once()
            finalize.assert_awaited_once()
        if action == LlmFailureAction.RETRY:
            assert state.exploration_report is report
            assert state.analysis_phase == AnalysisPhase.INNOVATION
            assert calls[failed_step].await_count == 2
            assert all(c.await_count == 1 for k, c in calls.items() if k != failed_step)
            assert sum(c.args[0] == "exploration_report" for c in wf._send_message.call_args_list) == 1
            produced = next(
                c.args[1] for c in wf._send_message.call_args_list
                if c.args[0] == "exploration_report"
            )
            from backend.services.round_snapshots.restore import (
                _strip_unavailable_report_urls,
            )
            assert produced["html_url"] == (
                f"/api/session/{state.id}/stage-review-report/{state.stage_review_reports[-1].id}/html"
            )
            assert _strip_unavailable_report_urls(produced)["html_url"] is None
        else:
            assert state.exploration_report is None
            assert state.analysis_phase == AnalysisPhase.EXPLORATION
            assert calls[failed_step].await_count == 1
        assert state.current_round == 1
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
