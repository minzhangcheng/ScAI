"""PI Chat corrections share a budget and preserve every executed attempt."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import AnalysisState
from backend.routes.chat_parts.delegation import _build_execution_context, _build_synthesis_prompt
from backend.routes.chat_parts.reviews import ChatReviewCallbacks, run_reviewed_analysis


def _execution(number, used):
    return {
        "status": "completed", "steps_executed": used, "findings": [f"finding-{number}"],
        "attempt_execution_ids": [f"e-{number}"],
        "code_history": [{"id": f"e-{number}", "code": f"code_{number}()", "error": None}],
        "analysis_log": [{"id": f"log-{number}", "step_number": number,
                          "step_type": "execute_code", "execution_id": f"e-{number}"}],
    }


@pytest.fixture
def chat_run():
    state = AnalysisState(id="chat-review", user_id="owner")
    pi = SimpleNamespace(review_chat_execution=AsyncMock())
    provider = SimpleNamespace(get_model=lambda name: "model")
    callbacks = ChatReviewCallbacks(state, provider, pi, AsyncMock(), AsyncMock(), None)
    agent = SimpleNamespace(run_autonomous_analysis=AsyncMock())
    saved = []

    async def persist():
        saved.append(state.model_dump_json())

    async def run():
        return await run_reviewed_analysis(
            state=state, instruction="Compare observed programs", bio_agent=agent,
            reviews=callbacks, persist=persist, collect_figures=lambda scopes: [],
            execution_context=_build_execution_context,
        )

    return state, pi, agent, saved, run


@pytest.mark.asyncio
@pytest.mark.parametrize("decisions,used,expected_status", [
    (["revise", "accept"], [47, 5], "accept"),
    (["revise", "revise", "revise"], [10, 10, 10], "budget_exhausted"),
    (["revise"], [60], "budget_exhausted"),
    (["limited"], [12], "limited"),
    (["unavailable"], [12], "unavailable"),
])
async def test_chat_corrections_preserve_evidence_and_share_budget(
    chat_run, decisions, used, expected_status,
):
    state, pi, agent, saved, run = chat_run
    pi.review_chat_execution.side_effect = [{
        "decision": decision, "revision_instruction": "Correct calculation",
        "review_unavailable": decision == "unavailable",
    } for decision in decisions]
    agent.run_autonomous_analysis.side_effect = [_execution(i, n) for i, n in enumerate(used)]
    result = await run()
    calls = agent.run_autonomous_analysis.await_args_list
    assert len(calls) == len(used)
    assert calls[0].kwargs["max_steps"] == 60
    for index, call in enumerate(calls[1:], 1):
        assert call.kwargs["max_steps"] == min(20, 60 - sum(used[:index]))
        assert "Correct calculation" in call.args[1]
        assert all(f"code_{prior}()" in call.args[1] for prior in range(index))
    assert len({call.kwargs["run_scope_id"] for call in calls}) == len(used)
    assert all(call.kwargs["resume_from_checkpoint"] is False for call in calls)
    assert [row["id"] for row in result["code_history"]] == [f"e-{i}" for i in range(len(used))]
    assert len(result["analysis_log"]) == len(used)
    assert result["findings"] == [f"finding-{len(used) - 1}"]
    assert result["status"] == "completed"
    assert result["chat_review"]["status"] == expected_status
    restored = AnalysisState.model_validate_json(saved[-1])
    assert len(restored.chat_analysis_reviews[0]["cycles"]) == len(used)
    assert restored.chat_analysis_reviews[0]["status"] == expected_status
    assert [receipt.id for receipt in restored.code_history] == [
        f"e-{i}" for i in range(len(used))
    ]
    assert [step.id for step in restored.analysis_log] == [
        f"log-{i}" for i in range(len(used))
    ]
    assert all(receipt.executor_run_kind == "chat" and receipt.plan_id is None
               and receipt.chat_review_id == restored.chat_analysis_reviews[0]["id"]
               for receipt in restored.code_history)
    assert all(step.executor_run_kind == "chat" and step.plan_id is None
               and step.chat_review_id == restored.chat_analysis_reviews[0]["id"]
               for step in restored.analysis_log)
    assert restored.round_results == []
    prompt = _build_synthesis_prompt("question", {}, {"bioinformatics": result})
    assert "PI Chat review" in prompt and expected_status in prompt


@pytest.mark.asyncio
async def test_chat_corrections_reuse_pi_source_and_preparation(chat_run):
    state, pi, agent, saved, _ = chat_run
    pi.review_chat_execution.side_effect = [
        {"decision": "revise", "rationale": "Check the plot", "revision_instruction": "Fix plot"},
        {"decision": "accept", "rationale": "Resolved"},
    ]
    first = _execution(0, 2)
    first["resolved_adata_source_context"] = {"preparation_id": "prep-chat"}
    agent.run_autonomous_analysis.side_effect = [first, _execution(1, 2)]

    async def persist():
        saved.append(state.model_dump_json())

    await run_reviewed_analysis(
        state=state, instruction="Plot the selected source", bio_agent=agent,
        reviews=SimpleNamespace(
            supervision_records=lambda: [], observations=[],
            _provider=SimpleNamespace(get_model=lambda _: "model"),
            _send_message=AsyncMock(), pi_agent=pi,
        ),
        persist=persist, collect_figures=lambda _: [],
        execution_context=_build_execution_context,
        source_decision={"selection": {"mode": "checkpoint_copy", "checkpoint_id": "cp"},
                         "active_adata_var": None, "rationale": "Use the selected stage"},
    )
    first_source = agent.run_autonomous_analysis.await_args_list[0].kwargs["adata_source"]
    second_source = agent.run_autonomous_analysis.await_args_list[1].kwargs["adata_source"]
    assert first_source["owner_id"] == second_source["owner_id"]
    assert first_source["preparation_id"] is None
    assert second_source["preparation_id"] == "prep-chat"


@pytest.mark.asyncio
async def test_cancellation_records_interruption_without_rerun(chat_run):
    state, pi, agent, saved, run = chat_run
    agent.run_autonomous_analysis.return_value = _execution(0, 4)
    pi.review_chat_execution.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await run()
    agent.run_autonomous_analysis.assert_awaited_once()
    assert state.chat_analysis_reviews[0]["status"] == "cancelled"
    assert len(saved) >= 2
    restored = AnalysisState.model_validate_json(saved[-1])
    assert restored.chat_analysis_reviews[0]["status"] == "cancelled"
    assert restored.chat_analysis_reviews[0]["cycles"][0]["execution_ids"] == ["e-0"]
    assert [receipt.id for receipt in restored.code_history] == ["e-0"]


@pytest.mark.asyncio
async def test_step_limit_is_durable_even_if_pi_accepts_partial_evidence(chat_run):
    state, pi, agent, saved, run = chat_run
    execution = _execution(0, 60)
    execution["status"] = "max_steps_reached"
    agent.run_autonomous_analysis.return_value = execution
    pi.review_chat_execution.return_value = {
        "decision": "accept", "rationale": "The existing result is useful",
    }

    result = await run()

    assert result["chat_review"]["status"] == "accept"
    assert result["chat_review"]["execution_incomplete"] is True
    assert "Continue in a new Chat request" in result["chat_review"]["unfinished_work"]
    restored = AnalysisState.model_validate_json(saved[-1])
    assert restored.chat_analysis_reviews[0]["execution_incomplete"] is True
    assert restored.code_history[0].executor_run_kind == "chat"
