"""PI Chat corrections share a budget and preserve every executed attempt."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.pi.figures import FiguresMixin
from backend.models.schemas import AnalysisState, PublicationFigure
from backend.routes.chat_parts import delegation, reviews
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
@pytest.mark.parametrize("figure_reviewer", ["pi", "bioinformatics", "supervisor"])
async def test_intermediate_pi_consultations_retain_user_scope_during_corrections(
    monkeypatch, tmp_path, figure_reviewer,
):
    state = AnalysisState(id="chat-scope", user_id="owner", initial_instruction="Investigate disease mechanisms")
    instruction = (
        "Update the requested visualization.\n\n"
        "Original user request (scope for final review):\n"
        "Produce one marker dotplot using existing groups; do not recluster."
    )
    class FigureReviewer(FiguresMixin, SimpleNamespace):
        pass

    def reviewer(**kwargs):
        return FigureReviewer(
            session_id=state.id, has_vision=True, model="model", system_prompt="Review evidence",
            llm_client=SimpleNamespace(base_url=""), _set_llm_context=lambda: None,
            _safe_parse_json=lambda text, default=None: json.loads(text), **kwargs,
        )

    pi = reviewer(
        set_session=lambda _: None,
        chat_without_history=AsyncMock(return_value="Explain uncertainty in the figure legend."),
        review_chat_execution=AsyncMock(side_effect=[
            {"decision": "revise", "revision_instruction": "Improve labels"},
            {"decision": "accept"},
        ]),
    )
    sandbox = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), km=None)
    provider = SimpleNamespace(
        get_model=lambda _: "model", cleanup=AsyncMock(),
        get_vision_setting=lambda name: name == figure_reviewer,
    )
    client = SimpleNamespace(close=AsyncMock())
    send = AsyncMock()
    consultations = []
    answers = []
    executions = []
    figure = PublicationFigure(
        name="marker_dotplot", figure_type="dotplot", description="Markers across existing groups",
        png_path="dotplot.png", pdf_path="dotplot.pdf", png_base64="image",
    )
    vision = AsyncMock(return_value=SimpleNamespace(content=json.dumps({
        "approved": False, "quality_score": 2, "feedback": "Marker evidence is insufficient",
        "must_fix": ["Disclose the unsupported label"], "requires_reanalysis": True,
    })))
    monkeypatch.setattr("backend.agents.pi.figures.replay_safe_vision_call", vision)
    monkeypatch.setattr("backend.workflow.callbacks.replay_safe_vision_call", vision)

    async def consult(callback, question, context):
        consultations.append((question, context))
        answers.append(await callback(question, context))

    def make_supervisor(**kwargs):
        async def evaluate(step, figures):
            await consult(kwargs["pi_callback"], "Review this checkpoint", "Broader atlas concerns")
            return reviews.SupervisorEvalResult()

        return reviewer(
            set_sandbox=lambda _: None, evaluate_step=evaluate, supervision_log=[],
            _compress_base64_figure=lambda image: image,
        )

    def make_bio_agent(**kwargs):
        async def run(state, task, **run_kwargs):
            cycle = len(executions)
            executions.append(task)
            await consult(kwargs["pi_callback"], "Resolve marker ambiguity", f"Execution cycle {cycle}")
            await kwargs["supervisor_callback"](SimpleNamespace(id=f"step-{cycle}"), [])
            figure.revision_count = cycle
            verdict = await kwargs["figure_review_callback"](figure)
            assert verdict["approved"] is False
            assert verdict["feedback"] == "Marker evidence is insufficient"
            if figure_reviewer != "supervisor":
                assert verdict["requires_reanalysis"] is True
            return _execution(cycle, 2)

        return reviewer(set_session=lambda _: None, run_autonomous_analysis=run)

    monkeypatch.setattr(delegation, "resolve_config_for_user", AsyncMock(return_value=SimpleNamespace(api=None)))
    monkeypatch.setattr(delegation, "AgentSettingsProvider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(delegation, "get_session_states", lambda _: tmp_path / "states")
    monkeypatch.setattr(delegation, "get_session_workspace", lambda _: tmp_path)
    monkeypatch.setattr(delegation, "_persist_chat_terminal", AsyncMock(return_value=True))
    monkeypatch.setattr(delegation, "_collect_session_figures", lambda *args, **kwargs: [])
    monkeypatch.setattr("backend.utils.sandbox.LocalJupyterSandbox", lambda **kwargs: sandbox)
    monkeypatch.setattr("backend.utils.agent_llm.get_agent_llm_client", lambda *args: client)
    monkeypatch.setattr("backend.agents.literature_agent.LiteratureAgent", MagicMock())
    monkeypatch.setattr("backend.agents.pi_agent.PIAgent", lambda **kwargs: pi)
    monkeypatch.setattr("backend.agents.bioinformatics_agent.BioinformaticsAgent", make_bio_agent)
    monkeypatch.setattr(reviews, "SupervisorAgent", make_supervisor)
    monkeypatch.setattr("backend.routes.chat_parts.source_selection.select_chat_source", AsyncMock(return_value=None))

    result, _ = await delegation._execute_chat_agent_inner(
        state.id, state, "bioinformatics", instruction, client, send, [],
    )

    assert len(executions) == 2 and "Improve labels" in executions[1]
    assert len(consultations) == pi.chat_without_history.await_count == 4
    for call, (question, context) in zip(pi.chat_without_history.await_args_list, consultations):
        prompt = call.args[0]
        assert instruction in prompt
        assert question in prompt and context in prompt
        assert "requested scope takes precedence" in prompt
        assert "separate follow-up" in prompt
    assert answers == [pi.chat_without_history.return_value] * 4
    responses = [call.args[0].content for call in send.await_args_list
                 if call.args[0].type == "pi_response"]
    assert responses == answers
    assert result["chat_review"]["status"] == "accept"
    assert state.id not in delegation.chat_sandboxes
    assert vision.await_count == 2
    for call in vision.await_args_list:
        prompt = call.kwargs["text_prompt"]
        assert instruction in prompt
        assert "separate follow-up" in prompt
        assert "do not justify approving an invalid figure" in prompt
        assert call.kwargs["images"] == ["image"]
    assert state.initial_instruction == "Investigate disease mechanisms"


@pytest.mark.asyncio
async def test_workflow_figure_review_without_chat_scope_keeps_existing_call():
    state = AnalysisState(id="workflow-review", user_id="owner")
    figure = PublicationFigure(
        name="plot", figure_type="dotplot", png_path="plot.png", pdf_path="plot.pdf",
        png_base64="image", description="Workflow figure",
    )
    pi = SimpleNamespace(review_figure=AsyncMock(return_value={"approved": True}))
    workflow = SimpleNamespace(
        session_id=state.id, _current_analysis_state=state, pi_agent=pi,
        supervisor_agent=None, ensure_agent_ready=lambda _: False,
        _vision_settings={"pi": True}, _send_message=AsyncMock(),
    )
    result = await reviews.CallbacksMixin._figure_review_callback(workflow, figure)
    assert result == {"approved": True}
    pi.review_figure.assert_awaited_once_with(state, figure, "image")


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
