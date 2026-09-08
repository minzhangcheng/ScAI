"""Context isolation and plan-revision resume tests for BioinformaticsAgent."""

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.bioinformatics_agent import (
    BioinformaticsAgent,
    _resume_budget_after_plan_revision,
    build_approved_plan_context,
    inject_revised_plan_context,
)
from backend.agents import bioinformatics_agent as bioinformatics_facade
from backend.agents.bioinformatics import executor_loop
from backend.constants import PLAN_REVISION_NEEDED_TAG
from backend.models.schemas import (
    AgentType,
    AnalysisPlan,
    AnalysisState,
    PlanStep,
)
from backend.tests.split_facade import setattr_facade
from backend.utils.llm_client import LLMToolResponse, ToolCall


def _plan(*, plan_id: str, revision_of: str | None = None) -> AnalysisPlan:
    return AnalysisPlan(
        id=plan_id,
        phase="exploration",
        stage="full",
        overview="Run the bounded revised workflow",
        steps=[
            PlanStep(id=1, title="QC", method="scanpy.pp.filter_cells", status="completed"),
            PlanStep(id=2, title="DE", method="scanpy.tl.rank_genes_groups"),
        ],
        key_decision_points=["Review DE evidence"],
        status="approved",
        revision_of=revision_of,
        revision_reason="The observed data require a different remaining method.",
    )


def _bare_agent(monkeypatch, tmp_path) -> BioinformaticsAgent:
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent.session_id = "test-session"
    agent.message_history = []
    agent.code_history = []
    agent.analysis_context = {}
    agent.last_result = None
    agent._recent_sup_feedback = set()
    agent._pause_event = None
    agent.literature_callback = None
    agent.pi_callback = None
    agent.annotate_clusters_callback = None
    agent.message_callback = AsyncMock()
    agent.figure_review_callback = None
    agent.supervisor_callback = None
    agent.technique_agent = None
    agent.species_validation_callback = None
    agent.marker_db_intervention_callback = None
    agent._provider = MagicMock()
    agent._provider.get_config_revision.return_value = 0
    agent._last_config_revision = 0
    agent.agent_type = AgentType.BIOINFORMATICS

    monkeypatch.setattr(BioinformaticsAgent, "model", property(lambda self: "test-model"))
    monkeypatch.setattr(BioinformaticsAgent, "system_prompt", property(lambda self: "system"))
    monkeypatch.setattr(
        BioinformaticsAgent,
        "config",
        property(lambda self: MagicMock(temperature=0.1)),
    )
    monkeypatch.setattr(
        BioinformaticsAgent,
        "_static_system_prompt",
        property(lambda self: "static system"),
    )
    monkeypatch.setattr(
        BioinformaticsAgent,
        "_build_dynamic_context",
        lambda self, state: "bounded dynamic context",
    )
    monkeypatch.setattr(BioinformaticsAgent, "_set_llm_context", lambda self: None)
    monkeypatch.setattr(BioinformaticsAgent, "_notify", AsyncMock())

    agent.sandbox = MagicMock()
    agent.sandbox.state_dir = tmp_path / "states"
    agent.sandbox.working_dir = tmp_path
    agent.sandbox.km = MagicMock()
    agent.sandbox.start = AsyncMock()
    agent.sandbox.save_session = AsyncMock()
    agent.sandbox.get_code_history.return_value = []
    agent.sandbox.sync_runtime_persistence_state.return_value = {
        "status": "current",
        "degraded_recovery": False,
        "requires_recompute": False,
    }
    return agent


@pytest.mark.asyncio
async def test_plan_generation_does_not_use_or_modify_default_history():
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    # Mirrors _agent() above: __init__ is bypassed, and _call_planning_llm
    # labels its retry wrapper from llm_client/model, both of which resolve
    # through BaseAgent._provider.
    agent._provider = MagicMock()
    agent.agent_type = AgentType.BIOINFORMATICS
    agent.message_history = [
        {"role": "user", "content": "legacy" * 100_000},
        {"role": "assistant", "content": "legacy response"},
    ]
    original_history = list(agent.message_history)
    agent.chat_without_history = AsyncMock(return_value=(
        '{"overview":"bounded","steps":[{"id":1,"title":"QC",'
        '"method":"scanpy.pp.filter_cells","depends_on":[]}],'
        '"key_decision_points":[],"expected_runtime":"1 step"}'
    ))

    state = MagicMock()
    state.build_experimental_design_summary.return_value = "small data summary"
    plan = await agent.generate_analysis_plan(
        state=state,
        pi_tasks=["Run QC"],
        overview="QC only",
        phase="exploration",
    )

    assert len(plan.steps) == 1
    agent.chat_without_history.assert_awaited_once()
    assert agent.message_history == original_history


@pytest.mark.asyncio
async def test_planning_dialogue_resets_legacy_history_and_passes_bounded_revision_context():
    from backend.workflow import SingleCellAnalysisWorkflow

    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = "test-session"
    workflow._send_message = AsyncMock()
    workflow._build_planning_context = MagicMock(
        return_value={"bioinformatics_findings": ["result-" + "x" * 20_000]}
    )
    workflow._require_core_agent = AsyncMock(return_value=None)

    old_plan = _plan(plan_id="old-plan")
    revised_plan = _plan(plan_id="new-plan", revision_of="old-plan")
    bioinfo = MagicMock()
    bioinfo.reset_history = MagicMock()
    bioinfo.revise_plan_incrementally = AsyncMock(return_value=revised_plan)
    workflow.bioinformatics_agent = bioinfo
    workflow.pi_agent = MagicMock()
    workflow.pi_agent.review_bioinfo_plan = AsyncMock(return_value={
        "decision": "approved",
        "feedback": "approved",
        "modifications": [],
    })

    analysis_state = AnalysisState(id="test-session", user_id="test-user")
    analysis_state.current_plan = old_plan
    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {"bioinformatics_tasks": ["Continue"], "overview": "Revise"},
        "planning_stage": "full",
        "plan_revision_needed": True,
        "plan_revision_context": {
            "question": "Change the method?",
            "context": "trigger-" + "y" * 20_000,
        },
    }

    await workflow._planning_dialogue(state)

    bioinfo.reset_history.assert_called_once_with()
    feedback = bioinfo.revise_plan_incrementally.await_args.kwargs["pi_feedback"]
    assert "Change the method?" in feedback
    assert "Bounded summary of results available at the pause" in feedback
    assert len(feedback) < 9_000
    assert "y" * 10_000 not in feedback
    assert "x" * 10_000 not in feedback


def test_revised_plan_is_injected_once_for_changed_plan_id():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "[Analysis Context]\ndata"},
    ]
    plan = _plan(plan_id="new-plan", revision_of="old-plan")

    assert inject_revised_plan_context(messages, plan, "old-plan") is True
    assert len(messages) == 3
    assert "APPROVED PLAN REVISION" in messages[-1]["content"]
    assert "Preserve completed work" in messages[-1]["content"]
    assert "Step 2: DE" in messages[-1]["content"]

    assert inject_revised_plan_context(messages, plan, "new-plan") is False
    assert len(messages) == 3


def test_fresh_approved_plan_context_is_not_a_resume_instruction():
    text = build_approved_plan_context(_plan(plan_id="initial"))
    assert "APPROVED ANALYSIS PLAN" in text
    assert "Resume from the existing checkpoint" not in text


def test_new_pi_plan_resets_exhausted_executor_to_a_fresh_attempt():
    checkpoint = {
        "step_number": 200,
        "total_steps": 200,
        "active_plan_id": "old-plan",
        "budget_exhausted": True,
    }

    assert _resume_budget_after_plan_revision(
        checkpoint,
        checkpoint_valid=True,
        step_count=200,
        current_max_steps=200,
        per_attempt_max_steps=200,
        active_plan_id="new-plan",
    ) == (400, 200, True)
    assert _resume_budget_after_plan_revision(
        checkpoint,
        checkpoint_valid=True,
        step_count=200,
        current_max_steps=200,
        per_attempt_max_steps=200,
        active_plan_id="old-plan",
    ) == (200, 0, False)


def test_new_pi_plan_resets_a_non_exhausted_executor_attempt():
    checkpoint = {
        "active_plan_id": "old-plan",
        "budget_exhausted": False,
    }
    assert _resume_budget_after_plan_revision(
        checkpoint,
        checkpoint_valid=True,
        step_count=80,
        current_max_steps=200,
        per_attempt_max_steps=200,
        active_plan_id="new-plan",
    ) == (280, 80, True)


def test_each_new_pi_plan_gets_its_own_fresh_executor_attempt():
    checkpoint = {
        "step_number": 230,
        "total_steps": 230,
        "attempt_start_step": 200,
        "active_plan_id": "plan-v2",
        "budget_exhausted": True,
    }

    assert _resume_budget_after_plan_revision(
        checkpoint,
        checkpoint_valid=True,
        step_count=230,
        current_max_steps=230,
        per_attempt_max_steps=200,
        active_plan_id="plan-v3",
    ) == (430, 230, True)
    assert _resume_budget_after_plan_revision(
        checkpoint,
        checkpoint_valid=True,
        step_count=230,
        current_max_steps=230,
        per_attempt_max_steps=200,
        active_plan_id="plan-v2",
    ) == (230, 200, False)


@pytest.mark.asyncio
async def test_resume_preserves_executor_messages_and_persists_new_plan(
    monkeypatch,
    tmp_path,
):
    agent = _bare_agent(monkeypatch, tmp_path)
    original_messages = [
        {"role": "system", "content": "static system"},
        {"role": "user", "content": "[Analysis Context]\nold context"},
        {"role": "user", "content": "Analysis instruction: original"},
    ]
    checkpoint = {
        "step_name": "step_1_plan_revision",
        "session_id": "test-session",
        "round_number": 0,
        "planning_stage": "full",
        "step_number": 1,
        "total_steps": 1,
        "active_plan_id": "old-plan",
        "messages": list(original_messages),
        "findings": ["existing finding"],
        "figures": [],
        "code_history": [],
        "analysis_log": [],
        "analysis_context": {"kept": True},
        "completed": False,
    }
    saves = []

    class _Checkpoint:
        def __init__(self, *args, **kwargs):
            pass

        async def load(self):
            return checkpoint

        async def save(self, data):
            saves.append(data)

        def clear(self):
            pass

        def should_checkpoint(self, step):
            return False

    setattr_facade(
        monkeypatch, "backend.agents.bioinformatics_agent", "CheckpointManager", _Checkpoint
    )
    state = AnalysisState(id="test-session", user_id="test-user")
    state.current_plan = _plan(plan_id="new-plan", revision_of="old-plan")

    result = await agent.run_autonomous_analysis(
        state,
        initial_instruction="This is intentionally discarded on resume.",
        max_steps=1,
        resume_from_checkpoint=True,
    )

    assert result["findings"] == ["existing finding"]
    assert saves, "the updated resume context must be persisted before execution"
    saved = saves[-1]
    assert saved["active_plan_id"] == "new-plan"
    assert saved["attempt_start_step"] == 1
    assert saved["total_steps"] == 2
    assert saved["messages"][:3] == original_messages
    assert "APPROVED PLAN REVISION" in saved["messages"][-1]["content"]
    assert saved["analysis_context"] == {"kept": True}


@pytest.mark.asyncio
async def test_checkpoint_restores_current_step_and_replaced_transcript(
    monkeypatch,
    tmp_path,
):
    agent = _bare_agent(monkeypatch, tmp_path)
    response = LLMToolResponse(
        content="continue",
        tool_calls=[],
        model="test-model",
        usage={},
        raw_response=None,
        raw_assistant_message=None,
    )
    fake_client = MagicMock()
    fake_client.base_url = "https://example.invalid/v1"
    fake_client.chat_with_tools = AsyncMock(return_value=response)
    monkeypatch.setattr(
        BioinformaticsAgent,
        "llm_client",
        property(lambda self: fake_client),
    )
    saved_payload = None

    class _Checkpoint:
        def __init__(self, *args, **kwargs):
            pass

        async def load(self):
            return deepcopy(saved_payload) if saved_payload is not None else None

        async def save(self, data):
            nonlocal saved_payload
            saved_payload = deepcopy(data)

        def clear(self):
            pass

        def should_checkpoint(self, step):
            return step == 2

    setattr_facade(
        monkeypatch, "backend.agents.bioinformatics_agent", "CheckpointManager", _Checkpoint
    )
    executor_loop.bind_facade(bioinformatics_facade)
    state = AnalysisState(id="test-session", user_id="test-user")
    context = await executor_loop.prepare_loop_context(
        agent,
        state,
        initial_instruction="Run analysis.",
        max_steps=5,
        resume_from_checkpoint=False,
    )
    context.max_text_responses = 3
    original_messages = [{"role": "system", "content": "system"}]
    original_messages.extend(
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"message-{index}",
        }
        for index in range(84)
    )
    context.messages = original_messages

    await executor_loop.start_turn(context)
    trimmed_messages = deepcopy(context.messages)
    assert len(trimmed_messages) < len(original_messages)
    assert trimmed_messages != original_messages

    await executor_loop.start_turn(context)

    assert saved_payload is not None
    assert saved_payload["step_number"] == context.step_count == 2
    assert saved_payload["messages"] == trimmed_messages

    restored_agent = _bare_agent(monkeypatch, tmp_path)
    restored_context = await executor_loop.prepare_loop_context(
        restored_agent,
        state,
        initial_instruction="Discarded on resume.",
        max_steps=5,
        resume_from_checkpoint=True,
    )
    assert restored_context.step_count == saved_payload["step_number"]
    assert restored_context.messages == saved_payload["messages"]


@pytest.mark.asyncio
async def test_plan_revision_forces_immediate_executor_and_sandbox_checkpoint(
    monkeypatch,
    tmp_path,
):
    agent = _bare_agent(monkeypatch, tmp_path)
    agent.pi_callback = AsyncMock(
        return_value=f"Change the remaining method. {PLAN_REVISION_NEEDED_TAG}"
    )
    tool_call = ToolCall(
        id="consult-1",
        name="consult_pi",
        arguments={
            "question": "Should the remaining method change?",
            "context": "Observed deviation",
            "options": [],
        },
    )
    response = LLMToolResponse(
        content="",
        tool_calls=[tool_call],
        model="test-model",
        usage={},
        raw_response=None,
        raw_assistant_message=None,
    )
    fake_client = MagicMock()
    fake_client.base_url = "https://example.invalid/v1"
    fake_client.chat_with_tools = AsyncMock(return_value=response)
    monkeypatch.setattr(
        BioinformaticsAgent,
        "llm_client",
        property(lambda self: fake_client),
    )
    saves = []

    class _Checkpoint:
        def __init__(self, *args, **kwargs):
            pass

        async def load(self):
            return None

        async def save(self, data):
            saves.append(data)

        def clear(self):
            pass

        def should_checkpoint(self, step):
            return False

    setattr_facade(
        monkeypatch, "backend.agents.bioinformatics_agent", "CheckpointManager", _Checkpoint
    )
    state = AnalysisState(id="test-session", user_id="test-user")
    state.current_plan = _plan(plan_id="old-plan")

    result = await agent.run_autonomous_analysis(
        state,
        initial_instruction="Run analysis.",
        max_steps=5,
        resume_from_checkpoint=False,
    )

    assert result["status"] == "paused_for_revision"
    revision_saves = [
        saved for saved in saves
        if saved["step_name"].endswith("_plan_revision")
    ]
    assert len(revision_saves) == 1
    assert revision_saves[0]["active_plan_id"] == "old-plan"
    assert any(msg.get("tool_call_id") == "consult-1" for msg in revision_saves[0]["messages"])
    agent.sandbox.save_session.assert_awaited_once()
