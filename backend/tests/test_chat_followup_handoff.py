"""Chat handoff validation and trusted-context delegation contracts."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from backend.config import SystemConfig
from backend.models.schemas import AnalysisState, AnalysisStep, CodeExecution
from backend.routes.chat_parts import delegation as chat_delegation
from backend.routes.chat_parts import delegation_flow
from backend.routes.chat_parts import endpoints as chat_endpoints


def _request(user_id: str = "owner"):
    return SimpleNamespace(state=SimpleNamespace(user_id=user_id))


def _config():
    return SimpleNamespace(
        models=SimpleNamespace(pi_model="test-model"),
        temperatures=SimpleNamespace(pi=0.1),
    )


@pytest.mark.asyncio
async def test_invalid_branch_reference_is_rejected_before_upload_or_message_mutation(
    monkeypatch,
):
    state = AnalysisState(
        id="chat-followup-preflight",
        user_id="owner",
        restore_branch_id="branch-a",
    )
    commit = AsyncMock(side_effect=AssertionError("upload mutation was reached"))
    resolve = MagicMock(side_effect=AssertionError("reference resolver was reached"))
    restore = MagicMock(side_effect=AssertionError("runtime switch was reached"))

    async def require_owner(*, session_id, user_id):
        assert session_id == state.id
        assert user_id == state.user_id
        return state

    async def resolve_config(_user_id):
        return _config()

    monkeypatch.setattr(chat_endpoints, "require_session_ownership", require_owner)
    monkeypatch.setattr(chat_endpoints, "resolve_config_for_user", resolve_config)
    monkeypatch.setattr(chat_endpoints, "_commit_chat_uploads", commit)
    monkeypatch.setattr(chat_endpoints, "resolve_followup_reference", resolve)
    monkeypatch.setattr(chat_endpoints, "ensure_session_loaded", restore)
    monkeypatch.setattr(
        chat_endpoints,
        "_build_chat_context",
        lambda *_args: ("", []),
    )
    with pytest.raises(HTTPException) as caught:
        await chat_endpoints.chat_with_pi(
            _request(),
            state.id,
            message="continue this question",
            target_agents="[]",
            file_groups="[]",
            file_samples="[]",
            file_types="[]",
            message_id="",
            files=[SimpleNamespace(filename="new.csv")],
            followup_id="fu-main",
            branch_id="main",
        )

    assert caught.value.status_code == 400
    assert state.chat_messages == []
    commit.assert_not_awaited()
    resolve.assert_not_called()
    restore.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_name", ["literature", "bioinformatics"])
async def test_delegation_receives_server_resolved_context_and_keeps_handoff_identity(
    monkeypatch, agent_name,
):
    chat_delegation.delegation_flow.bind_facade(chat_delegation)
    state = AnalysisState(id="chat-followup-delegation", user_id="owner")
    state.chat_messages = [{
        "id": "placeholder-id",
        "role": "assistant",
        "content": "Working",
        "timestamp": "2026-08-04T00:00:00",
        "delegation_active": True,
        "reply_to": "user-id",
        "followup_id": "fu-delegation",
        "branch_id": "branch-a",
    }]
    source_context = "EARLY_TRUSTED_DIRECTION_CONTEXT"
    delegate_instructions = []
    ws_events = []

    async def fake_execute_agent(
        _session_id, _state, _agent_name, instruction, _client, _callback, _files,
    ):
        delegate_instructions.append(instruction)
        if agent_name == "bioinformatics":
            return {
                "findings": ["Delegate completed a score."],
                "chat_review": {"status": "accept"},
                "figures": [{"name": "reviewed", "path": "/tmp/reviewed.jpg"}],
                "analysis_log": [AnalysisStep(
                    step_number=2, step_type="report_findings",
                    findings_summary="Executed panel: GeneA, GeneB.",
                    evidence_execution_ids=["delegated-execution"],
                )],
                "code_history": [CodeExecution(
                    id="delegated-execution",
                    code="score(genes=['GeneA', 'GeneB'], bins=25)",
                    output="Input is log-normalized.",
                )],
            }, []
        return {"summary": "delegate completed", "findings": []}, []

    class Provider:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_model(self, _agent_name):
            return "test-model"

        def get_temperature(self, _agent_name):
            return 0.1

        async def budget_text_dispatch(self, _agent_name, messages, **_kwargs):
            return SimpleNamespace(messages=list(messages), max_tokens=128)

        async def acknowledge_text_dispatch(self, *_args, **_kwargs):
            return None

    class Client:
        async def chat(self, **kwargs):
            self.last_kwargs = kwargs
            return SimpleNamespace(content=(
                "SYNTHESIS_RESPONSE [[FIGURE:raw_debug]] "
                "[[GENERATE_FIGURE:draw another plot]]"
                if agent_name == "bioinformatics" else "SYNTHESIS_RESPONSE"
            ))

    client = Client()
    manager = SimpleNamespace(openrouter=client)

    async def send_ws(_session_id, msg_type, content):
        ws_events.append((msg_type, content))

    monkeypatch.setattr(delegation_flow, "_execute_chat_agent", fake_execute_agent)
    monkeypatch.setattr(delegation_flow, "AgentSettingsProvider", Provider)
    monkeypatch.setattr(delegation_flow, "_send_chat_ws", send_ws)
    generate = AsyncMock(side_effect=AssertionError("post-review execution bypass"))
    monkeypatch.setattr(delegation_flow, "_generate_figures_in_chat", generate)
    monkeypatch.setattr(delegation_flow, "_build_chat_context", lambda *_: (
        "", [{"name": "raw_debug", "path": "/tmp/raw_debug.jpg"}],
    ))
    monkeypatch.setattr(
        delegation_flow,
        "_persist_chat_terminal",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "backend.utils.agent_llm.get_agent_llm_client",
        lambda *_args: client,
    )

    result = await delegation_flow._execute_delegation_success(
        state.id,
        state,
        {
            "plan": "delegate the selected question",
            "delegates": [{
                "agent": agent_name,
                "instruction": "Assess the selected question",
                "priority": 1,
            }],
            "synthesis_instruction": "Tie the result back to the question.",
        },
        "continue this question",
        [],
        manager,
        SystemConfig(),
        time.time(),
        [],
        followup_id="fu-delegation",
        branch_id="branch-a",
        source_context=source_context,
    )

    assert result is not delegation_flow._DELEGATION_STOP
    assert source_context in delegate_instructions[0]
    assert "Original user request" in delegate_instructions[0]
    assert source_context in client.last_kwargs["messages"][0]["content"]
    if agent_name == "bioinformatics":
        synthesis_input = client.last_kwargs["messages"][0]["content"]
        assert "Executed panel: GeneA, GeneB." in synthesis_input
        assert "score(genes=['GeneA', 'GeneB'], bins=25)" in synthesis_input
        assert "delegated-execution" in synthesis_input
        assert "Input is log-normalized." in synthesis_input
    terminal = state.chat_messages[-1]
    if agent_name == "bioinformatics":
        assert [figure["name"] for figure in terminal["figures"]] == ["reviewed"]
        assert "remain unfinished" in terminal["content"]
        generate.assert_not_awaited()
    assert terminal["id"] == "placeholder-id"
    assert terminal["reply_to"] == "user-id"
    assert terminal["followup_id"] == "fu-delegation"
    assert terminal["branch_id"] == "branch-a"
    result_events = [event for event in ws_events if event[0] == "chat_result"]
    assert result_events and result_events[-1][1]["followup_id"] == "fu-delegation"
