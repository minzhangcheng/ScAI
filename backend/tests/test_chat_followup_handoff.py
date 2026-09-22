"""Chat handoff validation and trusted-context delegation contracts."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from backend.config import SystemConfig
from backend.models.schemas import AnalysisState
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
async def test_delegation_receives_server_resolved_context_and_keeps_handoff_identity(
    monkeypatch,
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
            return SimpleNamespace(content="SYNTHESIS_RESPONSE")

    client = Client()
    manager = SimpleNamespace(openrouter=client)

    async def send_ws(_session_id, msg_type, content):
        ws_events.append((msg_type, content))

    monkeypatch.setattr(delegation_flow, "_execute_chat_agent", fake_execute_agent)
    monkeypatch.setattr(delegation_flow, "AgentSettingsProvider", Provider)
    monkeypatch.setattr(delegation_flow, "_send_chat_ws", send_ws)
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
                "agent": "literature",
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
    assert source_context in client.last_kwargs["messages"][0]["content"]
    terminal = state.chat_messages[-1]
    assert terminal["id"] == "placeholder-id"
    assert terminal["reply_to"] == "user-id"
    assert terminal["followup_id"] == "fu-delegation"
    assert terminal["branch_id"] == "branch-a"
    result_events = [event for event in ws_events if event[0] == "chat_result"]
    assert result_events and result_events[-1][1]["followup_id"] == "fu-delegation"


@pytest.mark.asyncio
async def test_cancelled_delegation_preserves_followup_identity(monkeypatch):
    chat_delegation.delegation_flow.bind_facade(chat_delegation)
    state = AnalysisState(id="chat-followup-cancel", user_id="owner")
    state.chat_messages = [{
        "id": "placeholder-cancel",
        "role": "assistant",
        "content": "Working",
        "timestamp": "2026-08-04T00:00:00",
        "delegation_active": True,
        "reply_to": "user-cancel",
        "followup_id": "fu-cancel",
        "branch_id": "branch-a",
    }]
    ws_events = []

    class FakeManager:
        def __init__(self, **_kwargs):
            self.openrouter = object()

        async def close(self):
            return None

    async def raise_cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        delegation_flow,
        "resolve_config_for_user",
        AsyncMock(return_value=SimpleNamespace(
            api=SimpleNamespace(api_key="", llm_base_url="", proxy_url=""),
        )),
    )
    monkeypatch.setattr("backend.utils.llm_client.LLMManager", FakeManager)
    monkeypatch.setattr(delegation_flow, "_execute_delegation_success", raise_cancelled)
    monkeypatch.setattr(
        delegation_flow,
        "_persist_chat_terminal",
        AsyncMock(return_value=True),
    )

    async def send_ws(_session_id, msg_type, content):
        ws_events.append((msg_type, content))

    monkeypatch.setattr(delegation_flow, "_send_chat_ws", send_ws)

    with pytest.raises(asyncio.CancelledError):
        await delegation_flow._run_chat_delegation(
            state.id,
            state,
            {"plan": "cancel", "delegates": []},
            "cancel this",
            [],
            [],
            followup_id="fu-cancel",
            branch_id="branch-a",
            source_context="trusted",
        )

    terminal = state.chat_messages[-1]
    assert terminal["id"] == "placeholder-cancel"
    assert terminal["reply_to"] == "user-cancel"
    assert terminal["followup_id"] == "fu-cancel"
    assert terminal["branch_id"] == "branch-a"
    cancel_events = [event for event in ws_events if event[0] == "chat_cancelled"]
    assert cancel_events and cancel_events[-1][1]["followup_id"] == "fu-cancel"


class _ReachedSynthesis(Exception):
    pass


def _parallel_delegation(monkeypatch, execute):
    chat_delegation.delegation_flow.bind_facade(chat_delegation)
    state = AnalysisState(id="parallel-delegation-budget", user_id="owner")
    client = SimpleNamespace()

    async def send_ws(_session_id, msg_type, content):
        if msg_type == "chat_delegation" and content.get("status") == "synthesizing":
            raise _ReachedSynthesis

    monkeypatch.setattr(delegation_flow, "_execute_chat_agent", execute)
    monkeypatch.setattr(delegation_flow, "_send_chat_ws", send_ws)
    monkeypatch.setattr("backend.utils.agent_llm.get_agent_llm_client", lambda *_args: client)
    return delegation_flow._execute_delegation_success(
        state.id, state,
        {"delegates": [
            {"agent": "literature", "instruction": "Review", "priority": 1},
            {"agent": "technique", "instruction": "Review", "priority": 1},
        ]},
        "review the evidence", [], SimpleNamespace(openrouter=client), _config(),
        time.time(), [],
    )


@pytest.mark.asyncio
async def test_parallel_delegation_does_not_cancel_later_generation_at_whole_agent_deadline(monkeypatch):
    completed = []
    monkeypatch.setattr("backend.config.CHAT_DELEGATION_TIMEOUT", 0.01)

    async def execute(_session_id, _state, name, *_args):
        for generation in range(2):
            await asyncio.sleep(0.02)
            completed.append((name, generation))
        return {"findings": []}, []

    with pytest.raises(_ReachedSynthesis):
        await _parallel_delegation(monkeypatch, execute)
    assert set(completed) == {
        (name, generation) for name in ("literature", "technique") for generation in range(2)
    }


@pytest.mark.asyncio
async def test_parallel_delegation_parent_cancel_awaits_all_child_cleanup(monkeypatch):
    started = {name: asyncio.Event() for name in ("literature", "technique")}
    cleaned = set()
    children = []

    async def execute(_session_id, _state, name, *_args):
        children.append(asyncio.current_task())
        started[name].set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.add(name)

    parent = asyncio.create_task(_parallel_delegation(monkeypatch, execute))
    await asyncio.gather(*(event.wait() for event in started.values()))
    parent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parent
    assert cleaned == set(started)
    assert all(child.done() for child in children)
