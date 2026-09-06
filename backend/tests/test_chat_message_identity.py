"""Stable chat-message identity and progressive figure response contracts."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import AnalysisState
from backend.routes.chat_parts import delegation as chat_delegation
from backend.routes.chat_parts import endpoints as chat_endpoints


async def _run_simple_chat(
    monkeypatch,
    state,
    *,
    message_id="",
    followup_id="",
    branch_id="",
    client_content="A stable answer",
    client_error="",
):
    async def require_owner(*, session_id, user_id):
        assert session_id == state.id
        assert user_id == state.user_id
        return state

    async def resolve_config(_user_id):
        return SimpleNamespace(
            models=SimpleNamespace(pi_model="test-model"),
            temperatures=SimpleNamespace(pi=0.1),
        )

    class Client:
        async def chat(self, **_kwargs):
            if client_error:
                raise RuntimeError(client_error)
            return SimpleNamespace(content=client_content)

    monkeypatch.setattr(chat_endpoints, "require_session_ownership", require_owner)
    monkeypatch.setattr(chat_endpoints, "resolve_config_for_user", resolve_config)
    monkeypatch.setattr(
        chat_endpoints,
        "_build_chat_context",
        lambda _state, _direction_context="": ("", []),
    )
    monkeypatch.setattr(
        chat_endpoints,
        "resolve_followup_reference",
        lambda _state, requested, *, lineage_id: {
            "followup_id": requested,
            "source_context": {"direction": "trusted direction"},
        },
    )
    monkeypatch.setattr(chat_endpoints, "async_save_session", AsyncMock())
    monkeypatch.setattr(chat_delegation, "_async_save_session_under_lifecycle", AsyncMock())
    monkeypatch.setattr(chat_endpoints, "_create_chat_snapshot", AsyncMock())

    from backend.state import sessions
    sessions[state.id] = state

    from backend.services.snapshot_v2 import safe_point
    from backend.utils import agent_llm

    monkeypatch.setattr(
        safe_point,
        "get_snapshot_safe_point_manager",
        lambda: SimpleNamespace(
            is_paused=lambda _session_id: False,
            has_pending_request=lambda _session_id: False,
        ),
    )
    monkeypatch.setattr(agent_llm, "get_agent_llm_client", lambda *_args: Client())

    try:
        return await chat_endpoints.chat_with_pi(
            SimpleNamespace(state=SimpleNamespace(user_id=state.user_id)),
            state.id,
            message="hello",
            target_agents="[]",
            file_groups="[]",
            file_samples="[]",
            file_types="[]",
            message_id=message_id,
            files=[],
            followup_id=followup_id,
            branch_id=branch_id,
        )
    finally:
        sessions.pop(state.id, None)


@pytest.mark.asyncio
async def test_client_user_id_and_assistant_reply_identity_are_persisted(monkeypatch):
    state = AnalysisState(id="chat-identity", user_id="owner")
    client_message_id = "3f435a33-bf5d-474f-b037-b51de804ba9a"

    result = await _run_simple_chat(
        monkeypatch, state, message_id=client_message_id,
    )

    assert len(state.chat_messages) == 2
    user_message, assistant_message = state.chat_messages
    assert user_message["id"] == client_message_id
    assert assistant_message["id"]
    assert assistant_message["reply_to"] == client_message_id
    assert result["id"] == assistant_message["id"]
    assert result["reply_to"] == client_message_id


@pytest.mark.asyncio
@pytest.mark.parametrize("client_error", ["", "provider unavailable"])
async def test_simple_chat_terminal_paths_preserve_followup_identity(
    monkeypatch, client_error,
):
    state = AnalysisState(id="chat-followup-simple", user_id="owner")
    client_message_id = "3f435a33-bf5d-474f-b037-b51de804ba9a"

    result = await _run_simple_chat(
        monkeypatch,
        state,
        message_id=client_message_id,
        followup_id="fu-simple",
        branch_id="main",
        client_error=client_error,
    )

    user_message, assistant_message = state.chat_messages
    assert user_message["followup_id"] == "fu-simple"
    assert user_message["branch_id"] == "main"
    assert assistant_message["followup_id"] == "fu-simple"
    assert assistant_message["branch_id"] == "main"
    assert assistant_message["reply_to"] == client_message_id
    assert result["followup_id"] == "fu-simple"
    assert result["branch_id"] == "main"
    if client_error:
        assert "provider unavailable" in assistant_message["content"]


def test_delegation_terminal_replacement_preserves_followup_identity():
    state = AnalysisState(id="delegation-followup-identity", user_id="owner")
    state.chat_messages = [{
        "id": "placeholder-id",
        "role": "assistant",
        "content": "Working on it",
        "timestamp": "2026-08-04T00:00:00",
        "delegation_active": True,
        "reply_to": "user-id",
        "followup_id": "fu-delegation",
        "branch_id": "branch-a",
    }]

    assert chat_delegation._replace_latest_chat_terminal(state, {
        "role": "assistant",
        "content": "Cancelled",
        "timestamp": "2026-08-04T00:01:00",
        "pending_figures": False,
    }) is True
    terminal = state.chat_messages[0]
    assert terminal["id"] == "placeholder-id"
    assert terminal["reply_to"] == "user-id"
    assert terminal["followup_id"] == "fu-delegation"
    assert terminal["branch_id"] == "branch-a"


def test_legacy_chat_history_ids_are_stable_without_mutating_state():
    messages = [{
        "role": "assistant",
        "content": "legacy response",
        "timestamp": "2026-08-04T00:00:00",
    }]

    first = chat_endpoints._chat_history_messages("session-a", messages)
    second = chat_endpoints._chat_history_messages("session-a", messages)

    assert first[0]["id"] == second[0]["id"]
    assert "id" not in messages[0]
    assert first[0]["id"] != chat_endpoints._chat_history_messages(
        "session-b", messages,
    )[0]["id"]


def test_delegation_terminal_paths_preserve_placeholder_identity():
    state = AnalysisState(id="delegation-message-id", user_id="owner")
    state.chat_messages = [{
        "id": "placeholder-id",
        "role": "assistant",
        "content": "Working on it",
        "timestamp": "2026-08-04T00:00:00",
        "delegation_active": True,
        "reply_to": "user-id",
    }]

    replaced = chat_delegation._replace_latest_delegation_message(state, {
        "role": "assistant",
        "content": "Complete",
        "timestamp": "2026-08-04T00:01:00",
        "pending_figures": False,
    })

    assert replaced is True
    assert state.chat_messages == [{
        "id": "placeholder-id",
        "role": "assistant",
        "content": "Complete",
        "timestamp": "2026-08-04T00:01:00",
        "pending_figures": False,
        "reply_to": "user-id",
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("figure_failure", [False, True])
async def test_chat_figure_response_updates_one_stable_message(
    monkeypatch,
    figure_failure,
):
    state = AnalysisState(id="chat-progressive-figure", user_id="owner")
    client_message_id = "3f435a33-bf5d-474f-b037-b51de804ba9a"
    saves: list[list[dict]] = []

    async def require_owner(*, session_id, user_id):
        assert session_id == state.id
        assert user_id == "owner"
        return state

    async def resolve_config(_user_id):
        return SimpleNamespace(
            models=SimpleNamespace(pi_model="test-model"),
            temperatures=SimpleNamespace(pi=0.1),
        )

    async def save(_session_id, current, **_kwargs):
        saves.append([dict(message) for message in current.chat_messages])

    async def generate(_session_id, current, requests):
        assert requests == ["a labeled result plot"]
        assert current.chat_messages[-1]["pending_figures"] is True
        if figure_failure:
            raise RuntimeError("render failed")
        return [{"name": "result_plot", "path": "/data/result_plot.png"}]

    class Client:
        async def chat(self, **_kwargs):
            return SimpleNamespace(
                content=(
                    "Here is the result.\n"
                    "[[GENERATE_FIGURE:a labeled result plot]]"
                ),
            )

    monkeypatch.setattr(chat_endpoints, "require_session_ownership", require_owner)
    monkeypatch.setattr(chat_endpoints, "resolve_config_for_user", resolve_config)
    monkeypatch.setattr(chat_endpoints, "_build_chat_context", lambda _state: ("", []))
    monkeypatch.setattr(chat_endpoints, "async_save_session", save)
    monkeypatch.setattr(chat_delegation, "_async_save_session_under_lifecycle", save)
    monkeypatch.setattr(chat_endpoints, "_generate_figures_in_chat", generate)
    monkeypatch.setattr(chat_endpoints, "_create_chat_snapshot", AsyncMock())

    from backend.services.snapshot_v2 import safe_point
    from backend.utils import agent_llm

    monkeypatch.setattr(
        safe_point,
        "get_snapshot_safe_point_manager",
        lambda: SimpleNamespace(
            is_paused=lambda _session_id: False,
            has_pending_request=lambda _session_id: False,
        ),
    )
    monkeypatch.setattr(agent_llm, "get_agent_llm_client", lambda *_args: Client())

    from backend.state import sessions
    sessions[state.id] = state
    try:
        result = await chat_endpoints.chat_with_pi(
            SimpleNamespace(state=SimpleNamespace(user_id="owner")),
            state.id,
            message="show the result",
            target_agents="[]",
            file_groups="[]",
            file_samples="[]",
            file_types="[]",
            message_id=client_message_id,
            files=[],
        )
    finally:
        sessions.pop(state.id, None)

    assert len(state.chat_messages) == 2
    user_message, assistant_message = state.chat_messages
    assert user_message["id"] == client_message_id
    assert assistant_message["reply_to"] == client_message_id
    assert assistant_message["pending_figures"] is False
    expected_figures = [] if figure_failure else [
        {"name": "result_plot", "path": "/data/result_plot.png"},
    ]
    assert assistant_message.get("figures", []) == expected_figures
    assert result["id"] == assistant_message["id"]
    assert result["reply_to"] == client_message_id
    assert result["pending_figures"] is False
    assert any(snapshot[-1]["pending_figures"] is True for snapshot in saves)
    assert saves[-1][-1]["pending_figures"] is False
