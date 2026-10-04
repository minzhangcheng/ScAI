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
    client_calls=None,
    with_image=False,
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
        async def chat(self, **kwargs):
            if client_calls is not None:
                client_calls.append(kwargs)
            if client_error:
                raise RuntimeError(client_error)
            return SimpleNamespace(content=client_content)

        chat_with_vision = chat

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
    if with_image:
        async def commit_uploads(_session_id, current, _files, **kwargs):
            current.chat_messages.append({
                "id": kwargs["message_id"], "role": "user",
                "content": kwargs["message"],
            })
            return [], ["incoming-image"], []

        monkeypatch.setattr(chat_endpoints, "_commit_chat_uploads", commit_uploads)

    from backend.state import sessions
    sessions[state.id] = state

    from backend.utils import agent_llm

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
            files=[SimpleNamespace(filename="incoming.png")] if with_image else [],
            followup_id=followup_id,
            branch_id=branch_id,
        )
    finally:
        sessions.pop(state.id, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_image", [False, True])
async def test_pi_receives_prior_figure_only_response_in_text_and_vision_history(
    monkeypatch, with_image,
):
    state = AnalysisState(id="chat-figure-history", user_id="owner")
    prior_response = {
        "role": "assistant", "content": "", "pending_figures": False,
        "figures": [{
            "name": "prior_marker_plot", "artifact_id": "prior-artifact",
            "figure_role": "publication", "review_status": "approved",
            "base64": "HISTORICAL_IMAGE_BYTES",
            "path": "/private/prior_marker_plot.png",
        }],
    }
    state.chat_messages = [
        {"role": "user", "content": "Draw the marker plot"}, prior_response,
    ]
    calls = []

    await _run_simple_chat(
        monkeypatch, state, client_calls=calls, with_image=with_image,
    )

    if with_image:
        history = calls[0]["text_prompt"]
        assert calls[0]["images"] == ["incoming-image"]
    else:
        history = calls[0]["messages"][2]["content"]
        assert calls[0]["messages"][-1] == {"role": "user", "content": "hello"}
    assert "prior_marker_plot" in history and "prior-artifact" in history
    assert '"pending_figures": false' in history
    assert "historical context, not a request to repeat work" in history
    assert "HISTORICAL_IMAGE_BYTES" not in history and "/private/" not in history
    assert state.chat_messages[1] is prior_response
    assert prior_response["content"] == ""
    assert prior_response["figures"][0]["base64"] == "HISTORICAL_IMAGE_BYTES"


def test_figure_history_keeps_pending_and_incomplete_status_visible():
    from backend.routes.chat_parts.endpoint_flow import _chat_history_content

    history = _chat_history_content({
        "role": "assistant", "content": "Usable partial results",
        "pending_figures": True, "execution_incomplete": True,
        "chat_review_ids": ["partial-review"],
        "figures": [{"name": "partial_plot"}],
    })

    assert history.startswith("Usable partial results")
    assert '"pending_figures": true' in history
    assert '"execution_incomplete": true' in history
    assert "partial-review" in history and "partial_plot" in history


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
