"""Durable realtime payload and paginated history contracts."""

import inspect
import json
import math
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.constants import STRIPPED_FIGURE_PLACEHOLDER
from backend.models.schemas import AnalysisState, WebSocketMessage
from backend.routes import sessions as session_routes
from backend.services import session_store
from backend.utils.websocket_delivery import (
    append_durable_realtime_message,
    json_safe_payload,
    serialize_realtime_message,
)


IMAGE_BASE64 = "iVBORw0KGgo" + ("A" * 2048)


class _Kind(Enum):
    EXAMPLE = "example"


def _message(index: int) -> WebSocketMessage:
    message = WebSocketMessage(
        type="message",
        content={"index": index},
        timestamp=datetime(2026, 8, 10, 0, 0, index, tzinfo=timezone.utc),
    )
    message.id = f"m-{index:02d}"
    return message


@pytest.mark.asyncio
async def test_session_load_is_light_and_messages_are_cursor_paginated(monkeypatch):
    state = AnalysisState(id="message-page-session", user_id="owner")
    state.realtime_messages = [_message(index) for index in range(45)]

    monkeypatch.setattr(
        session_routes,
        "require_session_read_ownership",
        AsyncMock(return_value=state),
        raising=False,
    )
    request = SimpleNamespace(state=SimpleNamespace(user_id="owner"))

    main = await session_routes.load_session(state.id, request)
    assert "messages" not in main

    load_page = getattr(session_routes, "load_realtime_messages", None)
    assert load_page is not None
    newest = await load_page(state.id, request, limit=20, before=None)
    assert newest["page_size"] == 20
    assert newest["has_more"] is True
    assert newest["next_before"] == "m-25"
    assert [message["id"] for message in newest["messages"]] == [
        f"m-{index:02d}" for index in range(25, 45)
    ]
    json.dumps(newest)

    middle = await load_page(state.id, request, limit=20, before="m-25")
    assert [message["id"] for message in middle["messages"]] == [
        f"m-{index:02d}" for index in range(5, 25)
    ]

    oldest = await load_page(state.id, request, limit=20, before="m-05")
    assert [message["id"] for message in oldest["messages"]] == [
        f"m-{index:02d}" for index in range(5)
    ]
    assert oldest["has_more"] is False
    assert oldest["next_before"] is None


@pytest.mark.asyncio
async def test_message_page_is_owner_scoped_and_rejects_expired_cursor(monkeypatch):
    state = AnalysisState(id="expired-message-cursor", user_id="owner")

    async def owner_only(*, session_id, user_id):
        if user_id != state.user_id:
            raise HTTPException(status_code=404, detail="Session not found")
        return state

    monkeypatch.setattr(
        session_routes,
        "require_session_read_ownership",
        owner_only,
        raising=False,
    )
    request = SimpleNamespace(state=SimpleNamespace(user_id="owner"))
    load_page = getattr(session_routes, "load_realtime_messages", None)
    assert load_page is not None

    with pytest.raises(HTTPException) as raised:
        await load_page(state.id, request, limit=20, before="missing")
    assert raised.value.status_code == 409
    assert raised.value.detail["code"] == "MESSAGE_CURSOR_EXPIRED"

    request.state.user_id = "other"
    with pytest.raises(HTTPException) as raised:
        await load_page(state.id, request, limit=20, before=None)
    assert raised.value.status_code == 404


def test_durable_history_is_json_safe_while_live_delivery_retains_images(
    tmp_path,
    monkeypatch,
):
    state = AnalysisState(id="sanitized-stream", user_id="owner")
    legacy_ref = "@scai-blob:" + ("a" * 64)
    message = WebSocketMessage(
        type="figure",
        content={
            "image": IMAGE_BASE64,
            "legacy_text": legacy_ref,
            "when": datetime(2026, 8, 10, tzinfo=timezone.utc),
            "kind": _Kind.EXAMPLE,
        },
    )
    original_content = dict(message.content)

    append_durable_realtime_message(state, message)
    durable = state.realtime_messages[0].content
    assert durable["image"] == STRIPPED_FIGURE_PLACEHOLDER
    json.dumps(json_safe_payload(durable))
    assert message.content == original_content

    history = session_routes._serialize_realtime_history_message(message)
    assert history["content"]["legacy_text"] == legacy_ref

    direct_figure = session_routes._serialize_realtime_history_message(
        WebSocketMessage(type="figure", content=legacy_ref)
    )
    assert direct_figure["content"] == STRIPPED_FIGURE_PLACEHOLDER

    pi_response = session_routes._serialize_realtime_history_message(
        WebSocketMessage(type="pi_response", content=legacy_ref)
    )
    assert pi_response["content"] == legacy_ref

    live = serialize_realtime_message(message, strip_binary=False)
    assert live["content"]["image"] == IMAGE_BASE64

    cyclic = {"when": datetime(2026, 8, 10, tzinfo=timezone.utc)}
    cyclic["self"] = cyclic
    safe = json_safe_payload({"cycle": cyclic, "kind": _Kind.EXAMPLE})
    json.dumps(safe)

    class Unknown:
        def __str__(self):
            return "unknown-value"

    durable_state = AnalysisState(id="roundtrip-safe", user_id="owner")
    raw_secret = "sk-p3-durable-secret"
    durable_state.agent_server_configs = {
        "pi": {"llm_base_url": "https://example.test/v1", "api_key": raw_secret},
    }
    durable_content = {
        "cycle": cyclic,
        "unknown": Unknown(),
        "path": Path("/tmp/example"),
        "set": {"b", "a"},
        "kind": _Kind.EXAMPLE,
        "nan": math.nan,
        "infinity": math.inf,
        "binary": b"payload",
    }
    durable_state.realtime_messages = [WebSocketMessage(
        type="status",
        content=durable_content,
    )]
    dumped = session_store._durable_state_dump(durable_state)
    rendered = json.dumps(dumped, allow_nan=False)
    assert raw_secret not in rendered
    roundtrip_content = dumped["realtime_messages"][0]["content"]
    assert roundtrip_content["unknown"] == "unknown-value"
    assert roundtrip_content["path"] == "/tmp/example"
    assert roundtrip_content["set"] == ["a", "b"]
    assert roundtrip_content["kind"] == "example"
    assert roundtrip_content["nan"] is None
    assert roundtrip_content["infinity"] is None
    assert roundtrip_content["binary"] == STRIPPED_FIGURE_PLACEHOLDER
    assert roundtrip_content["cycle"]["self"] == "[cycle]"

    # Exercise the actual durable writer/loader boundary as well as the
    # detached dump: no arbitrary realtime value may reach model_dump/json.
    session_root = tmp_path / "sessions"
    monkeypatch.setattr(session_store, "SESSION_DIR", session_root)
    import backend.config._config as config_module
    import backend.services.session_lifecycle as lifecycle

    monkeypatch.setattr(config_module, "SESSION_DIR", session_root)
    monkeypatch.setattr(lifecycle, "SESSION_DIR", session_root)
    session_store.save_session_to_disk("roundtrip-safe", durable_state)
    restored = session_store.load_session_from_disk("roundtrip-safe")
    assert restored is not None
    restored_content = restored.realtime_messages[0].content
    assert restored_content["cycle"]["self"] == "[cycle]"
    assert restored_content["unknown"] == "unknown-value"
    assert restored_content["binary"] == STRIPPED_FIGURE_PLACEHOLDER
    assert restored.agent_server_configs["pi"]["api_key"] == raw_secret
    json.dumps(restored.model_dump(mode="json"), allow_nan=False)


def test_websocket_attach_does_not_push_bulk_message_history():
    from backend.routes.websockets import websocket_endpoint

    source = inspect.getsource(websocket_endpoint)
    assert "message_history" not in source
