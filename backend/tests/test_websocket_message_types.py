"""Guard: every literal type passed to _send_message must be a registered
WebSocketMessage.type, or the node crashes at message construction time."""
import re
from pathlib import Path
from typing import get_args

from backend.models.schemas import WebSocketMessage


def _registered_types() -> set[str]:
    return set(get_args(WebSocketMessage.model_fields["type"].annotation))


def test_integration_types_are_registered():
    registered = _registered_types()
    assert "integration_gate" in registered
    assert "integration_complete" in registered


def test_websocket_message_accepts_integration_types():
    WebSocketMessage(type="integration_gate", content={})
    WebSocketMessage(type="integration_complete", content={})


def test_all_literal_send_message_types_are_registered():
    """Scan the whole backend/ (not just workflow/) so an unregistered literal
    _send_message type anywhere — routes, services, agents — fails here instead
    of crashing WebSocketMessage construction at runtime. Dynamic (variable)
    msg_type arguments are not matched and are out of scope for this guard."""
    registered = _registered_types()
    backend_dir = Path(__file__).resolve().parent.parent
    pattern = re.compile(r"""_send_message\(\s*["']([A-Za-z0-9_]+)["']""")
    missing: list[str] = []
    for py in backend_dir.rglob("*.py"):
        if "tests" in py.parts:
            continue
        for msg_type in pattern.findall(py.read_text(encoding="utf-8")):
            if msg_type not in registered:
                missing.append(f"{py.relative_to(backend_dir)}: {msg_type}")
    assert not missing, f"Unregistered _send_message types: {missing}"
