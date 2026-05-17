"""Tests for BaseAgent message history truncation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import MagicMock
from backend.agents.base import BaseAgent
from backend.models.schemas import AgentType


class _ConcreteAgent(BaseAgent):
    """Concrete subclass for testing — BaseAgent is abstract."""
    async def process(self, state, instruction, **kwargs):
        return {}


def _make_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = "test prompt"
    provider.get_temperature.return_value = 0.5
    provider.get_vision_setting.return_value = False
    provider.get_llm_client.return_value = MagicMock()
    return _ConcreteAgent(AgentType.PI, provider, session_id="test")


def test_history_kept_within_limit():
    """When history is within the limit, no truncation occurs."""
    agent = _make_agent()
    for i in range(50):
        agent.add_to_history("user", f"msg {i}")
    messages = agent.get_messages_for_llm("final question")
    # system + 50 history + 1 user = 52
    assert len(messages) == 52


def test_history_truncated_at_limit():
    """When history exceeds MAX_HISTORY_PAIRS * 2, oldest messages are dropped."""
    agent = _make_agent()
    # Add 200 history messages (100 pairs) — exceeds the 60-pair limit
    for i in range(200):
        agent.add_to_history("user" if i % 2 == 0 else "assistant", f"msg {i}")
    messages = agent.get_messages_for_llm("final question")
    # system + 120 (60*2) kept history + 1 user = 122
    history_in_messages = messages[1:-1]  # exclude system and final user
    assert len(history_in_messages) == 120
    # Should keep the *newest* messages
    assert "msg 199" in history_in_messages[-1]["content"]


def test_history_truncation_no_exception():
    """Truncation should not raise exceptions."""
    agent = _make_agent()
    for i in range(200):
        agent.add_to_history("user", f"msg {i}")
    messages = agent.get_messages_for_llm("question")
    assert len(messages) > 0
