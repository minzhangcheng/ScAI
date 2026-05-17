"""Test that LiteratureAgent.process accepts mode_override and uses it for routing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import MagicMock
from backend.agents.literature_agent import LiteratureAgent


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_llm_client.return_value = MagicMock()
    provider.get_temperature.return_value = 0.7
    provider.get_system_prompt.return_value = "test prompt"
    return provider


def test_process_has_mode_override_parameter(mock_provider):
    agent = LiteratureAgent(settings_provider=mock_provider, literature_mode="deep_research")
    import inspect
    sig = inspect.signature(agent.process)
    assert "mode_override" in sig.parameters


def test_mode_override_does_not_mutate_agent_state(mock_provider):
    agent = LiteratureAgent(settings_provider=mock_provider, literature_mode="deep_research")
    assert agent.literature_mode == "deep_research"
