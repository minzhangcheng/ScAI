"""Ensure PI agent name is sourced from a single constant."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import config as cfg
from backend.models.schemas import AgentType
from backend.routes import chat, session_config, reports


def test_pi_agent_name_equals_agent_type_value():
    """PI_AGENT_NAME must be derived from AgentType.PI.value — single source of truth."""
    assert cfg.PI_AGENT_NAME == AgentType.PI.value


def test_routes_import_from_config():
    # All three routes must use the shared constant, not redefine it locally.
    assert chat.PI_AGENT_NAME is cfg.PI_AGENT_NAME
    assert session_config.PI_AGENT_NAME is cfg.PI_AGENT_NAME
    assert reports.PI_AGENT_NAME is cfg.PI_AGENT_NAME


def test_no_local_redefinitions():
    # Each route module should NOT define its own _PI_AGENT_NAME.
    for mod in (chat, session_config, reports):
        assert not hasattr(mod, "_PI_AGENT_NAME"), (
            f"{mod.__name__} still defines local _PI_AGENT_NAME; "
            "import PI_AGENT_NAME from backend.config instead"
        )
