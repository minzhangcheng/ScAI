"""E4: summarize must release LLM-failure AND disabled-agent gates via the
signal_llm_failure_action path (not signal_continue)."""
from types import SimpleNamespace

from backend.routes.lifecycle import resolve_hitl_skip_target
from backend.constants import WaitInputReason


def test_llm_failure_returns_node_and_generation():
    state = SimpleNamespace(
        wait_input_reason=WaitInputReason.LLM_FAILURE.value,
        llm_failure_info={"node_name": "pathology_analysis", "generation": 3},
        agent_disabled_info=None,
    )
    assert resolve_hitl_skip_target(state) == ("pathology_analysis", 3)


def test_agent_disabled_returns_node_and_generation():
    state = SimpleNamespace(
        wait_input_reason=WaitInputReason.AGENT_DISABLED.value,
        llm_failure_info=None,
        agent_disabled_info={"node_name": "bioinformatics_analysis", "generation": 7},
    )
    assert resolve_hitl_skip_target(state) == ("bioinformatics_analysis", 7)


def test_round_completion_returns_none():
    state = SimpleNamespace(
        wait_input_reason=None,
        llm_failure_info=None,
        agent_disabled_info=None,
    )
    assert resolve_hitl_skip_target(state) is None


def test_missing_info_defaults_safely():
    state = SimpleNamespace(
        wait_input_reason=WaitInputReason.AGENT_DISABLED.value,
        llm_failure_info=None,
        agent_disabled_info=None,
    )
    assert resolve_hitl_skip_target(state) == ("", 0)
