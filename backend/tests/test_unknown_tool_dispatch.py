import backend.agents.bioinformatics_agent as bio_mod
from backend.agents.bioinformatics import executor_dispatch
import inspect


def test_known_tool_names_matches_tools_list():
    """KNOWN_TOOL_NAMES must be derived from TOOLS, not hand-maintained."""
    expected = frozenset(t["function"]["name"] for t in bio_mod.TOOLS)
    assert bio_mod.KNOWN_TOOL_NAMES == expected
    assert "execute_code" in bio_mod.KNOWN_TOOL_NAMES
    assert "report_findings" in bio_mod.KNOWN_TOOL_NAMES
    assert "hallucinated_tool" not in bio_mod.KNOWN_TOOL_NAMES


def test_known_tool_names_is_immutable():
    """Use frozenset so callers cannot mutate it."""
    assert isinstance(bio_mod.KNOWN_TOOL_NAMES, frozenset)


import re


def test_dispatch_chain_has_unknown_tool_else():
    """The dispatch chain must end with an else branch that emits a
    role=tool message; otherwise hallucinated tool names create orphan
    tool_calls that the next request rejects."""
    src = inspect.getsource(executor_dispatch.dispatch_tool)
    assert "tool_name in context.known_tool_names" in src
    assert "unknown_tool_message" in src
    assert "append_tool_message" in src


def test_unknown_tool_message_template_includes_known_names_sorted():
    """The error message must list known tool names so the LLM can self-correct."""
    src = inspect.getsource(executor_dispatch.unknown_tool_message)
    assert "sorted(known_tool_names)" in src
    assert "Available tools:" in src


def test_unknown_tool_in_dispatch_appends_role_tool_message():
    """Verify the else branch actually appends a role=tool entry to messages.
    This is what prevents the orphan tool_call -> 400 chain."""
    branch = inspect.getsource(executor_dispatch.append_tool_message)
    assert '"role": "tool"' in branch
    assert '"tool_call_id": tool_call.id' in branch
