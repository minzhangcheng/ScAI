import backend.agents.bioinformatics_agent as bio_mod


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
    src = open(bio_mod.__file__).read()

    # Find the dispatch chain
    start = src.find('if tool_name == "execute_code":')
    assert start > 0, "could not locate dispatch chain"

    # Find the next dedent marker after the chain
    end_marker = src.find("# Clear reasoning after all tool calls", start)
    assert end_marker > 0, "could not locate end-of-chain marker"

    chain = src[start:end_marker]
    # Must contain an `else:` at the same dedent depth as the elif branches
    # (16 spaces) and write a role=tool response inside it.
    assert re.search(r"\n {16}else:\s*\n", chain), \
        "dispatch chain missing else branch for unknown tool names"
    # The else branch must append a tool-role message so the conversation
    # stays well-formed.
    assert 'KNOWN_TOOL_NAMES' in chain, "else branch should reference KNOWN_TOOL_NAMES"
    assert '"role": "tool"' in chain or "'role': 'tool'" in chain


def test_unknown_tool_message_template_includes_known_names_sorted():
    """The error message must list known tool names so the LLM can self-correct."""
    src = open(bio_mod.__file__).read()
    # Find the else branch by locating the comment unique to it
    start = src.find("# Unknown tool name — most commonly")
    assert start > 0, "could not locate unknown-tool else branch"
    snippet = src[start:start + 2000]
    assert "sorted(KNOWN_TOOL_NAMES)" in snippet, \
        "else branch must list tool names alphabetically for stable error messages"
    assert "Available tools:" in snippet


def test_unknown_tool_in_dispatch_appends_role_tool_message():
    """Verify the else branch actually appends a role=tool entry to messages.
    This is what prevents the orphan tool_call -> 400 chain."""
    src = open(bio_mod.__file__).read()
    start = src.find("# Unknown tool name — most commonly")
    end = src.find("# Clear reasoning", start)
    branch = src[start:end]
    # Must append a dict with role=tool and tool_call_id keyed off tool_call.id
    assert '"role": "tool"' in branch
    assert '"tool_call_id": tool_call.id' in branch
