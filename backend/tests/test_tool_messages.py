"""E1/E2: every assistant tool_call must have a paired role=tool result."""
from backend.utils.tool_messages import tool_calls_are_paired, append_missing_tool_results


def _assistant(*ids):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": i, "type": "function"} for i in ids]}


def _tool(tc_id):
    return {"role": "tool", "tool_call_id": tc_id, "content": "ok"}


def test_paired_when_all_answered():
    msgs = [_assistant("a", "b"), _tool("a"), _tool("b")]
    assert tool_calls_are_paired(msgs) is True


def test_unpaired_when_one_missing():
    msgs = [_assistant("a", "b"), _tool("a")]  # 'b' has no result
    assert tool_calls_are_paired(msgs) is False


def test_paired_when_no_tool_calls():
    msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    assert tool_calls_are_paired(msgs) is True


def test_append_missing_adds_only_unanswered():
    msgs = [_assistant("a", "b", "c"), _tool("a")]
    added = append_missing_tool_results(msgs, ["a", "b", "c"], note="skipped")
    assert added == 2
    assert tool_calls_are_paired(msgs) is True
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert {m["tool_call_id"] for m in tool_msgs} == {"a", "b", "c"}
    assert all(m["content"] for m in tool_msgs)


def test_append_missing_noop_when_all_present():
    msgs = [_assistant("a"), _tool("a")]
    added = append_missing_tool_results(msgs, ["a"], note="skipped")
    assert added == 0
