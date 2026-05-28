"""Helpers to keep OpenAI-style tool_call / tool-result messages paired.

An assistant message carrying tool_calls MUST be followed by a role=tool result
for every tool_call id, or upstream providers (OpenAI/Gemini) reject the next
request with HTTP 400. These helpers detect and repair missing pairings.
"""
from typing import Any, Dict, List, Sequence, Set


def _assistant_tool_call_ids(messages: Sequence[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    ids.append(tc_id)
    return ids


def _answered_tool_call_ids(messages: Sequence[Dict[str, Any]]) -> Set[str]:
    return {
        m.get("tool_call_id")
        for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id")
    }


def tool_calls_are_paired(messages: Sequence[Dict[str, Any]]) -> bool:
    """True iff every assistant tool_call id has a matching role=tool result."""
    answered = _answered_tool_call_ids(messages)
    return all(tc_id in answered for tc_id in _assistant_tool_call_ids(messages))


def append_missing_tool_results(
    messages: List[Dict[str, Any]],
    tool_call_ids: Sequence[str],
    note: str,
) -> int:
    """Append placeholder role=tool results for any unanswered tool_call ids.

    Returns the number of placeholders added. Mutates `messages` in place.
    """
    answered = _answered_tool_call_ids(messages)
    added = 0
    for tc_id in tool_call_ids:
        if tc_id and tc_id not in answered:
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": note})
            answered.add(tc_id)
            added += 1
    return added
