"""Tests for _estimate_context_tokens and _compact_context in BioinformaticsAgent."""

import json
import pytest

from backend.agents.bioinformatics_agent import BioinformaticsAgent


def _has_orphan_tool_result(messages: list) -> bool:
    """True if any role=tool message lacks a preceding assistant with matching tool_call_id."""
    answered_ids = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    answered_ids.add(tc_id)
        if m.get("role") == "tool":
            if m.get("tool_call_id") not in answered_ids:
                return True
    return False


def _make_messages(
    n_tool_results: int = 0,
    tool_result_chars: int = 50_000,
    n_assistant_with_args: int = 0,
    args_chars: int = 20_000,
) -> list:
    msgs = [
        {"role": "system", "content": "You are a bioinformatics analyst."},
        {"role": "user", "content": "[Analysis Context]\nData info here."},
        {"role": "user", "content": "Analysis instruction: Do analysis."},
    ]
    for i in range(n_tool_results):
        tc_id = f"tc_{i}"
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "arguments": json.dumps({
                            "code": f"# step {i}\nprint('hello')",
                            "description": f"Step {i}",
                            "step_type": "technical",
                        }),
                    },
                }
            ],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": "x" * tool_result_chars,
        })
    for i in range(n_assistant_with_args):
        tc_id = f"tc_args_{i}"
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "arguments": "a" * args_chars,
                    },
                }
            ],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": "small result",
        })
    return msgs


class TestEstimateContextTokens:

    def test_empty_messages(self):
        assert BioinformaticsAgent._estimate_context_tokens([]) == 0

    def test_counts_string_content(self):
        msgs = [{"role": "user", "content": "a" * 350}]
        est = BioinformaticsAgent._estimate_context_tokens(msgs)
        assert est == 100  # 350 / 3.5

    def test_counts_tool_calls_arguments(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {
                            "name": "execute_code",
                            "arguments": "b" * 350,
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "small"},
        ]
        est = BioinformaticsAgent._estimate_context_tokens(msgs)
        # arguments(350) + small(5) + tc_1 id overhead ~ 355+5 / 3.5 ≈ 102
        assert est > 90  # arguments are counted

    def test_arguments_not_double_counted_with_content(self):
        msgs_content_only = [{"role": "user", "content": "x" * 350}]
        msgs_with_args = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "y" * 350},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": ""},
        ]
        est_content = BioinformaticsAgent._estimate_context_tokens(msgs_content_only)
        est_args = BioinformaticsAgent._estimate_context_tokens(msgs_with_args)
        # Both have ~350 significant chars, estimates should be similar
        assert abs(est_content - est_args) < 30

    def test_large_realistic_context(self):
        msgs = _make_messages(n_tool_results=6, tool_result_chars=80_000)
        est = BioinformaticsAgent._estimate_context_tokens(msgs)
        # 6 * 80K = 480K chars from tool results + overhead ≈ 137K tokens
        assert 120_000 < est < 160_000


class TestCompactContext:

    def test_no_compaction_below_threshold(self):
        msgs = [{"role": "user", "content": "hello"}]
        result, compacted = BioinformaticsAgent._compact_context(msgs, 100_000)
        assert not compacted
        assert result == msgs

    def test_compaction_reduces_estimated_tokens(self):
        msgs = _make_messages(n_tool_results=8, tool_result_chars=80_000)
        est_before = BioinformaticsAgent._estimate_context_tokens(msgs)
        assert est_before > 105_000
        result, compacted = BioinformaticsAgent._compact_context(msgs, 105_000)
        assert compacted
        est_after = BioinformaticsAgent._estimate_context_tokens(result)
        assert est_after <= 105_000

    def test_preserves_system_and_dynamic_context(self):
        msgs = _make_messages(n_tool_results=8, tool_result_chars=80_000)
        result, _ = BioinformaticsAgent._compact_context(msgs, 105_000)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == msgs[0]["content"]
        assert result[1]["role"] == "user"
        assert result[1]["content"] == msgs[1]["content"]

    def test_preserves_recent_messages(self):
        msgs = _make_messages(n_tool_results=8, tool_result_chars=80_000)
        recent_original = [dict(m) for m in msgs[-6:]]
        result, _ = BioinformaticsAgent._compact_context(msgs, 105_000)
        result_recent = result[-6:]
        for orig, res in zip(recent_original, result_recent):
            assert orig["role"] == res["role"]
            assert orig.get("tool_call_id") == res.get("tool_call_id")

    def test_tool_calls_are_paired_after_compaction(self):
        from backend.utils.tool_messages import tool_calls_are_paired
        msgs = _make_messages(n_tool_results=8, tool_result_chars=80_000)
        assert tool_calls_are_paired(msgs)
        result, _ = BioinformaticsAgent._compact_context(msgs, 105_000)
        assert tool_calls_are_paired(result)
        assert not _has_orphan_tool_result(result)

    def test_assistant_arguments_not_truncated(self):
        msgs = _make_messages(
            n_tool_results=4, tool_result_chars=80_000,
            n_assistant_with_args=4, args_chars=20_000,
        )
        original_args = []
        for m in msgs:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    original_args.append(tc["function"]["arguments"])

        result, _ = BioinformaticsAgent._compact_context(msgs, 105_000)

        result_args = []
        for m in result:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    result_args.append(tc["function"]["arguments"])

        for orig, res in zip(original_args, result_args):
            assert orig == res

    def test_truncates_tool_user_content_not_assistant(self):
        big_content = "A" * 100_000
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "[Analysis Context]\ndata"},
            {"role": "user", "content": "instruction"},
            {"role": "tool", "tool_call_id": "tc_1", "content": big_content},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc_2", "type": "function", "function": {"name": "f", "arguments": "B" * 50_000}},
            ]},
            {"role": "tool", "tool_call_id": "tc_2", "content": "small"},
        ]
        # Need system + dynamic context at [0],[1]
        msgs.insert(0, msgs.pop(0))

        result, compacted = BioinformaticsAgent._compact_context(msgs, 30_000)
        assert compacted
        for m in result:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    assert tc["function"]["arguments"] == "B" * 50_000

    def test_converges_with_very_large_protected_window(self):
        # 10 recent messages, each 80K → protected window alone is 800K
        msgs = _make_messages(n_tool_results=10, tool_result_chars=80_000)
        result, compacted = BioinformaticsAgent._compact_context(msgs, 50_000)
        # Phase 3 (fallback) must truncate protected window to converge
        est = BioinformaticsAgent._estimate_context_tokens(result)
        assert est <= 50_000

    def test_drop_phase_preserves_pairing(self):
        from backend.utils.tool_messages import tool_calls_are_paired
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "[Analysis Context]\ndata"},
        ]
        for i in range(20):
            tc_id = f"tc_{i}"
            msgs.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "f", "arguments": f"arg{i}"}}],
            })
            msgs.append({"role": "tool", "tool_call_id": tc_id, "content": "r" * 30_000})

        result, compacted = BioinformaticsAgent._compact_context(msgs, 50_000)
        assert compacted
        assert tool_calls_are_paired(result)
        assert not _has_orphan_tool_result(result)
        est = BioinformaticsAgent._estimate_context_tokens(result)
        assert est <= 50_000
        assert len(result) < len(msgs)

    def test_many_small_messages_triggers_drop(self):
        from backend.utils.tool_messages import tool_calls_are_paired
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "[Analysis Context]\ndata"},
        ]
        for i in range(60):
            tc_id = f"tc_{i}"
            msgs.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "f", "arguments": f"arg{i}" * 200}}],
            })
            msgs.append({"role": "tool", "tool_call_id": tc_id, "content": "r" * 5000})

        est_before = BioinformaticsAgent._estimate_context_tokens(msgs)
        assert est_before > 50_000

        result, compacted = BioinformaticsAgent._compact_context(msgs, 50_000)
        assert compacted
        est_after = BioinformaticsAgent._estimate_context_tokens(result)
        assert est_after <= 50_000
        assert est_after < est_before
        assert tool_calls_are_paired(result)
        assert not _has_orphan_tool_result(result)
        assert len(result) < len(msgs)

    def test_compacted_false_when_no_reduction(self):
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hello"},
        ]
        result, compacted = BioinformaticsAgent._compact_context(msgs, 100_000)
        assert not compacted
        assert result == msgs
