"""Shared prompt-budget behavior stays best-effort and observational."""


from backend.utils.context_budget import (
    ContextBudget,
    apply_budget,
    compact_messages,
    withheld_declaration,
)
from backend.utils.tool_messages import tool_calls_are_paired


def test_uncompactable_anchor_returns_best_effort_and_declaration_without_raise():
    output, report = apply_budget(
        [{"role": "system", "content": "s" * 400_000}],
        context_window=20_000,
        completion_reserve=2_000,
    )
    assert output
    assert report.fits is False
    assert report.final_tokens > report.ceiling
    assert "dispatch continues" in output[-1]["content"]
    assert len(withheld_declaration(report)) <= 2_000


def test_compaction_preserves_anchor_recent_context_and_tool_pairing():
    messages = [{"role": "system", "content": "ANCHOR"}]
    for index in range(20):
        call_id = f"call-{index}"
        messages.extend([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "function": {"name": "run", "arguments": "x" * 1_000},
                }],
            },
            {"role": "tool", "tool_call_id": call_id, "content": "y" * 30_000},
        ])
    output, report = compact_messages(
        messages,
        ContextBudget(context_window=30_000, completion_reserve=4_000),
    )
    assert report.actions
    assert output[0]["content"] == "ANCHOR"
    assert tool_calls_are_paired(output)
