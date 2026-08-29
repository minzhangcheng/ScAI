"""Shared prompt-budget behavior stays best-effort and observational."""

import json

from backend.utils.context_budget import (
    CHARS_PER_TOKEN,
    ContextBudget,
    apply_budget,
    compact_messages,
    estimate_text_tokens,
    estimate_tokens,
    withheld_declaration,
)
from backend.utils.tool_messages import tool_calls_are_paired


def test_estimator_counts_text_lists_and_tool_arguments():
    messages = [
        {"role": "user", "content": "a" * 350},
        {"role": "user", "content": [{"type": "text", "text": "b" * 40}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "run", "arguments": "c" * 350},
            }],
        },
    ]
    assert estimate_text_tokens("a" * 350) == int(350 / CHARS_PER_TOKEN)
    assert estimate_tokens(messages) >= int(
        (700 + len(json.dumps(messages[1]["content"]))) / CHARS_PER_TOKEN
    )


def test_small_window_reduces_completion_reserve_and_keeps_prompt_positive():
    budget = ContextBudget(context_window=8_192, completion_reserve=32_768)
    assert budget.prompt_ceiling == 1
    assert budget.effective_completion_reserve == 7_679

    output, report = apply_budget(
        [{"role": "user", "content": "hello"}],
        context_window=8_192,
        completion_reserve=32_768,
    )
    assert output
    assert report.effective_completion_reserve == 7_679
    assert report.requested_completion_reserve == 32_768


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


def test_default_executor_formula_matches_the_spec_tuning():
    budget = ContextBudget(
        context_window=200_000,
        completion_reserve=32_768,
        budget_factor=0.90,
    )
    assert budget.prompt_ceiling == 150_048
    assert int(budget.prompt_ceiling * 0.70) == 105_033
