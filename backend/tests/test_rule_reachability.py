"""Every rule must actually be able to reach the model.

The corpus supports non-blocking observations and guidance retrieved on demand.
Scientific rules are deliberately unable to refuse a call; acceptance belongs
to the PI.

A rule the model never sees is worse than no rule: it makes the corpus look
larger than the system's real behaviour, and the gap is invisible from inside.
These tests close it.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import backend.agents.bioinformatics_agent as bio_mod
from backend.agents.bioinformatics import executor_collaboration
from backend.services.method_rules import ENFORCEMENT_ADVISORY, all_rules

_REPO = Path(__file__).resolve().parent.parent.parent
_BACKEND = _REPO / "backend"


def _production_sources() -> str:
    """Concatenated backend source, excluding tests."""
    chunks = []
    for path in _BACKEND.rglob("*.py"):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


_SOURCES = _production_sources()


@pytest.mark.parametrize(
    "rule", [r for r in all_rules() if r["enforcement"] == ENFORCEMENT_ADVISORY],
    ids=lambda r: r["id"])
def test_advisory_rules_are_emitted_without_blocking(rule):
    """An advisory must reach the model, and must not refuse the call."""
    assert rule["id"] in _SOURCES, (
        f"{rule['id']} is advisory but no production path emits or records it"
    )


def test_method_question_alone_is_a_valid_call():
    """Asking a method question must not require inventing a function path."""
    schema = next(t["function"]["parameters"] for t in bio_mod.TOOLS
                  if t["function"]["name"] == "lookup_documentation")
    assert "function_path" not in schema.get("required", [])


@pytest.mark.asyncio
async def test_method_question_only_does_not_call_technique_agent():
    executor_collaboration.bind_facade(bio_mod)
    technique_agent = SimpleNamespace(lookup_docs=AsyncMock())
    context = SimpleNamespace(
        agent=SimpleNamespace(technique_agent=technique_agent),
        messages=[],
        analysis_log=[],
        step_count=1,
        agent_reasoning="compare methods",
    )
    call = SimpleNamespace(
        id="method-only",
        arguments={"method_question": "multiple testing family q-value"},
    )

    await executor_collaboration.handle_lookup_documentation(context, call)

    assert "[method rule:" in context.messages[-1]["content"]
    technique_agent.lookup_docs.assert_not_awaited()


@pytest.mark.asyncio
async def test_lookup_documentation_rejects_empty_target_without_technique_call():
    executor_collaboration.bind_facade(bio_mod)
    technique_agent = SimpleNamespace(lookup_docs=AsyncMock())
    context = SimpleNamespace(
        agent=SimpleNamespace(technique_agent=technique_agent),
        messages=[],
        analysis_log=[],
        step_count=1,
        agent_reasoning="lookup docs",
    )
    call = SimpleNamespace(id="empty-lookup", arguments={})

    await executor_collaboration.handle_lookup_documentation(context, call)

    assert "requires at least one" in context.messages[-1]["content"]
    technique_agent.lookup_docs.assert_not_awaited()


# ---------------------------------------------- the specific review findings
