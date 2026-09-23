"""Tests for swallowed-error handling: prompt guardrail (Fix A), the
steering-feedback builder, and the main-loop detection/lookup helper (Fix B)."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schemas import AgentType


def _make_agent(technique_agent=None):
    """Build a minimal agent for testing _detect_and_lookup_swallowed_error
    without running __init__ (heavy deps)."""
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent.technique_agent = technique_agent
    agent.sandbox = SimpleNamespace()
    return agent


def _result(error="", output=""):
    return SimpleNamespace(error=error, output=output)


_SWALLOWED_OUTPUT = (
    "{'condition': 'Con', 'status': 'failed', "
    "'error': \"AttributeError(\\\"module 'scvelo' has no attribute 'read'\\\")\", "
    "'decision': 'skip_velocity'}"
)


def _render_static_prompt() -> str:
    """Render _static_system_prompt without running __init__ (heavy deps).

    The static prompt f-string references self.system_prompt (a property that
    reads from self._provider), self._infer_modality(), and self.sandbox.working_dir.
    """
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent.agent_type = AgentType.BIOINFORMATICS
    agent._provider = SimpleNamespace(get_system_prompt=lambda _: "BASE-SYSTEM-PROMPT")
    agent._cached_static_sp_key = None
    agent.sandbox = SimpleNamespace(working_dir="/tmp/work")
    agent._infer_modality = lambda: "scRNA-seq"
    return agent._static_system_prompt


def test_prompt_drops_misleading_auto_lookup_sentence():
    """The false 'automatically look up ... if you hit the same API error twice'
    promise must be gone — that auto-lookup never ran in the main loop."""
    prompt = _render_static_prompt()
    assert "if you hit the same API error twice" not in prompt


def test_prompt_forbids_swallow_and_skip():
    """Prompt must explicitly forbid try/except-swallow-then-skip of API errors
    and require lookup_documentation + retry, with a method fallback."""
    prompt = _render_static_prompt()
    assert "NEVER swallow an API error" in prompt
    assert "lookup_documentation" in prompt
    assert "DPT/PAGA" in prompt


def test_prompt_requires_readonly_input_diagnosis_before_skip():
    """The rendered agent prompt must separate access failures from missing data."""
    prompt = _render_static_prompt()

    assert 'with loompy.connect(path, mode="r") as ds:' in prompt
    assert "retry in read-only mode before deciding to skip" in prompt
    assert "An unreadable file does not establish" in prompt
    assert "confirmed missing data after successful" in prompt
    assert "the PI decides scientific feasibility" in prompt
    assert "Do not change input permissions or weaken the sandbox" in prompt


def test_prompt_requires_collision_safe_cross_file_barcode_helper():
    prompt = _render_static_prompt()

    assert "from scai_barcodes import canonicalize_barcodes, barcode_index" in prompt
    assert "sample prefixes" in prompt
    assert "-1" in prompt
    assert "Loom" in prompt and "x" in prompt
    assert "collision" in prompt.lower()
    assert "ad-hoc" in prompt
    assert "split" in prompt and "replace" in prompt
    assert "regex" in prompt


def test_build_swallowed_error_feedback_includes_type_and_docs():
    from backend.agents.bioinformatics_agent import _build_swallowed_error_feedback
    msg = _build_swallowed_error_feedback(
        "AttributeError", "scvelo.read_loom(path) is the correct call."
    )
    assert "AttributeError" in msg
    assert "do NOT skip" in msg.lower() or "do not skip" in msg.lower()
    assert "scvelo.read_loom" in msg


def test_build_swallowed_error_feedback_without_docs():
    from backend.agents.bioinformatics_agent import _build_swallowed_error_feedback
    msg = _build_swallowed_error_feedback("AttributeError", "")
    assert "AttributeError" in msg
    assert "lookup_documentation" in msg


# --- Main-loop helper: _detect_and_lookup_swallowed_error (the path that
#     regressed with an UnboundLocalError when a hard error was hit first) ---

@pytest.mark.asyncio
async def test_hard_error_returns_none_and_never_calls_technique():
    """Hard result.error path must short-circuit: no detection, no lookup, and
    a defined (None, "") return so the loop's `if swallowed_type:` never crashes."""
    tech = SimpleNamespace(lookup_from_error=AsyncMock())
    agent = _make_agent(technique_agent=tech)
    result = _result(error="AttributeError: module 'scvelo' has no attribute 'read'",
                     output="")
    swallowed_type, doc_hint = await agent._detect_and_lookup_swallowed_error(result, "code")
    assert swallowed_type is None
    assert doc_hint == ""
    tech.lookup_from_error.assert_not_called()


@pytest.mark.asyncio
async def test_swallowed_error_consults_technique_agent():
    """Swallowed error + technique agent → returns type and the looked-up docs."""
    tech = SimpleNamespace(
        lookup_from_error=AsyncMock(return_value="scvelo.read_loom(path) is correct.")
    )
    agent = _make_agent(technique_agent=tech)
    result = _result(error="", output=_SWALLOWED_OUTPUT)
    swallowed_type, doc_hint = await agent._detect_and_lookup_swallowed_error(result, "code")
    assert swallowed_type == "AttributeError"
    assert doc_hint == "scvelo.read_loom(path) is correct."
    tech.lookup_from_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_swallowed_error_without_technique_agent():
    """Swallowed error but no technique agent → type returned, empty doc hint, no crash."""
    agent = _make_agent(technique_agent=None)
    result = _result(error="", output=_SWALLOWED_OUTPUT)
    swallowed_type, doc_hint = await agent._detect_and_lookup_swallowed_error(result, "code")
    assert swallowed_type == "AttributeError"
    assert doc_hint == ""


@pytest.mark.asyncio
async def test_lookup_exception_is_swallowed():
    """A failing technique lookup must not propagate; returns ('AttributeError', '')."""
    tech = SimpleNamespace(lookup_from_error=AsyncMock(side_effect=RuntimeError("boom")))
    agent = _make_agent(technique_agent=tech)
    result = _result(error="", output=_SWALLOWED_OUTPUT)
    swallowed_type, doc_hint = await agent._detect_and_lookup_swallowed_error(result, "code")
    assert swallowed_type == "AttributeError"
    assert doc_hint == ""


@pytest.mark.asyncio
async def test_clean_success_output_returns_none():
    """Normal successful output with no swallowed error → (None, "")."""
    tech = SimpleNamespace(lookup_from_error=AsyncMock())
    agent = _make_agent(technique_agent=tech)
    result = _result(error="", output="obs columns: ['leiden']\nDone.")
    swallowed_type, doc_hint = await agent._detect_and_lookup_swallowed_error(result, "code")
    assert swallowed_type is None
    assert doc_hint == ""
    tech.lookup_from_error.assert_not_called()
