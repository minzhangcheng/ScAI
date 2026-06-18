"""Pathology image-level LLM-failure HITL.

Regression for the silent-skip bug: when a pathology vision/chat call exhausts
its upstream retries, the agent must surface an LLM-failure intervention
(retry the single call / skip this image / terminate) instead of silently
dropping the image.
"""
import pytest

import backend.agents.pathology_agent as mod
from backend.agents.pathology_agent import PathologyAgent, _ImageSkipRequested
from backend.constants import LlmFailureAction
from backend.utils.llm_client import UpstreamExhaustedError
from backend.workflow.helpers import WorkflowTerminatedError


def _bare_agent() -> PathologyAgent:
    """Build a PathologyAgent without running __init__ (avoids needing a real
    settings_provider); only the attributes used by the code under test are set."""
    agent = object.__new__(PathologyAgent)
    agent.llm_intervention_callback = None
    return agent


def _make_error() -> UpstreamExhaustedError:
    return UpstreamExhaustedError("timeout", agent_name="pathology", model="m")


@pytest.mark.asyncio
async def test_retry_reattempts_only_the_failing_call():
    agent = _bare_agent()
    calls = {"n": 0}

    async def call_factory():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _make_error()
        return "ok"

    async def cb(error, image_label):
        return LlmFailureAction.RETRY

    agent.llm_intervention_callback = cb
    result = await agent._call_llm_with_hitl(call_factory, "img.tif", "overview")
    assert result == "ok"
    assert calls["n"] == 2  # failed once, retried once — no extra work


@pytest.mark.asyncio
async def test_skip_raises_image_skip_sentinel():
    agent = _bare_agent()

    async def call_factory():
        raise _make_error()

    async def cb(error, image_label):
        return LlmFailureAction.SKIP

    agent.llm_intervention_callback = cb
    with pytest.raises(_ImageSkipRequested) as ei:
        await agent._call_llm_with_hitl(call_factory, "img.tif", "overview")
    assert ei.value.image_label == "img.tif"


@pytest.mark.asyncio
async def test_terminate_raises_workflow_terminated():
    agent = _bare_agent()

    async def call_factory():
        raise _make_error()

    async def cb(error, image_label):
        return LlmFailureAction.TERMINATE

    agent.llm_intervention_callback = cb
    with pytest.raises(WorkflowTerminatedError):
        await agent._call_llm_with_hitl(call_factory, "img.tif", "overview")


@pytest.mark.asyncio
async def test_no_callback_propagates_error():
    """Without a wired callback, the error propagates (process() then skips the
    image) — preserves legacy behaviour for non-workflow callers."""
    agent = _bare_agent()  # llm_intervention_callback is None

    async def call_factory():
        raise _make_error()

    with pytest.raises(UpstreamExhaustedError):
        await agent._call_llm_with_hitl(call_factory, "img.tif", "overview")


@pytest.mark.asyncio
async def test_process_skips_one_image_and_continues(monkeypatch):
    """A skipped image must not abort the batch: remaining images still analyzed."""
    agent = _bare_agent()
    agent.llm_intervention_callback = lambda *a: None  # not used here

    class FakeImg:
        def __init__(self, name):
            self.id = name
            self.filename = name

    imgs = [FakeImg("a.tif"), FakeImg("b.tif")]

    async def fake_analyze(img_info, instruction):
        if img_info.filename == "a.tif":
            raise _ImageSkipRequested("a.tif", last_error=_make_error())
        return {"image_id": img_info.id, "findings": [f"finding-{img_info.filename}"]}

    async def fake_synth(instruction, detailed_results, state):
        return {"summary": "ok"}

    monkeypatch.setattr(agent, "_analyze_image", fake_analyze)
    monkeypatch.setattr(agent, "_synthesize_findings", fake_synth)

    state = type("S", (), {"get_files_by_section": lambda self, s: imgs})()
    result = await agent.process(state, "analyze", images=imgs)

    assert result["status"] == "completed"
    assert result["findings"] == ["finding-b.tif"]  # b survived, a skipped
    skipped = [r for r in result["detailed_results"] if r.get("skipped")]
    assert len(skipped) == 1 and skipped[0]["image_id"] == "a.tif"


@pytest.mark.asyncio
async def test_process_propagates_terminate(monkeypatch):
    """TERMINATE inside image analysis must abort process(), not be swallowed."""
    agent = _bare_agent()

    class FakeImg:
        id = "a.tif"
        filename = "a.tif"

    async def fake_analyze(img_info, instruction):
        raise WorkflowTerminatedError("user terminated")

    monkeypatch.setattr(agent, "_analyze_image", fake_analyze)
    imgs = [FakeImg()]
    state = type("S", (), {"get_files_by_section": lambda self, s: imgs})()
    with pytest.raises(WorkflowTerminatedError):
        await agent.process(state, "analyze", images=imgs)
