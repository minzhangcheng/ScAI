"""Pathology image-level LLM-failure HITL.

Regression for the silent-skip bug: when a pathology vision/chat call exhausts
its upstream retries, the agent must surface an LLM-failure intervention
(retry the single call / skip this image / terminate) instead of silently
dropping the image.
"""
import pytest
from unittest.mock import MagicMock

import backend.agents.pathology_agent as mod
from backend.agents.pathology_agent import PathologyAgent, _ImageSkipRequested
from backend.constants import LlmFailureAction
from backend.models.schemas import AgentType
from backend.utils.llm_client import BadRequestError, UpstreamExhaustedError
from backend.workflow.helpers import WorkflowTerminatedError
from backend.tests.split_facade import setattr_facade


def _bare_agent() -> PathologyAgent:
    """Build a PathologyAgent without running __init__ (avoids needing a real
    settings_provider); only the attributes used by the code under test are set."""
    agent = object.__new__(PathologyAgent)
    agent.llm_intervention_callback = None
    agent.agent_type = AgentType.PATHOLOGY
    agent._provider = MagicMock()
    agent._provider.get_config_revision = MagicMock(return_value=0)
    agent._last_config_revision = 0
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
async def test_deterministic_vision_shape_failure_does_not_open_hitl():
    agent = _bare_agent()
    callback_calls = 0
    provider_error = BadRequestError(
        message="Error code: 400 - invalid image url",
        response=MagicMock(status_code=400),
        body={"error": {
            "message": "Invalid image_url payload",
            "param": "messages[0].content[1].image_url",
            "code": 1214,
        }},
    )
    wrapped = _make_error()
    wrapped.__cause__ = provider_error

    async def call_factory():
        raise wrapped

    async def cb(error, image_label):
        nonlocal callback_calls
        callback_calls += 1
        return LlmFailureAction.RETRY

    agent.llm_intervention_callback = cb
    with pytest.raises(UpstreamExhaustedError):
        await agent._call_llm_with_hitl(call_factory, "img.tif", "overview")
    assert callback_calls == 0


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

    assert result["status"] == "partial"
    assert result["findings"] == ["finding-b.tif"]  # b survived, a skipped
    skipped = [r for r in result["detailed_results"] if r.get("skipped")]
    assert len(skipped) == 1 and skipped[0]["image_id"] == "a.tif"


@pytest.mark.asyncio
async def test_process_skipped_and_failed_images_preserve_image_type(monkeypatch):
    """Per-image failure records retain the stain label for later disclosure."""
    agent = _bare_agent()

    class FakeImg:
        def __init__(self, name, data_type):
            self.id = name
            self.filename = name
            self.data_type = data_type

    images = [FakeImg("skip.tif", "H&E"), FakeImg("failed.tif", "IHC")]

    async def fake_analyze(img_info, instruction):
        if img_info.filename == "skip.tif":
            raise _ImageSkipRequested(img_info.filename, last_error=_make_error())
        raise RuntimeError("image decoder failed")

    monkeypatch.setattr(agent, "_analyze_image", fake_analyze)
    state = type("S", (), {"get_files_by_section": lambda self, section: images})()

    result = await agent.process(state, "analyze", images=images)

    by_id = {item["image_id"]: item for item in result["detailed_results"]}
    assert by_id["skip.tif"]["image_type"] == "H&E"
    assert by_id["failed.tif"]["image_type"] == "IHC"


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


def _patch_analyze_image_setup(monkeypatch, agent):
    agent.tile_size = 224
    agent.max_tiles = 1
    monkeypatch.setattr(agent, "refresh_runtime_config_if_needed", lambda: None)
    monkeypatch.setattr(agent, "_set_llm_context", lambda: None)
    monkeypatch.setattr(mod.os.path, "exists", lambda path: True)
    setattr_facade(monkeypatch, mod.__name__, "load_image", lambda path: object())
    setattr_facade(
        monkeypatch, mod.__name__, "process_pathology_image", lambda *a, **k: ["tile"]
    )
    setattr_facade(
        monkeypatch, mod.__name__, "create_overview_image", lambda image: "overview"
    )

    class FakeImg:
        id = "img-1"
        filename = "img-1.tif"
        file_path = "/fake/img-1.tif"
        data_type = "H&E"

    return FakeImg()


@pytest.mark.asyncio
async def test_analyze_image_does_not_swallow_skip_sentinel(monkeypatch):
    """Exercise the real broad exception boundary inside _analyze_image."""
    agent = _bare_agent()
    image = _patch_analyze_image_setup(monkeypatch, agent)

    async def raise_skip(*args, **kwargs):
        raise _ImageSkipRequested(image.filename, last_error=_make_error())

    monkeypatch.setattr(agent, "_call_llm_with_hitl", raise_skip)
    with pytest.raises(_ImageSkipRequested):
        await agent._analyze_image(image, "analyze")


@pytest.mark.asyncio
async def test_analyze_image_does_not_swallow_terminate(monkeypatch):
    """TERMINATE must cross _analyze_image and reach the workflow boundary."""
    agent = _bare_agent()
    image = _patch_analyze_image_setup(monkeypatch, agent)

    async def raise_terminate(*args, **kwargs):
        raise WorkflowTerminatedError("user terminated")

    monkeypatch.setattr(agent, "_call_llm_with_hitl", raise_terminate)
    with pytest.raises(WorkflowTerminatedError):
        await agent._analyze_image(image, "analyze")
