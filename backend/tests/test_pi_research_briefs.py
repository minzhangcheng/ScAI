"""PI research-brief generators: focused, background-as-reference, no citation format."""
import types
import pytest
from backend.agents.pi_agent import PIAgent


class _FakePI(PIAgent):
    """Bypass __init__; capture the prompt sent to chat_without_history."""
    def __init__(self, response="A FOCUSED BRIEF"):
        self.captured = {}
        async def _chat(message, temperature=None, **kwargs):
            self.captured["prompt"] = message
            return response
        self.chat_without_history = _chat


MARKERS = {"0": ["Cd79a", "Cd79b", "Ms4a1"], "1": ["Cd3d", "Cd3e", "Cd8a"]}


@pytest.mark.asyncio
async def test_annotation_brief_returns_text_and_includes_context():
    pi = _FakePI()
    out = await pi.generate_annotation_research_brief(MARKERS, "mouse", "liver", state=None)
    assert out == "A FOCUSED BRIEF"
    p = pi.captured["prompt"]
    assert "mouse" in p and "liver" in p
    assert "Cd79a" in p or "Cd3d" in p
    assert "FOCUSED" in p
    assert "Do NOT include any citation" in p


@pytest.mark.asyncio
async def test_annotation_brief_background_is_reference_only():
    pi = _FakePI()
    state = types.SimpleNamespace(initial_instruction="STUDY THE LIVER FIBROSIS MODEL")
    await pi.generate_annotation_research_brief(MARKERS, "mouse", "liver", state=state)
    p = pi.captured["prompt"]
    assert "reference only" in p.lower()
    assert "STUDY THE LIVER FIBROSIS MODEL" in p


@pytest.mark.asyncio
async def test_annotation_brief_strips_whitespace():
    pi = _FakePI(response="  brief with spaces  ")
    out = await pi.generate_annotation_research_brief(MARKERS, "mouse", "liver", state=None)
    assert out == "brief with spaces"


@pytest.mark.asyncio
async def test_round_brief_returns_text_and_includes_topic():
    pi = _FakePI(response="ROUND BRIEF")
    out = await pi.generate_round_research_brief("TGF-beta signaling in fibrosis", None, state=None)
    assert out == "ROUND BRIEF"
    p = pi.captured["prompt"]
    assert "TGF-beta signaling in fibrosis" in p
    assert "broad" in p.lower() and "focused" in p.lower()   # breadth judgement


@pytest.mark.asyncio
async def test_round_brief_background_reference_only():
    pi = _FakePI()
    state = types.SimpleNamespace(initial_instruction="LIVER FIBROSIS DRUG STUDY")
    await pi.generate_round_research_brief("topic X", ["finding A"], state=state)
    p = pi.captured["prompt"]
    assert "reference only" in p.lower()
    assert "LIVER FIBROSIS DRUG STUDY" in p
    assert "finding A" in p


@pytest.mark.asyncio
async def test_round_brief_no_citation_directive():
    pi = _FakePI()
    await pi.generate_round_research_brief("topic", None, state=None)
    assert "Do NOT include any citation" in pi.captured["prompt"]


@pytest.mark.asyncio
async def test_round_brief_strips_whitespace():
    pi = _FakePI(response="  round brief  ")
    out = await pi.generate_round_research_brief("topic", None, state=None)
    assert out == "round brief"


@pytest.mark.asyncio
async def test_annotation_brief_includes_all_clusters_no_gene_sample_line():
    pi = _FakePI()
    markers = {str(i): [f"G{i}a", f"G{i}b"] for i in range(22)}
    await pi.generate_annotation_research_brief(markers, "mouse", "liver", state=None)
    p = pi.captured["prompt"]
    for i in range(22):
        assert f"Cluster {i}:" in p
    assert "Key marker genes across clusters" not in p
