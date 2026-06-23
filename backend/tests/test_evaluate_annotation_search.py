"""evaluate_annotation_search_results: markers_summary must list ALL clusters."""
import json
import pytest
from backend.agents.pi_agent import PIAgent


class _FakePI(PIAgent):
    def __init__(self):
        self.captured = {}
        async def _chat(message, temperature=None, **kwargs):
            self.captured["prompt"] = message
            return json.dumps({"sufficient": True})
        self.chat_without_history = _chat


@pytest.mark.asyncio
async def test_evaluator_sees_all_clusters():
    pi = _FakePI()
    markers = {str(i): [f"G{i}a", f"G{i}b"] for i in range(22)}
    await pi.evaluate_annotation_search_results(
        cluster_markers=markers, organism="mouse", tissue="liver",
        articles=[], search_history=[],
    )
    p = pi.captured["prompt"]
    for i in range(22):
        assert f"Cluster {i}:" in p
