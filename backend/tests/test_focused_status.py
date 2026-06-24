"""focused_evidence_status: 计数映射四态;judge 携带 bucket/markers context;limited 不丢弃。"""
import json
import pytest
from backend.agents.literature_agent import status_from_judge, LiteratureAgent


def test_status_mapping():
    assert status_from_judge({"strong": 2, "moderate": 0, "contradictory": False}) == "strong"
    assert status_from_judge({"strong": 1, "moderate": 1, "contradictory": False}) == "limited"
    assert status_from_judge({"strong": 0, "moderate": 0, "contradictory": False}) == "exhausted"
    assert status_from_judge({"strong": 0, "moderate": 3, "contradictory": True}) == "contradictory"


class _FakeLit(LiteratureAgent):
    def __init__(self, response):
        self.captured = {}
        async def _chat(message, temperature=None, **kwargs):
            self.captured["prompt"] = message
            return response
        self.chat_without_history = _chat


@pytest.mark.asyncio
async def test_judge_prompt_has_markers_and_returns_status():
    fake = _FakeLit(json.dumps({"strong": 1, "moderate": 1, "contradictory": False}))
    res = await fake._judge_focused_relevance(
        item={"candidate_celltype": "cholangiocyte", "top_markers": ["Pkhd1", "Glis3"]},
        tissue="liver", articles=[{"title": "Cholangiocyte markers", "abstract": "Pkhd1"}],
        bucket="rare_unresolved",
    )
    assert res["status"] == "limited"
    assert "Pkhd1" in fake.captured["prompt"]            # markers 进入 judge prompt
    assert "cholangiocyte" in fake.captured["prompt"].lower()
