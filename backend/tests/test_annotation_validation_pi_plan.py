import json
import pytest
from backend.agents.pi_agent import PIAgent


def _agent(response):
    a = PIAgent.__new__(PIAgent); a.session_id = "s"
    async def fake_chat(prompt, temperature=None):
        a.last_prompt = prompt; return response
    a.chat = fake_chat
    a.refresh_runtime_config_if_needed = lambda: None
    return a


@pytest.mark.asyncio
async def test_plan_returns_structured_dict_and_surfaces_context():
    plan = {"scope": "annotation_validation_only",
            "tasks": [{"task_id": "t1", "cluster_ids": ["5"], "question": "lineage?"}]}
    a = _agent(json.dumps(plan))
    out = await a.generate_annotation_validation_questions(
        cluster_markers={"5": ["A"]}, annotations={"5": {"cell_type": "X", "confidence": "low"}},
        tissue="liver", organism="mouse", triage={"rare_unresolved": [{"cluster_id": "5"}]},
        focused_evidence={"5": "weak NK support"}, database_priors={"5": ["Stromal"]},
        metadata_context={"qc_like": ["doublet_score"]})
    assert out["tasks"][0]["cluster_ids"] == ["5"]
    assert "doublet_score" in a.last_prompt
    assert "weak NK support" in a.last_prompt


@pytest.mark.asyncio
async def test_plan_empty_on_garbage():
    a = _agent("not json")
    out = await a.generate_annotation_validation_questions(
        cluster_markers={"5": ["A"]}, annotations={"5": {}}, tissue="liver", organism="mouse",
        triage={}, focused_evidence={}, database_priors={}, metadata_context={})
    assert out == {}
