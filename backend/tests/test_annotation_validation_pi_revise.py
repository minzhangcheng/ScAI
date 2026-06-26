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
async def test_revision_merges_changes_and_summary():
    resp = {"revisions": {"7": {"cell_type": "Technical/Contamination", "confidence": "low",
                                "low_confidence_reason": "neuronal genes implausible in liver"}},
            "validation_summary_for_user": "Cluster 7 reclassified as technical."}
    a = _agent(json.dumps(resp))
    annotations = {"7": {"cell_type": "Hepatic neurons", "confidence": "medium", "key_markers": ["Nrxn1"]}}
    out = await a.revise_annotations_after_validation(
        annotations=annotations, validation_plan={"tasks": [{"task_id": "t"}]},
        validation_result={"status": "completed", "cluster_results": {}},
        literature_context="hepatic nerves rare", tissue="liver", organism="mouse")
    assert out["annotations"]["7"]["cell_type"] == "Technical/Contamination"
    assert out["annotations"]["7"]["key_markers"] == ["Nrxn1"]  # untouched fields preserved
    assert out["validation_summary_for_user"] == "Cluster 7 reclassified as technical."


@pytest.mark.asyncio
async def test_revision_keeps_originals_on_garbage():
    a = _agent("not json")
    annotations = {"7": {"cell_type": "Hepatic neurons", "confidence": "medium"}}
    out = await a.revise_annotations_after_validation(
        annotations=annotations, validation_plan={}, validation_result={"status": "failed"},
        literature_context="", tissue="liver", organism="mouse")
    assert out["annotations"]["7"]["cell_type"] == "Hepatic neurons"
    assert isinstance(out["validation_summary_for_user"], str)
