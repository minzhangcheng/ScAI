"""review_cluster_annotations: prompt 含 snRNA/marker 原则;污染 reason 确定性透传 + 低置信 + 候选入 alt。"""
import json
import pytest
from backend.agents.pi_agent import PIAgent
from backend.models.schemas import AnalysisState


class _FakePI(PIAgent):
    def __init__(self, response):
        self.captured = {}
        async def _chat(message, temperature=None, **kwargs):
            self.captured["prompt"] = message
            return response
        self.chat = _chat
    def refresh_runtime_config_if_needed(self):
        pass


@pytest.mark.asyncio
async def test_principles_and_deterministic_reason_passthrough():
    # LLM 故意返回空 reason / high confidence,后处理必须纠正
    resp = json.dumps({"8": {"cell_type": "Neuronal", "confidence": "high",
                              "alternative_cell_types": [], "low_confidence_reason": ""}})
    pi = _FakePI(resp)
    state = AnalysisState(user_id="u", name="n")
    triage = {"suspected_contamination": [{"cluster_id": "8", "candidate_celltype": "neuronal",
              "reason": "high mitochondrial fraction, likely low-quality cluster"}],
              "rare_unresolved": [], "context_dependent_state": []}
    out = await pi.review_cluster_annotations(
        state, annotations={"8": {"cell_type": "Neuronal", "key_markers": ["Nrxn1"]}},
        tissue="liver", organism="mouse", triage=triage)
    p = pi.captured["prompt"]
    assert "single-nucleus" in p.lower() or "snrna" in p.lower()
    assert "marker set coherence" in p.lower() or "negative marker" in p.lower()
    assert "high mitochondrial fraction" in p              # reason 进 prompt
    # 确定性后处理: 低置信 + reason 透传 + 候选入 alt
    assert out["8"]["confidence"] == "low"
    assert out["8"]["low_confidence_reason"] == "high mitochondrial fraction, likely low-quality cluster"
    assert "neuronal" in out["8"]["alternative_cell_types"]
