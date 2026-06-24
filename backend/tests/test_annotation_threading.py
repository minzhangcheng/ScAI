"""串接: evaluate 闭包传 state;annotate_clusters skip_matching 返回带出 focused_evidence/triage。"""
import pytest
from backend.agents.literature_agent import LiteratureAgent


class _Lit(LiteratureAgent):
    """最小化: 跳过真实检索,直接验证 skip_matching 分支透传 _multi_round_pubmed 的新字段。"""
    def __init__(self):
        self.literature_mode = "pubmed_llm"
        self._annotation_cache = {}
        self._ANNOTATION_CACHE_MAX = 32
        async def _mrp(*a, **k):
            return {"report": "REVIEW", "references": [],
                    "focused_evidence": {"22": {"articles": [{"title": "p"}], "status": "strong", "tier": 1}},
                    "triage": {"rare_unresolved": [{"cluster_id": "22"}], "suspected_contamination": [],
                               "context_dependent_state": []}}
        self._multi_round_pubmed = _mrp
    def refresh_runtime_config_if_needed(self):
        pass


@pytest.mark.asyncio
async def test_annotate_clusters_carries_focused_and_triage():
    lit = _Lit()
    res = await lit.annotate_clusters(
        cluster_markers={"22": ["Siglech"]}, organism="mouse", tissue="liver",
        pi_initial_query="q", pi_evaluate_fn=lambda *a: None, skip_matching=True)
    assert res["status"] == "literature_only"
    assert res["focused_evidence"]["22"]["status"] == "strong"
    assert res["triage"]["rare_unresolved"][0]["cluster_id"] == "22"
