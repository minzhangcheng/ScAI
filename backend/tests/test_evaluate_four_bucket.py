"""evaluate_annotation_search_results: 四分类 prompt + 确定性 triage_complete/守卫 + candidate=null."""
import json
import pytest
from backend.agents.pi_agent import PIAgent


class _FakePI(PIAgent):
    def __init__(self, response: str):
        self.captured = {}
        async def _chat(message, temperature=None, **kwargs):
            self.captured["prompt"] = message
            return response
        self.chat_without_history = _chat


@pytest.mark.asyncio
async def test_prompt_requests_four_buckets():
    pi = _FakePI(json.dumps({"sufficient": True}))
    await pi.evaluate_annotation_search_results(
        cluster_markers={"0": ["Cd3d"]}, organism="mouse", tissue="liver",
        articles=[], search_history=[],
    )
    p = pi.captured["prompt"]
    for bucket in ("covered", "rare_unresolved", "suspected_contamination", "context_dependent_state"):
        assert bucket in p


@pytest.mark.asyncio
async def test_state_context_injected():
    pi = _FakePI(json.dumps({"sufficient": True}))
    class _S:
        initial_instruction = "Ctr1/ZnT1 DKO mouse liver copper study"
        def build_experimental_design_summary(self):
            return "Experimental Design: 2 group(s) Con/DKO"
    await pi.evaluate_annotation_search_results(
        cluster_markers={"0": ["Cd3d"]}, organism="mouse", tissue="liver",
        articles=[], search_history=[], state=_S(),
    )
    p = pi.captured["prompt"]
    assert "DKO mouse liver copper" in p
    assert "Con/DKO" in p


@pytest.mark.asyncio
async def test_deterministic_triage_fields():
    payload = {
        "sufficient": False, "refined_query": "q", "missing_aspects": [],
        "covered": ["0"],
        "rare_unresolved": [{"cluster_id": "1", "candidate_celltype": "pDC",
                              "candidate_celltype_confidence": "high", "synonyms": [],
                              "top_markers": ["Siglech"], "triage_confidence": "high", "evidence_need": "x"}],
        "suspected_contamination": [{"cluster_id": "2", "candidate_celltype": None,
                                      "top_markers": ["Nrxn1"], "triage_confidence": "medium",
                                      "reason": "neuronal", "evidence_need": "y"}],
        "context_dependent_state": [],
    }
    pi = _FakePI(json.dumps(payload))
    out = await pi.evaluate_annotation_search_results(
        cluster_markers={"0": ["a"], "1": ["b"], "2": ["c"]},  # 3 clusters, all triaged
        organism="mouse", tissue="liver", articles=[], search_history=[],
    )
    assert out["suspected_contamination"][0]["candidate_celltype"] is None
    assert out["untriaged_cluster_ids"] == []
    assert out["triage_complete"] is True
    # 非 covered 桶 = 2/3 = 66% > 40% → low confidence
    assert out["triage_low_confidence"] is True


@pytest.mark.asyncio
async def test_untriaged_detected_when_missing():
    payload = {"sufficient": False, "covered": ["0"], "rare_unresolved": [],
               "suspected_contamination": [], "context_dependent_state": []}
    pi = _FakePI(json.dumps(payload))
    out = await pi.evaluate_annotation_search_results(
        cluster_markers={"0": ["a"], "5": ["b"]}, organism="mouse", tissue="liver",
        articles=[], search_history=[],
    )
    assert out["untriaged_cluster_ids"] == ["5"]
    assert out["triage_complete"] is False


@pytest.mark.asyncio
async def test_hallucinated_cluster_id_filtered():
    # LLM 幻觉 "999"(不存在)+ 重复归桶,不应虚增守卫分子
    payload = {"sufficient": False, "covered": ["0"],
               "rare_unresolved": [{"cluster_id": "999", "candidate_celltype": "x",
                                     "synonyms": [], "top_markers": ["A"]}],
               "suspected_contamination": [], "context_dependent_state": []}
    pi = _FakePI(json.dumps(payload))
    out = await pi.evaluate_annotation_search_results(
        cluster_markers={"0": ["a"], "1": ["b"]}, organism="mouse", tissue="liver",
        articles=[], search_history=[],
    )
    # 999 被过滤: 非 covered 桶有效数=0 → low_confidence False;1 未归桶
    assert out["triage_low_confidence"] is False
    assert out["untriaged_cluster_ids"] == ["1"]
    assert out["rare_unresolved"] == []          # 幻觉项从桶本身移除(防 focused 浪费查询)


@pytest.mark.asyncio
async def test_missing_buckets_default_empty():
    pi = _FakePI(json.dumps({"sufficient": True}))
    out = await pi.evaluate_annotation_search_results(
        cluster_markers={"0": ["Cd3d"]}, organism="mouse", tissue="liver",
        articles=[], search_history=[],
    )
    for k in ("covered", "rare_unresolved", "suspected_contamination", "context_dependent_state"):
        assert out[k] == []


@pytest.mark.asyncio
async def test_duplicate_cluster_keeps_covered_only():
    payload = {
        "sufficient": False,
        "covered": ["0"],
        "rare_unresolved": [{"cluster_id": "0", "candidate_celltype": "pDC",
                              "candidate_celltype_confidence": "low", "synonyms": [],
                              "top_markers": ["Siglech"], "triage_confidence": "low",
                              "evidence_need": "duplicate should be removed"}],
        "suspected_contamination": [],
        "context_dependent_state": [],
    }
    pi = _FakePI(json.dumps(payload))
    out = await pi.evaluate_annotation_search_results(
        cluster_markers={"0": ["Siglech"]},
        organism="mouse",
        tissue="liver",
        articles=[],
        search_history=[],
    )
    assert out["covered"] == ["0"]
    assert out["rare_unresolved"] == []
    assert out["triage_complete"] is True


@pytest.mark.asyncio
async def test_duplicate_noncovered_bucket_priority():
    payload = {
        "sufficient": False,
        "covered": [],
        "rare_unresolved": [{"cluster_id": "8", "candidate_celltype": "neuronal",
                              "candidate_celltype_confidence": "low", "synonyms": [],
                              "top_markers": ["Nrxn1"], "triage_confidence": "low",
                              "evidence_need": "duplicate rare"}],
        "suspected_contamination": [{"cluster_id": "8", "candidate_celltype": None,
                                      "top_markers": ["Nrxn1"], "triage_confidence": "medium",
                                      "reason": "neuronal markers in liver",
                                      "evidence_need": "plausibility check"}],
        "context_dependent_state": [],
    }
    pi = _FakePI(json.dumps(payload))
    out = await pi.evaluate_annotation_search_results(
        cluster_markers={"8": ["Nrxn1"]},
        organism="mouse",
        tissue="liver",
        articles=[],
        search_history=[],
    )
    assert out["rare_unresolved"] == []
    assert out["suspected_contamination"][0]["cluster_id"] == "8"
    assert out["triage_complete"] is True


@pytest.mark.asyncio
async def test_context_state_wins_over_other_noncovered_buckets():
    payload = {
        "sufficient": False,
        "covered": [],
        "rare_unresolved": [{"cluster_id": "18", "candidate_celltype": "hepatocyte",
                              "candidate_celltype_confidence": "medium", "synonyms": [],
                              "top_markers": ["Mt1"], "triage_confidence": "medium",
                              "evidence_need": "duplicate rare"}],
        "suspected_contamination": [{"cluster_id": "18", "candidate_celltype": None,
                                      "top_markers": ["Mt1"], "triage_confidence": "low",
                                      "reason": "duplicate suspected",
                                      "evidence_need": "duplicate suspected"}],
        "context_dependent_state": [{"cluster_id": "18", "parent_celltype": "hepatocyte",
                                      "context": "DKO stress", "top_markers": ["Mt1"],
                                      "triage_confidence": "medium",
                                      "state_evidence_note": "state variant"}],
    }
    pi = _FakePI(json.dumps(payload))
    out = await pi.evaluate_annotation_search_results(
        cluster_markers={"18": ["Mt1"]},
        organism="mouse",
        tissue="liver",
        articles=[],
        search_history=[],
    )
    assert out["rare_unresolved"] == []
    assert out["suspected_contamination"] == []
    assert out["context_dependent_state"][0]["cluster_id"] == "18"
    assert out["triage_complete"] is True
