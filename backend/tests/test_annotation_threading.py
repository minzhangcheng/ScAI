"""串接: evaluate 闭包传 state;annotate_clusters skip_matching 返回带出 focused_evidence/triage。"""
import pytest
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.agents.literature_agent import LiteratureAgent
from backend.workflow import annotation
from backend.workflow.annotation_parts import callback


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


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    {"source_failure": {"mode": "pubmed_llm", "reason": "ValueError: invalid XML"}},
    {"source_processing_error": "RuntimeError: extraction failed"},
    {"search_meta": {"evaluation_failed": True}},
])
async def test_annotation_matching_keeps_research_failure_visible_and_retryable(failure):
    lit = _Lit()
    lit._multi_round_pubmed = AsyncMock(return_value={
        "report": "retained evidence" * 2000, "status": "partial",
        "source_provenance": "pubmed_retrieved_partial", **failure,
    })
    lit.chat_without_history = AsyncMock(return_value=json.dumps({
        "annotations": {"22": {"cell_type": "Candidate type"}},
    }))
    arguments = {
        "cluster_markers": {"22": ["MarkerA"]}, "organism": "species A",
        "tissue": "organ A", "pi_initial_query": "query",
        "pi_evaluate_fn": AsyncMock(),
    }

    for _ in range(2):
        result = await lit.annotate_clusters(**arguments)
        assert result["annotations"]["22"]["cell_type"] == "Candidate type"
        assert result["research_status"] == "partial"
        assert result["matching_status"] == "completed"
        assert result["status"] == "partial"
        for key, value in failure.items():
            assert result[key] == value
            assert result["triage"]["research_provenance"][key] == value
    assert lit._multi_round_pubmed.await_count == 2
    assert not lit._annotation_cache

    result = await lit.annotate_clusters(**arguments, skip_matching=True)
    assert result["status"] == "literature_only"
    workflow = SimpleNamespace(
        _send_message=AsyncMock(),
        pi_agent=SimpleNamespace(annotate_clusters_from_literature=AsyncMock(return_value={
            "annotations": {"22": {"cell_type": "PI chosen type"}},
        })),
    )
    context = {"state": None, "n_clusters": 1, "result": result, "annotation_brief": None}
    callback.bind_facade(annotation)
    await callback.match_annotations(
        workflow, context, arguments["cluster_markers"], "organ A", "species A",
    )
    assert context["result"]["status"] == "partial"
    assert context["result"]["research_status"] == "partial"
    assert context["result"]["matching_status"] == "completed"
    assert context["annotations"]["22"]["cell_type"] == "PI chosen type"
    call = workflow.pi_agent.annotate_clusters_from_literature.call_args.kwargs
    for key in failure:
        assert key in call["literature_report"][:2000]
