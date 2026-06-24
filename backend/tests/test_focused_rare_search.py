"""_focused_rare_search: rare 串行升级、污染并行两条取高、空 query 跳过。"""
import pytest
from backend.agents.literature_agent import LiteratureAgent


class _FakeLit(LiteratureAgent):
    def __init__(self):
        self.searched = []
        async def _search(query, max_results=10):
            self.searched.append(query)
            return [] if query == "" else [{"pmid": "1", "title": "t", "abstract": "a"}]
        self._pubmed_search = _search


@pytest.mark.asyncio
async def test_rare_tier1_strong_skips_tier2():
    fake = _FakeLit()
    async def judge(item, tissue, articles, bucket):
        return {"status": "strong", "judge": {}}
    fake._judge_focused_relevance = judge
    items = [{"cluster_id": "22", "candidate_celltype": "pDC", "synonyms": [], "top_markers": ["Siglech"]}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="rare_unresolved")
    assert out["22"]["status"] == "strong" and out["22"]["tier"] == 1
    assert all("liver" in q.lower() for q in fake.searched)   # 未发起 tier2(去 tissue)


@pytest.mark.asyncio
async def test_rare_tier1_weak_escalates_tier2():
    fake = _FakeLit()
    statuses = iter(["exhausted", "limited"])
    async def judge(item, tissue, articles, bucket):
        return {"status": next(statuses), "judge": {}}
    fake._judge_focused_relevance = judge
    items = [{"cluster_id": "19", "candidate_celltype": "cholangiocyte", "synonyms": [], "top_markers": ["Pkhd1"]}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="rare_unresolved")
    assert len(fake.searched) == 2 and out["19"]["status"] == "limited" and out["19"]["tier"] == 2


@pytest.mark.asyncio
async def test_contamination_runs_both_queries_and_takes_higher():
    fake = _FakeLit()
    seq = iter([("strong", 1), ("exhausted", 2)])
    async def judge(item, tissue, articles, bucket):
        s, _t = next(seq)
        return {"status": s, "judge": {}}
    fake._judge_focused_relevance = judge
    items = [{"cluster_id": "8", "candidate_celltype": None, "top_markers": ["Nrxn1", "Reln"]}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="suspected_contamination")
    assert len(fake.searched) == 2                 # 两条都查(不串行短路)
    assert out["8"]["status"] == "strong"          # 取相关性更高者


@pytest.mark.asyncio
async def test_pubmed_failure_returns_exhausted_not_raise():
    fake = _FakeLit()

    async def broken_search(query, max_results=10):
        raise RuntimeError("NCBI temporary failure")

    fake._pubmed_search = broken_search

    async def judge(item, tissue, articles, bucket):
        raise AssertionError("judge should not run when PubMed search fails")

    fake._judge_focused_relevance = judge

    items = [{"cluster_id": "22", "candidate_celltype": "pDC", "synonyms": [], "top_markers": ["Siglech"]}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="rare_unresolved")

    assert out["22"]["status"] == "exhausted"
    assert out["22"]["articles"] == []


@pytest.mark.asyncio
async def test_judge_failure_returns_exhausted_not_raise():
    fake = _FakeLit()

    async def judge(item, tissue, articles, bucket):
        raise RuntimeError("LLM judge failed")

    fake._judge_focused_relevance = judge

    items = [{"cluster_id": "19", "candidate_celltype": "cholangiocyte", "synonyms": [], "top_markers": ["Pkhd1"]}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="rare_unresolved")

    assert out["19"]["status"] == "exhausted"
    assert out["19"]["articles"] == []


@pytest.mark.asyncio
async def test_empty_query_skipped_as_exhausted():
    fake = _FakeLit()
    async def judge(item, tissue, articles, bucket):
        return {"status": "strong", "judge": {}}   # 不应被调用
    fake._judge_focused_relevance = judge
    items = [{"cluster_id": "3", "candidate_celltype": None, "synonyms": [], "top_markers": []}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="rare_unresolved")
    assert out["3"]["status"] == "exhausted" and out["3"]["articles"] == []
    assert fake.searched == []                      # 空 query 不发请求


@pytest.mark.asyncio
async def test_no_pubmed_results_marked_separately_from_runtime_failure():
    fake = _FakeLit()

    async def search(query, max_results=10):
        return []
    fake._pubmed_search = search

    items = [{"cluster_id": "7", "candidate_celltype": "pDC", "synonyms": [], "top_markers": ["Siglech"]}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="rare_unresolved")

    assert out["7"]["status"] == "exhausted"
    assert out["7"]["exhaustion_reason"] == "no_pubmed_results"
    assert out["7"].get("error") is None


@pytest.mark.asyncio
async def test_pubmed_failure_marked_as_search_failed():
    fake = _FakeLit()

    async def search(query, max_results=10):
        raise RuntimeError("NCBI unavailable")
    fake._pubmed_search = search

    items = [{"cluster_id": "7", "candidate_celltype": "pDC", "synonyms": [], "top_markers": ["Siglech"]}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="rare_unresolved")

    assert out["7"]["status"] == "exhausted"
    assert out["7"]["exhaustion_reason"] == "search_failed"
    assert "NCBI unavailable" in out["7"]["error"]


@pytest.mark.asyncio
async def test_judge_failure_marked_as_judge_failed():
    fake = _FakeLit()

    async def search(query, max_results=10):
        return [{"pmid": "1", "title": "p", "abstract": "Siglech"}]
    fake._pubmed_search = search

    async def judge(item, tissue, articles, bucket):
        raise RuntimeError("judge timeout")
    fake._judge_focused_relevance = judge

    items = [{"cluster_id": "7", "candidate_celltype": "pDC", "synonyms": [], "top_markers": ["Siglech"]}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="rare_unresolved")

    assert out["7"]["status"] == "exhausted"
    assert out["7"]["exhaustion_reason"] == "judge_failed"
    assert "judge timeout" in out["7"]["error"]


@pytest.mark.asyncio
async def test_judged_exhausted_marked_as_no_relevant_evidence():
    fake = _FakeLit()

    async def search(query, max_results=10):
        return [{"pmid": "1", "title": "p", "abstract": "generic marker paper"}]
    fake._pubmed_search = search

    async def judge(item, tissue, articles, bucket):
        return {"status": "exhausted", "judge": {"strong": 0, "moderate": 0, "contradictory": False}}
    fake._judge_focused_relevance = judge

    items = [{"cluster_id": "7", "candidate_celltype": "pDC", "synonyms": [], "top_markers": ["Siglech"]}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="rare_unresolved")

    assert out["7"]["status"] == "exhausted"
    assert out["7"]["exhaustion_reason"] == "no_relevant_evidence"
    assert out["7"].get("error") is None


@pytest.mark.asyncio
async def test_rare_tier2_failure_not_hidden_by_tier1_no_results():
    fake = _FakeLit()
    calls = {"n": 0}

    async def search(query, max_results=10):
        calls["n"] += 1
        if calls["n"] == 1:
            return []
        raise RuntimeError("tier2 network failure")
    fake._pubmed_search = search

    items = [{"cluster_id": "7", "candidate_celltype": "pDC", "synonyms": [], "top_markers": ["Siglech"]}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="rare_unresolved")

    assert calls["n"] == 2
    assert out["7"]["status"] == "exhausted"
    assert out["7"]["exhaustion_reason"] == "search_failed"
    assert "tier2 network failure" in out["7"]["error"]


@pytest.mark.asyncio
async def test_contamination_plausibility_failure_not_hidden_by_discovery_no_results():
    fake = _FakeLit()
    calls = {"n": 0}

    async def search(query, max_results=10):
        calls["n"] += 1
        if calls["n"] == 1:
            return []
        raise RuntimeError("plausibility endpoint failure")
    fake._pubmed_search = search

    items = [{"cluster_id": "8", "candidate_celltype": None, "top_markers": ["Nrxn1", "Reln"]}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="suspected_contamination")

    assert calls["n"] == 2
    assert out["8"]["status"] == "exhausted"
    assert out["8"]["exhaustion_reason"] == "search_failed"
    assert "plausibility endpoint failure" in out["8"]["error"]


@pytest.mark.asyncio
async def test_successful_later_tier_still_beats_runtime_failure():
    fake = _FakeLit()
    calls = {"n": 0}

    async def search(query, max_results=10):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("tier1 network failure")
        return [{"pmid": "2", "title": "pDC", "abstract": "Siglech Tcf4"}]
    fake._pubmed_search = search

    async def judge(item, tissue, articles, bucket):
        return {"status": "limited", "judge": {"strong": 1, "moderate": 1, "contradictory": False}}
    fake._judge_focused_relevance = judge

    items = [{"cluster_id": "7", "candidate_celltype": "pDC", "synonyms": [], "top_markers": ["Siglech"]}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="rare_unresolved")

    assert calls["n"] == 2
    assert out["7"]["status"] == "limited"
    assert out["7"]["exhaustion_reason"] is None
    assert out["7"]["error"] is None
