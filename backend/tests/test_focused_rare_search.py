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
async def test_multitier_choice_retains_raw_articles_from_every_executed_tier():
    fake = _FakeLit()
    calls = {"n": 0}

    async def search(_query, max_results=10):
        calls["n"] += 1
        return [{"pmid": str(calls["n"]), "title": f"tier-{calls['n']}", "abstract": "a"}]

    fake._pubmed_search = search
    statuses = iter(["strong", "exhausted"])

    async def judge(_item, _tissue, _articles, _bucket):
        return {"status": next(statuses), "judge": {}}

    fake._judge_focused_relevance = judge
    out = await fake._focused_rare_search(
        [{"cluster_id": "8", "candidate_celltype": None, "top_markers": ["GeneA"]}],
        "species A", "organ A", bucket="suspected_contamination",
    )

    assert out["8"]["status"] == "strong"
    assert [article["pmid"] for article in out["8"]["articles"]] == ["1"]
    assert {article["pmid"] for article in out["8"]["retained_articles"]} == {"1", "2"}


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
    assert out["19"]["articles"][0]["pmid"] == "1"
    assert out["19"]["retained_articles"][0]["pmid"] == "1"


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

    # Three attempts now: tier 1 returns nothing, tier 2 fails, and the cluster
    # is still without evidence so tier 3 tries once more without the marker
    # block. What this test is about is unchanged -- the tier-2 failure must
    # still be what gets recorded, not tier 1's empty result.
    assert calls["n"] == 3
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


# --- tier 3: the marker block stops being a precondition ---------------------
#
# Measured against live PubMed on one run's real cluster labels: at tier 1,
# 11 of 15 clusters retrieved nothing, and the production tier-2 relaxation
# (dropping tissue) recovered exactly one of them. The binding constraint was
# the mandatory marker conjunction, which turned even a clean label such as
# "Pericentral Hepatocyte" into a zero-result query. With tier 3 the same 15
# clusters all retrieved evidence, and 8 of them still resolved at tier 1, so
# markers remain a precision filter where they work.


def test_a_compound_label_is_normalised_before_it_becomes_a_phrase():
    from backend.agents.literature.shared import build_focused_query

    q = build_focused_query(
        {"candidate_celltype": "Hepatocyte (Stress/Inflammatory State)",
         "synonyms": [], "top_markers": ["Mx2"]},
        "Mus musculus", "liver", tier=1, bucket="rare_unresolved",
    )

    assert '"Hepatocyte (Stress/Inflammatory State)"' not in q
    assert '"Hepatocyte"[tiab]' in q


def test_a_parenthesised_acronym_survives_as_an_alternative():
    from backend.agents.literature.shared import build_focused_query

    q = build_focused_query(
        {"candidate_celltype": "Liver Sinusoidal Endothelial Cell (LSEC)",
         "synonyms": [], "top_markers": ["Stab2"]},
        "Mus musculus", "liver", tier=1, bucket="rare_unresolved",
    )

    assert '"Liver Sinusoidal Endothelial Cell"[tiab]' in q
    assert '"LSEC"[tiab]' in q


def test_tier3_drops_the_markers_and_keeps_the_tissue():
    from backend.agents.literature.shared import build_focused_query

    item = {"candidate_celltype": "Pericentral Hepatocyte", "synonyms": [],
            "top_markers": ["Cyp2d9", "Slco1a1"]}

    t1 = build_focused_query(item, "Mus musculus", "liver", tier=1, bucket="rare_unresolved")
    t2 = build_focused_query(item, "Mus musculus", "liver", tier=2, bucket="rare_unresolved")
    t3 = build_focused_query(item, "Mus musculus", "liver", tier=3, bucket="rare_unresolved")

    assert "Cyp2d9" in t1 and "liver" in t1
    assert "Cyp2d9" in t2 and "liver" not in t2      # tier 2 drops tissue only
    assert "Cyp2d9" not in t3 and "liver" in t3      # tier 3 drops markers only
    assert '"Pericentral Hepatocyte"[tiab]' in t3


def test_tier3_is_skipped_when_there_is_no_candidate_name():
    from backend.agents.literature.shared import build_focused_query

    # Markers are the only subject the query has; dropping them would search
    # for the tissue alone, which says nothing about this cluster.
    q = build_focused_query(
        {"candidate_celltype": None, "synonyms": [], "top_markers": ["Siglech"]},
        "Mus musculus", "liver", tier=3, bucket="rare_unresolved",
    )

    assert q == ""


@pytest.mark.asyncio
async def test_a_cluster_still_empty_after_tier2_gets_a_tier3_attempt():
    fake = _FakeLit()
    seen = []

    async def _search(query, max_results=10):
        seen.append(query)
        # Nothing matches while the marker block is present.
        return [] if "Siglech" in query else [{"pmid": "9", "title": "t", "abstract": "a"}]

    fake._pubmed_search = _search

    async def judge(item, tissue, articles, bucket):
        return {"status": "strong" if articles else "exhausted", "judge": {}}

    fake._judge_focused_relevance = judge

    items = [{"cluster_id": "7", "candidate_celltype": "pDC", "synonyms": [],
              "top_markers": ["Siglech"]}]
    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="rare_unresolved")

    assert len(seen) == 3
    assert "Siglech" not in seen[2]
    assert out["7"]["tier"] == 3
    assert out["7"]["status"] == "strong"


@pytest.mark.asyncio
async def test_tier3_is_not_reached_when_tier1_already_worked():
    fake = _FakeLit()

    async def judge(item, tissue, articles, bucket):
        return {"status": "strong", "judge": {}}

    fake._judge_focused_relevance = judge
    items = [{"cluster_id": "22", "candidate_celltype": "pDC", "synonyms": [],
              "top_markers": ["Siglech"]}]

    out = await fake._focused_rare_search(items, "mouse", "liver", bucket="rare_unresolved")

    assert len(fake.searched) == 1 and out["22"]["tier"] == 1
