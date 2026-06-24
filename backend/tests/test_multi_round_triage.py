"""_multi_round_pubmed: 确定性语义终止、3轮0新增早停、focused 并入、单次合成、早返回保留。"""
import pytest
from backend.agents.literature_agent import LiteratureAgent


class _Lit(LiteratureAgent):
    def __init__(self, evaluations, first_articles=1):
        self._evals = list(evaluations); self._eval_i = 0
        self.synth_calls = 0; self.focused_called = []
        self.max_search_rounds = 40
        self._first = first_articles

        async def _vd(query, topic, max_results=20):
            arts = [{"pmid": "1", "title": "t", "abstract": "a"}] if self._first else []
            return query, arts
        self._validate_and_degrade_query = _vd

        async def _search(query, max_results=20):
            return []
        self._pubmed_search = _search

        async def _synth(topic, articles):
            self.synth_calls += 1
            return "REPORT"
        self._synthesize_pubmed_results = _synth

        async def _final(topic, report, refs, state, n):
            return {"report": report, "references": refs}
        self._finalize_result = _final

        async def _quick(topic, state):
            return {"report": "QUICK", "references": []}
        self._quick_search = _quick

        async def _focused(items, organism, tissue, bucket="rare_unresolved"):
            self.focused_called.append((bucket, [i["cluster_id"] for i in items]))
            return {i["cluster_id"]: {"articles": [{"pmid": "9", "title": "F", "abstract": "b"}],
                                       "status": "strong", "tier": 1} for i in items}
        self._focused_rare_search = _focused

    async def _eval_fn(self, articles, history):
        e = self._evals[min(self._eval_i, len(self._evals) - 1)]; self._eval_i += 1
        return e


@pytest.mark.asyncio
async def test_semantic_termination_focused_and_single_synthesis():
    # triage_complete=True 且 low_confidence=False → 确定性终止
    ev = {"sufficient": False, "refined_query": "q2", "triage_complete": True,
          "triage_low_confidence": False, "covered": [],
          "rare_unresolved": [{"cluster_id": "22", "candidate_celltype": "pDC",
                                "synonyms": [], "top_markers": ["Siglech"]}],
          "suspected_contamination": [], "context_dependent_state": []}
    lit = _Lit([ev])
    res = await lit._multi_round_pubmed("t", state=None, pubmed_max_results=50,
        initial_query="q1", evaluate_fn=lit._eval_fn, organism="mouse", tissue="liver")
    assert ("rare_unresolved", ["22"]) in lit.focused_called
    assert lit.synth_calls == 1
    assert res["focused_evidence"]["22"]["status"] == "strong"
    assert res["triage"]["triage_complete"] is True


@pytest.mark.asyncio
async def test_low_confidence_blocks_semantic_termination():
    # triage_complete=True 但 low_confidence=True → 不终止,靠 sufficient/0新增
    ev = {"sufficient": False, "refined_query": "q", "triage_complete": True,
          "triage_low_confidence": True, "covered": [],
          "rare_unresolved": [{"cluster_id": "1", "candidate_celltype": "x",
                                "synonyms": [], "top_markers": ["A"]}],
          "suspected_contamination": [], "context_dependent_state": []}
    lit = _Lit([ev])
    await lit._multi_round_pubmed("t", state=None, pubmed_max_results=50,
        initial_query="q1", evaluate_fn=lit._eval_fn, organism="mouse", tissue="liver")
    # 未在第1轮语义终止 → 走到 3轮0新增早停,evaluate 被多次调用
    assert lit._eval_i >= 2


@pytest.mark.asyncio
async def test_focused_articles_prioritized_for_synthesis_when_main_pool_large():
    class _LargeMainPool(LiteratureAgent):
        def __init__(self):
            self.max_search_rounds = 40
            self.synth_first_pmids = []
            self.focused_called = []

            async def _vd(query, topic, max_results=20):
                articles = [
                    {"pmid": str(i), "title": f"main-{i}", "abstract": "main"}
                    for i in range(1, 26)
                ]
                return query, articles

            self._validate_and_degrade_query = _vd

            async def _search(query, max_results=20):
                return []

            self._pubmed_search = _search

            async def _synth(topic, articles):
                self.synth_first_pmids = [a["pmid"] for a in articles[:20]]
                return "REPORT"

            self._synthesize_pubmed_results = _synth

            async def _final(topic, report, refs, state, n):
                return {"report": report, "references": refs}

            self._finalize_result = _final

            async def _focused(items, organism, tissue, bucket="rare_unresolved"):
                self.focused_called.append(bucket)
                return {
                    "22": {
                        "articles": [{"pmid": "F22", "title": "focused-pdc", "abstract": "focused evidence"}],
                        "status": "strong",
                        "tier": 1,
                    }
                }

            self._focused_rare_search = _focused

        async def _eval_fn(self, articles, history):
            return {
                "sufficient": False,
                "triage_complete": True,
                "triage_low_confidence": False,
                "covered": [],
                "rare_unresolved": [{"cluster_id": "22", "candidate_celltype": "pDC", "synonyms": [], "top_markers": ["Siglech"]}],
                "suspected_contamination": [],
                "context_dependent_state": [],
            }

    lit = _LargeMainPool()
    res = await lit._multi_round_pubmed(
        "Cell types in mouse liver",
        state=None,
        pubmed_max_results=50,
        initial_query="q1",
        evaluate_fn=lit._eval_fn,
        organism="mouse",
        tissue="liver",
    )

    assert "rare_unresolved" in lit.focused_called
    assert lit.synth_first_pmids[0] == "F22"
    assert any(ref["url"].endswith("/F22/") for ref in res["references"])


@pytest.mark.asyncio
async def test_no_articles_returns_quick_search_without_focused():
    lit = _Lit([{"sufficient": True}], first_articles=0)
    res = await lit._multi_round_pubmed("t", state=None, pubmed_max_results=50,
        initial_query="q1", evaluate_fn=lit._eval_fn, organism="mouse", tissue="liver")
    assert res["report"] == "QUICK"           # 早返回保留
    assert lit.focused_called == []           # 无文章不触发 focused
    assert lit.synth_calls == 0
