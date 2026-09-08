"""Annotation literature search: refinement after triage, evidence caps, null candidates.

Each test here is red on the tree before the 2026-09-08 annotation-search fixes and
documents one observed defect from the biogenous-gpu runs of that date.
"""
import json

import pytest

from backend.agents.literature_agent import (
    LiteratureAgent,
    build_focused_query,
    status_from_judge,
)
from backend.agents.pi_agent import PIAgent

# New symbols are imported inside the tests that need them so the rest of this
# module still collects (and fails for its own reason) on a tree without them.


# ---------------------------------------------------------------------------
# Null-like candidate strings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [None, "", "null", "None", " NULL ", "n/a", "unknown", '"null"'])
def test_normalize_candidate_celltype_treats_null_tokens_as_absent(raw):
    from backend.agents.literature_agent import normalize_candidate_celltype
    assert normalize_candidate_celltype(raw) is None


def test_normalize_candidate_celltype_keeps_real_terms():
    from backend.agents.literature_agent import normalize_candidate_celltype
    assert normalize_candidate_celltype("  mesothelial cell ") == "mesothelial cell"


def test_focused_query_never_searches_for_the_word_null():
    item = {"candidate_celltype": "null", "synonyms": ["None"], "top_markers": ["GeneA"]}
    query = build_focused_query(item, "species A", "organ A", 1, "rare_unresolved")
    assert '"null"[tiab]' not in query
    assert '"None"[tiab]' not in query
    assert '"GeneA"[tiab]' in query


def test_focused_query_requires_candidate_and_markers_together():
    """Candidate name and gene symbols were one OR group, so a tumour-antigen gene
    symbol alone pulled unrelated papers. They must now be separate AND groups."""
    item = {"candidate_celltype": "candidate cell type A", "synonyms": [], "top_markers": ["GeneA"]}
    query = build_focused_query(item, "species A", "organ A", 1, "rare_unresolved")
    groups = [g.strip() for g in query.split(" AND ")]
    candidate_groups = [g for g in groups if "candidate cell type A" in g]
    marker_groups = [g for g in groups if '"GeneA"[tiab]' in g]
    assert candidate_groups and marker_groups
    assert candidate_groups[0] != marker_groups[0]


# ---------------------------------------------------------------------------
# Judge status: strong hits are not erased by a contradictory flag
# ---------------------------------------------------------------------------

def test_strong_hits_win_over_contradictory_flag():
    assert status_from_judge({"strong": 2, "moderate": 2, "contradictory": True}) == "strong"
    assert status_from_judge({"strong": 1, "moderate": 1, "contradictory": True}) == "limited"
    assert status_from_judge({"strong": 0, "moderate": 3, "contradictory": True}) == "contradictory"
    assert status_from_judge({"strong": 1, "moderate": 0, "contradictory": True}) == "contradictory"


class _JudgeLit(LiteratureAgent):
    def __init__(self, response):
        self.captured = {}

        async def _chat(message, temperature=None, **kwargs):
            self.captured["prompt"] = message
            return response
        self.chat_without_history = _chat


@pytest.mark.asyncio
async def test_judge_prompt_does_not_name_a_null_candidate():
    lit = _JudgeLit(json.dumps({"strong": 0, "moderate": 0, "contradictory": False}))
    await lit._judge_focused_relevance(
        item={"candidate_celltype": "null", "top_markers": ["GeneA"]},
        tissue="organ A",
        articles=[{"title": "t", "abstract": "a" * 500}],
        bucket="rare_unresolved",
    )
    prompt = lit.captured["prompt"]
    assert '"null"' not in prompt
    assert "marker-defined population" in prompt
    # abstracts are no longer clipped to 200 characters
    assert "a" * 300 in prompt


# ---------------------------------------------------------------------------
# PI evaluation post-processing and article window
# ---------------------------------------------------------------------------

class _FakePI(PIAgent):
    def __init__(self, response):
        self.captured = {}

        async def _chat(message, temperature=None, **kwargs):
            self.captured["prompt"] = message
            return response
        self.chat_without_history = _chat

    def refresh_runtime_config_if_needed(self):
        pass


@pytest.mark.asyncio
async def test_evaluation_normalizes_string_null_candidates():
    payload = {
        "sufficient": False, "covered": ["0"],
        "rare_unresolved": [{"cluster_id": "1", "candidate_celltype": "null",
                              "synonyms": ["null", "Alias A"], "top_markers": ["GeneA"]}],
        "suspected_contamination": [{"cluster_id": "2", "candidate_celltype": "None",
                                      "top_markers": ["GeneB"], "reason": "r"}],
        "context_dependent_state": [{"cluster_id": "3", "parent_celltype": "null",
                                      "top_markers": ["GeneC"]}],
    }
    pi = _FakePI(json.dumps(payload))
    out = await pi.evaluate_annotation_search_results(
        cluster_markers={"0": ["a"], "1": ["b"], "2": ["c"], "3": ["d"]},
        organism="species A", tissue="organ A", articles=[], search_history=[],
    )
    assert out["rare_unresolved"][0]["candidate_celltype"] is None
    assert out["rare_unresolved"][0]["synonyms"] == ["Alias A"]
    assert out["suspected_contamination"][0]["candidate_celltype"] is None
    assert out["context_dependent_state"][0]["parent_celltype"] is None


@pytest.mark.asyncio
async def test_evaluation_shows_latest_round_articles_first():
    """Round 2+ used to show the first 15 cumulative articles, so the PI never saw
    what the refined query retrieved."""
    pi = _FakePI(json.dumps({"sufficient": True}))
    articles = [{"title": f"old-{i}", "abstract": "x", "year": "2020"} for i in range(40)]
    articles += [{"title": f"new-{i}", "abstract": "y", "year": "2021"} for i in range(5)]
    history = [
        {"round": 1, "query": "q1", "new_results": 40, "total_results": 40},
        {"round": 2, "query": "q2", "new_results": 5, "total_results": 45},
    ]
    await pi.evaluate_annotation_search_results(
        cluster_markers={"0": ["a"]}, organism="species A", tissue="organ A",
        articles=articles, search_history=history,
    )
    prompt = pi.captured["prompt"]
    for i in range(5):
        assert f"new-{i}" in prompt
    assert prompt.index("new-0") < prompt.index("old-0")


@pytest.mark.asyncio
async def test_evaluation_abstract_window_is_wider_than_200_chars():
    pi = _FakePI(json.dumps({"sufficient": True}))
    articles = [{"title": "t", "abstract": "b" * 400, "year": "2020"}]
    await pi.evaluate_annotation_search_results(
        cluster_markers={"0": ["a"]}, organism="species A", tissue="organ A",
        articles=articles, search_history=[{"round": 1, "query": "q", "new_results": 1}],
    )
    assert "b" * 250 in pi.captured["prompt"]


@pytest.mark.asyncio
async def test_search_query_prompt_forbids_generic_or_terms():
    pi = _FakePI("q")
    await pi.generate_annotation_search_query(
        cluster_markers={"0": ["GeneA"]}, organism="species A", tissue="organ A",
    )
    prompt = pi.captured["prompt"].lower()
    assert "do not" in prompt and "standalone" in prompt
    assert "20-200" not in prompt


# ---------------------------------------------------------------------------
# PI matching: the whole review reaches the prompt; null candidates render safely
# ---------------------------------------------------------------------------

class _MatchPI(PIAgent):
    def __init__(self):
        self.captured = {}

        async def _chat(message, temperature=None, **kwargs):
            self.captured["prompt"] = message
            return json.dumps({"annotations": {"0": {"cell_type": "x"}}, "tissue_summary": ""})
        self.chat_without_history = _chat

    def refresh_runtime_config_if_needed(self):
        pass


@pytest.mark.asyncio
async def test_matching_prompt_keeps_review_beyond_12000_chars():
    review = ("section " * 2000)[:20000] + "SENTINEL_LATE_SECTION"
    pi = _MatchPI()
    await pi.annotate_clusters_from_literature(
        cluster_markers={"0": ["GeneA"]}, organism="species A", tissue="organ A",
        literature_report=review,
    )
    assert "SENTINEL_LATE_SECTION" in pi.captured["prompt"]


@pytest.mark.asyncio
async def test_matching_hint_renders_null_candidate_as_marker_defined():
    pi = _MatchPI()
    triage = {"rare_unresolved": [{"cluster_id": "0", "candidate_celltype": "null"}],
              "suspected_contamination": [], "context_dependent_state": []}
    focused = {"0": {"articles": [{"title": "p", "abstract": "a"}], "status": "strong", "tier": 1}}
    await pi.annotate_clusters_from_literature(
        cluster_markers={"0": ["GeneA"]}, organism="species A", tissue="organ A",
        literature_report="R", triage=triage, focused_evidence=focused,
    )
    prompt = pi.captured["prompt"]
    assert "candidate: null" not in prompt
    assert "candidate: marker-defined population" in prompt


class _MatchLit(LiteratureAgent):
    def __init__(self, review):
        self.captured = {}
        self.literature_mode = "pubmed_llm"
        self._annotation_cache = {}
        self._ANNOTATION_CACHE_MAX = 10

        async def _research(topic, state, pubmed_max_results=None, research_brief=None):
            return {"report": review, "references": []}
        self._do_research = _research

        async def _chat(message, temperature=None, **kwargs):
            self.captured["prompt"] = message
            return json.dumps({"annotations": {"0": {"cell_type": "x"}}, "tissue_summary": ""})
        self.chat_without_history = _chat

    def refresh_runtime_config_if_needed(self):
        pass


@pytest.mark.asyncio
async def test_literature_agent_matching_prompt_keeps_review_beyond_12000_chars():
    review = ("section " * 2000)[:20000] + "SENTINEL_LATE_SECTION"
    lit = _MatchLit(review)
    await lit.annotate_clusters(cluster_markers={"0": ["GeneA"]}, organism="species A", tissue="organ A")
    assert "SENTINEL_LATE_SECTION" in lit.captured["prompt"]


# ---------------------------------------------------------------------------
# Search loop: triage_complete no longer overrides sufficient=false
# ---------------------------------------------------------------------------

class _LoopLit(LiteratureAgent):
    def __init__(self, evaluations):
        self._evals = list(evaluations)
        self._eval_i = 0
        self.searched = []
        self.synth_articles = None
        self.synth_max_articles = None
        self.max_search_rounds = 40

        async def _vd(query, topic, max_results=20, **kwargs):
            self.searched.append(query)
            return query, [{"pmid": f"r1-{i}", "title": "t", "abstract": "a"} for i in range(30)]
        self._validate_and_degrade_query = _vd

        async def _search(query, max_results=20):
            self.searched.append(query)
            return [{"pmid": f"{query}-{i}", "title": "t", "abstract": "a"} for i in range(3)]
        self._pubmed_search = _search

        async def _synth(topic, articles, evaluation=None, max_articles=None):
            self.synth_articles = list(articles)
            self.synth_max_articles = max_articles
            return "REPORT"
        self._synthesize_pubmed_results = _synth

        async def _final(topic, report, refs, state, n):
            return {"report": report, "references": refs}
        self._finalize_result = _final

        async def _focused(items, organism, tissue, bucket="rare_unresolved", deadline=None):
            return {}
        self._focused_rare_search = _focused

    async def _eval_fn(self, articles, history):
        e = self._evals[min(self._eval_i, len(self._evals) - 1)]
        self._eval_i += 1
        return dict(e)


def _triage(sufficient, refined):
    return {"sufficient": sufficient, "refined_query": refined, "triage_complete": True,
            "triage_low_confidence": False, "covered": ["0"], "rare_unresolved": [],
            "suspected_contamination": [], "context_dependent_state": []}


@pytest.mark.asyncio
async def test_insufficient_verdict_runs_the_refined_query_after_triage():
    lit = _LoopLit([_triage(False, "q2"), _triage(True, "q3")])
    res = await lit._multi_round_pubmed("t", state=None, pubmed_max_results=50,
        initial_query="q1", evaluate_fn=lit._eval_fn, organism="species A", tissue="organ A")
    assert lit.searched[:2] == ["q1", "q2"]
    assert res["search_meta"]["rounds"] == 2
    assert res["search_meta"]["stop_reason"] == "triage_complete"


@pytest.mark.asyncio
async def test_sufficient_verdict_still_stops_after_first_round():
    """Control: passes before and after the fix; guards the sufficient=true exit."""
    lit = _LoopLit([_triage(True, "q2")])
    res = await lit._multi_round_pubmed("t", state=None, pubmed_max_results=50,
        initial_query="q1", evaluate_fn=lit._eval_fn, organism="species A", tissue="organ A")
    assert lit.searched == ["q1"]
    assert res["search_meta"]["rounds"] == 1


@pytest.mark.asyncio
async def test_post_triage_refinement_is_bounded():
    evals = [_triage(False, f"q{i}") for i in range(2, 20)]
    lit = _LoopLit(evals)
    lit.post_triage_refinement_rounds = 2
    res = await lit._multi_round_pubmed("t", state=None, pubmed_max_results=50,
        initial_query="q1", evaluate_fn=lit._eval_fn, organism="species A", tissue="organ A")
    assert res["search_meta"]["rounds"] == 3
    assert res["search_meta"]["stop_reason"] == "triage_complete"


@pytest.mark.asyncio
async def test_annotation_synthesis_reads_more_than_twenty_articles():
    lit = _LoopLit([_triage(True, "q2")])
    await lit._multi_round_pubmed("t", state=None, pubmed_max_results=50,
        initial_query="q1", evaluate_fn=lit._eval_fn, organism="species A", tissue="organ A")
    assert lit.synth_max_articles is not None and lit.synth_max_articles > 20


@pytest.mark.asyncio
async def test_synthesizer_honours_max_articles_argument():
    class _Synth(LiteratureAgent):
        def __init__(self):
            self.captured = {}

            async def _chat_with_fallback(prompt, temperature=0.3):
                self.captured["prompt"] = prompt
                return "REVIEW", False
            self._chat_with_fallback = _chat_with_fallback

    lit = _Synth()
    articles = [{"pmid": str(i), "title": f"paper-{i}", "abstract": "a", "year": "2020"} for i in range(30)]
    await lit._synthesize_pubmed_results("topic", articles, max_articles=25)
    prompt = lit.captured["prompt"]
    assert "paper-24" in prompt and "paper-25" not in prompt
    await lit._synthesize_pubmed_results("topic", articles)
    assert "paper-19" in lit.captured["prompt"] and "paper-20" not in lit.captured["prompt"]
