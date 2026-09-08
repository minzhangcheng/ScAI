"""Atlas-first seeding, per-cluster targeted retrieval, per-cluster evidence in the matcher.

Companion to test_annotation_search_refinement.py. Every test here except the ones
whose docstring says "control" is red on the tree before the 2026-09-08 annotation
search redesign, and each documents one observed defect from the biogenous-gpu runs:
atlas-level studies were never retrieved, only two of four triage buckets received any
targeted search, and the matcher saw a pooled review instead of evidence tied to a cluster.
"""
import json

import pytest

from backend.agents.literature_agent import LiteratureAgent
from backend.agents.pi_agent import PIAgent

# Symbols added by the redesign are imported inside the tests that need them, so the
# rest of this module still collects on a tree without them.


# ---------------------------------------------------------------------------
# 1. Atlas-first: a deterministic seed query
# ---------------------------------------------------------------------------

def test_atlas_query_anchors_on_organism_tissue_design_and_cell_inventory():
    from backend.agents.literature_agent import build_atlas_query
    query = build_atlas_query("species A", "organ A")
    assert "species A" in query and "organ A" in query
    lowered = query.lower()
    assert "single-cell" in lowered
    assert "atlas" in lowered
    assert len(query.split(" AND ")) >= 4


def test_atlas_query_carries_no_domain_specific_vocabulary():
    """The template must name only the dataset's own organism and tissue."""
    from backend.agents.literature_agent import build_atlas_query
    a = build_atlas_query("species A", "organ A")
    b = build_atlas_query("species B", "organ B")
    assert a.replace("species A", "species B").replace("organ A", "organ B") == b


def test_atlas_query_leaves_the_organism_untagged_for_pubmed_term_mapping():
    """A field-tagged organism phrase gets no species expansion: measured against
    live PubMed on 2026-09-08, `"Mus musculus"[all]` returned 3 hits where the bare
    term returned 418. Reading synonyms off PubMed also avoids a built-in table."""
    from backend.agents.literature_agent import build_atlas_query
    query = build_atlas_query("species A", "organ A")
    assert "species A" in query
    assert '"species A"[all]' not in query
    assert '"species A"[tiab]' not in query


def test_atlas_query_tier1_anchors_the_tissue_in_the_title_and_tier2_relaxes_it():
    from backend.agents.literature_agent import build_atlas_query
    tier1 = build_atlas_query("species A", "organ A", tier=1)
    tier2 = build_atlas_query("species A", "organ A", tier=2)
    assert '"organ A"[ti]' in tier1
    assert '"organ A"[ti]' not in tier2
    assert '"organ A"[tiab]' in tier2 and '"organ A"[MeSH Terms]' in tier2


def test_atlas_scope_excludes_the_broad_wording_that_matches_any_subject():
    """The broad forms returned 418 live hits with 10/20 on topic; the narrow set
    returned 41 with 15/20. Keeping them out is the whole point of the seed."""
    from backend.agents.literature_agent import build_atlas_query
    query = build_atlas_query("species A", "organ A")
    assert '"heterogeneity"[tiab]' not in query
    assert '"cell type"[tiab]' not in query
    assert '"landscape"[tiab]' not in query
    assert '"atlas"[tiab]' in query


def test_atlas_query_without_tissue_still_anchors_on_organism():
    from backend.agents.literature_agent import build_atlas_query
    query = build_atlas_query("species A", "")
    assert "species A" in query and "atlas" in query.lower()


def test_atlas_query_empty_without_organism_and_tissue():
    from backend.agents.literature_agent import build_atlas_query
    assert build_atlas_query("", "") == ""


class _SeedLit(LiteratureAgent):
    def __init__(self, seed_results=None):
        self.searched = []
        self.eval_articles = []
        self.max_search_rounds = 40
        self._seed_results = seed_results or {}

        async def _vd(query, topic, max_results=20, **kwargs):
            self.searched.append(("primary", query))
            return query, [{"pmid": "p1", "title": "primary", "abstract": "a"}]
        self._validate_and_degrade_query = _vd

        async def _search(query, max_results=20):
            self.searched.append(("search", query))
            outcome = self._seed_results.get(query, [])
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        self._pubmed_search = _search

        async def _synth(topic, articles, evaluation=None, max_articles=None):
            return "REPORT"
        self._synthesize_pubmed_results = _synth

        async def _final(topic, report, refs, state, n):
            return {"report": report, "references": refs}
        self._finalize_result = _final

    async def _eval_fn(self, articles, history):
        self.eval_articles.append([a["pmid"] for a in articles])
        return {"sufficient": True, "triage_complete": True, "triage_low_confidence": False,
                "covered": [], "rare_unresolved": [], "suspected_contamination": [],
                "context_dependent_state": []}

    async def _run(self, **kwargs):
        return await self._multi_round_pubmed(
            "t", state=None, pubmed_max_results=50, initial_query="q1",
            evaluate_fn=self._eval_fn, organism="species A", tissue="organ A", **kwargs)


@pytest.mark.asyncio
async def test_seed_query_articles_join_the_first_round_ahead_of_the_primary_query():
    lit = _SeedLit({"ATLAS": [{"pmid": "atlas-1", "title": "atlas", "abstract": "b"}]})
    res = await lit._run(seed_queries=["ATLAS"])

    assert ("search", "ATLAS") in lit.searched
    assert lit.eval_articles and "atlas-1" in lit.eval_articles[0]
    pmids = [ref["url"].rstrip("/").split("/")[-1] for ref in res["references"]]
    assert pmids.index("atlas-1") < pmids.index("p1")
    assert res["search_meta"]["seed_queries"] == 1


@pytest.mark.asyncio
async def test_seed_query_failure_is_recorded_and_does_not_abort_the_search():
    lit = _SeedLit({"ATLAS": RuntimeError("NCBI down")})
    res = await lit._run(seed_queries=["ATLAS"])

    assert res["search_meta"]["stop_reason"] != "pubmed_error"
    assert any(
        attempt.get("kind") == "seed" and attempt.get("error")
        for attempt in res["search_meta"]["query_attempts"]
    )


@pytest.mark.asyncio
async def test_later_seed_queries_are_skipped_once_the_budget_is_filled():
    lit = _SeedLit({
        "TIER1": [{"pmid": f"t1-{i}", "title": "t", "abstract": "a"} for i in range(20)],
        "TIER2": [{"pmid": "t2-1", "title": "t", "abstract": "a"}],
    })
    await lit._run(seed_queries=["TIER1", "TIER2"])
    assert ("search", "TIER1") in lit.searched
    assert ("search", "TIER2") not in lit.searched


@pytest.mark.asyncio
async def test_a_later_seed_query_runs_when_the_first_comes_up_short():
    lit = _SeedLit({
        "TIER1": [{"pmid": "t1-1", "title": "t", "abstract": "a"}],
        "TIER2": [{"pmid": "t2-1", "title": "t", "abstract": "a"}],
    })
    res = await lit._run(seed_queries=["TIER1", "TIER2"])
    assert ("search", "TIER2") in lit.searched
    assert res["search_meta"]["seed_queries"] == 2


@pytest.mark.asyncio
async def test_search_without_seeds_is_unchanged():
    """Control: the seed path is opt-in and adds no queries when unused."""
    lit = _SeedLit()
    res = await lit._run()
    assert [kind for kind, _ in lit.searched] == ["primary"]
    assert res["search_meta"]["seed_queries"] == 0


class _AnnotateLit(LiteratureAgent):
    def __init__(self):
        self.captured = {}
        self.literature_mode = "pubmed_llm"
        self._annotation_cache = {}
        self._ANNOTATION_CACHE_MAX = 10

        async def _mrp(topic, state=None, **kwargs):
            self.captured.update(kwargs)
            return {"report": "R", "references": [], "focused_evidence": {}, "triage": {}}
        self._multi_round_pubmed = _mrp

    def refresh_runtime_config_if_needed(self):
        pass


@pytest.mark.asyncio
async def test_annotate_clusters_seeds_the_search_with_an_atlas_query():
    lit = _AnnotateLit()

    async def _evaluate(articles, history):
        return {"sufficient": True}

    await lit.annotate_clusters(
        cluster_markers={"0": ["GeneA"]}, organism="species A", tissue="organ A",
        pi_initial_query="q", pi_evaluate_fn=_evaluate, skip_matching=True,
    )
    seeds = lit.captured.get("seed_queries") or []
    assert len(seeds) == 2                       # precise query, then broader fallback
    assert all("atlas" in seed.lower() for seed in seeds)
    assert all("species A" in seed and "organ A" in seed for seed in seeds)
    assert '"organ A"[ti]' in seeds[0]           # the precise one is tried first
    assert lit.captured.get("cluster_markers") == {"0": ["GeneA"]}


# ---------------------------------------------------------------------------
# 2. Per-cluster targeted retrieval for the two buckets that had none
# ---------------------------------------------------------------------------

class _RefLit(LiteratureAgent):
    def __init__(self, articles=None):
        self.searched = []
        self.judged = 0
        self._articles = (
            articles if articles is not None
            else [{"pmid": "r1", "title": "reference paper", "abstract": "abs"}]
        )

        async def _search(query, max_results=10):
            self.searched.append(query)
            if isinstance(self._articles, Exception):
                raise self._articles
            return list(self._articles)
        self._pubmed_search = _search

        async def _judge(item, tissue, articles, bucket):
            self.judged += 1
            return {"status": "strong", "judge": {}}
        self._judge_focused_relevance = _judge


@pytest.mark.asyncio
async def test_cluster_reference_search_retrieves_without_paying_for_a_judge():
    lit = _RefLit()
    items = [{"cluster_id": "4", "candidate_celltype": "parent type A",
              "top_markers": ["GeneA"], "bucket": "context_dependent_state"}]
    out = await lit._cluster_reference_search(items, "species A", "organ A")

    assert out["4"]["status"] == "retrieved"
    assert out["4"]["bucket"] == "context_dependent_state"
    assert out["4"]["articles"][0]["pmid"] == "r1"
    assert lit.judged == 0
    assert "parent type A" in lit.searched[0] and "organ A" in lit.searched[0]


@pytest.mark.asyncio
async def test_cluster_reference_search_reports_no_results_separately_from_failure():
    empty = _RefLit(articles=[])
    items = [{"cluster_id": "4", "candidate_celltype": "parent type A",
              "top_markers": ["GeneA"], "bucket": "covered"}]
    out = await empty._cluster_reference_search(items, "species A", "organ A")
    assert out["4"]["status"] == "exhausted"
    assert out["4"]["exhaustion_reason"] == "no_pubmed_results"

    broken = _RefLit(articles=RuntimeError("NCBI unavailable"))
    out = await broken._cluster_reference_search(items, "species A", "organ A")
    assert out["4"]["exhaustion_reason"] == "search_failed"
    assert "NCBI unavailable" in out["4"]["error"]


@pytest.mark.asyncio
async def test_cluster_reference_search_skips_a_cluster_with_no_query_terms():
    lit = _RefLit()
    items = [{"cluster_id": "9", "candidate_celltype": None, "top_markers": [], "bucket": "covered"}]
    out = await lit._cluster_reference_search(items, "species A", "organ A")
    assert out["9"]["status"] == "exhausted"
    assert out["9"]["exhaustion_reason"] == "empty_query"
    assert lit.searched == []


def test_reference_items_cover_state_and_covered_clusters_with_names_and_markers():
    lit = _RefLit()
    triage = {
        "covered": ["0", "1"],
        "covered_celltypes": {"0": "known type A"},
        "context_dependent_state": [
            {"cluster_id": "3", "parent_celltype": "parent type A", "top_markers": ["GeneD"]}
        ],
        "rare_unresolved": [{"cluster_id": "2", "candidate_celltype": "rare type"}],
        "suspected_contamination": [],
    }
    items = lit._reference_search_items(
        triage, {"0": ["GeneA", "GeneB"], "1": ["GeneC"], "3": ["GeneZ"]}
    )
    by_id = {item["cluster_id"]: item for item in items}

    assert set(by_id) == {"0", "1", "3"}          # rare_unresolved keeps the adjudicated path
    assert by_id["0"]["candidate_celltype"] == "known type A"
    assert by_id["0"]["top_markers"] == ["GeneA", "GeneB"]
    assert by_id["1"]["candidate_celltype"] is None
    assert by_id["1"]["top_markers"] == ["GeneC"]
    assert by_id["3"]["candidate_celltype"] == "parent type A"
    assert by_id["3"]["bucket"] == "context_dependent_state"


class _CollectLit(LiteratureAgent):
    def __init__(self):
        self.focused_calls = []
        self.reference_items = []

        async def _focused(items, organism, tissue, bucket="rare_unresolved", deadline=None):
            self.focused_calls.append((bucket, [i["cluster_id"] for i in items]))
            return {i["cluster_id"]: {"articles": [], "status": "strong", "tier": 1} for i in items}
        self._focused_rare_search = _focused

        async def _reference(items, organism, tissue, deadline=None):
            self.reference_items.extend(items)
            return {
                i["cluster_id"]: {"articles": [], "status": "retrieved", "tier": 1,
                                  "bucket": i.get("bucket")}
                for i in items
            }
        self._cluster_reference_search = _reference


@pytest.mark.asyncio
async def test_collect_cluster_evidence_routes_each_bucket_to_its_regime():
    lit = _CollectLit()
    triage = {
        "covered": ["0"],
        "covered_celltypes": {"0": "known type A"},
        "rare_unresolved": [{"cluster_id": "1", "candidate_celltype": "rare type",
                             "top_markers": ["GeneB"]}],
        "suspected_contamination": [{"cluster_id": "2", "candidate_celltype": None,
                                     "top_markers": ["GeneC"]}],
        "context_dependent_state": [{"cluster_id": "3", "parent_celltype": "parent type",
                                     "top_markers": ["GeneD"]}],
    }
    out = await lit._collect_cluster_evidence(
        triage, "species A", "organ A", cluster_markers={"0": ["GeneA"]}
    )

    assert ("rare_unresolved", ["1"]) in lit.focused_calls
    assert ("suspected_contamination", ["2"]) in lit.focused_calls
    assert {i["cluster_id"] for i in lit.reference_items} == {"0", "3"}
    assert set(out) == {"0", "1", "2", "3"}
    assert out["1"]["bucket"] == "rare_unresolved"
    assert out["3"]["status"] == "retrieved"


@pytest.mark.asyncio
async def test_multi_round_pubmed_attaches_reference_articles_to_the_references():
    class _Loop(_SeedLit):
        async def _eval_fn(self, articles, history):
            return {"sufficient": True, "triage_complete": True, "triage_low_confidence": False,
                    "covered": ["0"], "covered_celltypes": {"0": "known type A"},
                    "rare_unresolved": [], "suspected_contamination": [],
                    "context_dependent_state": []}

    lit = _Loop()

    async def _reference(items, organism, tissue, deadline=None):
        return {
            i["cluster_id"]: {
                "articles": [{"pmid": "ref-1", "title": "reference", "abstract": "x"}],
                "status": "retrieved", "tier": 1, "bucket": i.get("bucket"),
            }
            for i in items
        }
    lit._cluster_reference_search = _reference

    res = await lit._run(cluster_markers={"0": ["GeneA"]})
    assert res["focused_evidence"]["0"]["status"] == "retrieved"
    assert any(ref["url"].endswith("/ref-1/") for ref in res["references"])


# ---------------------------------------------------------------------------
# 3. The PI triage names covered clusters so they can be searched
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
async def test_evaluation_prompt_asks_for_a_covered_celltype_per_cluster():
    pi = _FakePI(json.dumps({"sufficient": True}))
    await pi.evaluate_annotation_search_results(
        cluster_markers={"0": ["GeneA"]}, organism="species A", tissue="organ A",
        articles=[], search_history=[],
    )
    assert "covered_celltypes" in pi.captured["prompt"]


@pytest.mark.asyncio
async def test_covered_celltypes_are_normalized_and_restricted_to_covered_clusters():
    payload = {
        "sufficient": True, "covered": ["0", "1"],
        "covered_celltypes": {"0": " known type A ", "1": "null", "9": "hallucinated"},
        "rare_unresolved": [], "suspected_contamination": [], "context_dependent_state": [],
    }
    pi = _FakePI(json.dumps(payload))
    out = await pi.evaluate_annotation_search_results(
        cluster_markers={"0": ["a"], "1": ["b"]}, organism="species A", tissue="organ A",
        articles=[], search_history=[],
    )
    assert out["covered_celltypes"] == {"0": "known type A"}


@pytest.mark.asyncio
async def test_covered_celltypes_default_to_an_empty_map():
    pi = _FakePI(json.dumps({"sufficient": True, "covered": ["0"]}))
    out = await pi.evaluate_annotation_search_results(
        cluster_markers={"0": ["a"]}, organism="species A", tissue="organ A",
        articles=[], search_history=[],
    )
    assert out["covered_celltypes"] == {}


# ---------------------------------------------------------------------------
# 4. Per-cluster evidence leads the matching prompt
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


async def _match(pi, triage, focused, markers=None, review="BACKGROUND_REVIEW_MARKER"):
    await pi.annotate_clusters_from_literature(
        cluster_markers=markers or {"0": ["GeneA"]}, organism="species A", tissue="organ A",
        literature_report=review, triage=triage, focused_evidence=focused,
    )
    return pi.captured["prompt"]


@pytest.mark.asyncio
async def test_covered_cluster_evidence_reaches_the_matching_prompt():
    triage = {"covered": ["0"], "covered_celltypes": {"0": "known type A"},
              "rare_unresolved": [], "suspected_contamination": [], "context_dependent_state": []}
    focused = {"0": {"articles": [{"title": "covered paper", "abstract": "covered abstract"}],
                     "status": "retrieved", "tier": 1, "bucket": "covered"}}
    prompt = await _match(_MatchPI(), triage, focused)

    assert "Cluster 0" in prompt
    assert "covered paper" in prompt and "covered abstract" in prompt
    assert "known type A" in prompt


@pytest.mark.asyncio
async def test_context_state_cluster_evidence_reaches_the_matching_prompt():
    triage = {"covered": [], "rare_unresolved": [], "suspected_contamination": [],
              "context_dependent_state": [{"cluster_id": "0", "parent_celltype": "parent type A",
                                           "context": "condition A", "top_markers": ["GeneA"]}]}
    focused = {"0": {"articles": [{"title": "state paper", "abstract": "state abstract"}],
                     "status": "retrieved", "tier": 1, "bucket": "context_dependent_state"}}
    prompt = await _match(_MatchPI(), triage, focused)

    assert "context-dependent state" in prompt.lower()
    assert "state paper" in prompt


@pytest.mark.asyncio
async def test_retrieved_evidence_is_labelled_unadjudicated():
    triage = {"covered": ["0"], "covered_celltypes": {"0": "known type A"},
              "rare_unresolved": [], "suspected_contamination": [], "context_dependent_state": []}
    focused = {"0": {"articles": [{"title": "covered paper", "abstract": "x"}],
                     "status": "retrieved", "tier": 1, "bucket": "covered"}}
    prompt = await _match(_MatchPI(), triage, focused)
    assert "not relevance-adjudicated" in prompt.lower()


@pytest.mark.asyncio
async def test_per_cluster_evidence_precedes_the_background_review():
    triage = {"covered": ["0"], "covered_celltypes": {"0": "known type A"},
              "rare_unresolved": [], "suspected_contamination": [], "context_dependent_state": []}
    focused = {"0": {"articles": [{"title": "covered paper", "abstract": "x"}],
                     "status": "retrieved", "tier": 1, "bucket": "covered"}}
    prompt = await _match(_MatchPI(), triage, focused)

    assert prompt.index("covered paper") < prompt.index("BACKGROUND_REVIEW_MARKER")
    assert "secondary" in prompt.lower()


@pytest.mark.asyncio
async def test_covered_cluster_without_evidence_adds_no_noise():
    """Control: a covered cluster the search never reached stays silent."""
    triage = {"covered": ["0"], "rare_unresolved": [], "suspected_contamination": [],
              "context_dependent_state": []}
    prompt = await _match(_MatchPI(), triage, {})
    assert "Focused evidence exhausted for Cluster 0" not in prompt


@pytest.mark.asyncio
async def test_cluster_evidence_is_rendered_in_cluster_order():
    triage = {"covered": ["1"], "covered_celltypes": {"1": "known type B"},
              "rare_unresolved": [{"cluster_id": "0", "candidate_celltype": "rare type A"}],
              "suspected_contamination": [], "context_dependent_state": []}
    focused = {
        "0": {"articles": [{"title": "rare paper", "abstract": "x"}], "status": "strong", "tier": 1},
        "1": {"articles": [{"title": "covered paper", "abstract": "x"}], "status": "retrieved",
              "tier": 1, "bucket": "covered"},
    }
    prompt = await _match(_MatchPI(), triage, focused, markers={"0": ["GeneA"], "1": ["GeneB"]})
    assert prompt.index("rare paper") < prompt.index("covered paper")
