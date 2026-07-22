"""Generic replay: triage, focused search, one synthesis, and limited evidence retention."""
import pytest
from backend.agents.literature_agent import LiteratureAgent, build_focused_query, status_from_judge, plausibility_terms_for_tissue


def test_rare_query_uses_supplied_term_and_contamination_avoids_candidate():
    rare = {"candidate_celltype": "candidate cell type A", "synonyms": ["Alias A"],
            "top_markers": ["GeneA", "GeneB"]}
    assert "candidate cell type A" in build_focused_query(
        rare, "species A", "organ A", 1, "rare_unresolved"
    )
    suspect = {"candidate_celltype": "candidate B", "top_markers": ["GeneC", "GeneD"]}
    discovery = build_focused_query(
        suspect, "species A", "organ A", 1, "suspected_contamination"
    )
    assert "candidate B" not in discovery and '"GeneC"[tiab]' in discovery


def test_limited_status_not_dropped():
    assert status_from_judge({"strong": 1, "moderate": 1, "contradictory": False}) == "limited"


def test_replay_uses_no_static_tissue_vocabulary():
    assert plausibility_terms_for_tissue("species A organ A") == []
    assert plausibility_terms_for_tissue("species B organ B") == []


class _Replay(LiteratureAgent):
    def __init__(self):
        self.max_search_rounds = 40; self.synth_calls = 0; self.focused_buckets = []
        async def _vd(q, t, max_results=20): return q, [{"pmid": "1", "title": "a", "abstract": "b"}]
        self._validate_and_degrade_query = _vd
        async def _s(q, max_results=20): return []
        self._pubmed_search = _s
        async def _syn(t, arts): self.synth_calls += 1; return "REVIEW"
        self._synthesize_pubmed_results = _syn
        async def _fin(t, r, refs, st, n): return {"report": r, "references": refs}
        self._finalize_result = _fin
        async def _foc(items, organism, tissue, bucket="rare_unresolved"):
            self.focused_buckets.append(bucket)
            out = {}
            for it in items:
                cid = it["cluster_id"]
                st = "limited" if cid == "19" else "strong"
                out[cid] = {"articles": [{"pmid": "9", "title": "F", "abstract": "x"}],
                            "status": st, "tier": 1, "exhaustion_reason": None, "error": None}
            return out
        self._focused_rare_search = _foc

    async def _eval(self, articles, history):
        return {"sufficient": False, "refined_query": "q", "triage_complete": True,
                "triage_low_confidence": False, "covered": ["0"],
                "rare_unresolved": [{"cluster_id": "22", "candidate_celltype": "candidate A", "synonyms": [], "top_markers": ["GeneA"]},
                                     {"cluster_id": "19", "candidate_celltype": "candidate B", "synonyms": [], "top_markers": ["GeneB"]}],
                "suspected_contamination": [{"cluster_id": "8", "candidate_celltype": None, "top_markers": ["GeneC"]}],
                "context_dependent_state": [{"cluster_id": "18", "parent_celltype": "parent A", "context": "condition A"}]}


@pytest.mark.asyncio
async def test_replay_terminates_triggers_focused_single_synthesis():
    lit = _Replay()
    res = await lit._multi_round_pubmed("Cell types in species A organ A", state=None, pubmed_max_results=50,
        initial_query="q1", evaluate_fn=lit._eval, organism="species A", tissue="organ A")
    assert "rare_unresolved" in lit.focused_buckets and "suspected_contamination" in lit.focused_buckets
    assert lit.synth_calls == 1
    assert res["focused_evidence"]["19"]["status"] == "limited"
    assert res["focused_evidence"]["22"]["status"] == "strong"
    assert "18" not in res["focused_evidence"]      # context_dependent_state 不做 focused
