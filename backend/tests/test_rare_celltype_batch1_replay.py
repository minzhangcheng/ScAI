"""回放 7753ea99 典型形态: 全簇归桶→语义终止→focused 按桶触发→单次合成,且 limited 不丢弃。"""
import pytest
from backend.agents.literature_agent import LiteratureAgent, build_focused_query, status_from_judge, plausibility_terms_for_tissue


def test_pdc_query_uses_official_term_and_neuronal_avoids_candidate():
    pdc = {"candidate_celltype": "plasmacytoid dendritic cell", "synonyms": ["pDC"],
           "top_markers": ["Siglech", "Tcf4"]}
    assert "plasmacytoid dendritic cell" in build_focused_query(pdc, "mouse", "liver", 1, "rare_unresolved")
    neuro = {"candidate_celltype": "neuronal", "top_markers": ["Nrxn1", "Reln"]}
    disc = build_focused_query(neuro, "mouse", "liver", 1, "suspected_contamination")
    assert "neuronal" not in disc and '"Nrxn1"[tiab]' in disc


def test_limited_status_not_dropped():
    assert status_from_judge({"strong": 1, "moderate": 1, "contradictory": False}) == "limited"


def test_liver_terms_do_not_leak_to_kidney_replay_guard():
    liver_terms = " ".join(plausibility_terms_for_tissue("mouse liver"))
    kidney_terms = " ".join(plausibility_terms_for_tissue("mouse kidney"))
    assert "portal tract" in liver_terms
    assert "portal tract" not in kidney_terms
    assert "Schwann" in kidney_terms


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
                "rare_unresolved": [{"cluster_id": "22", "candidate_celltype": "pDC", "synonyms": [], "top_markers": ["Siglech"]},
                                     {"cluster_id": "19", "candidate_celltype": "cholangiocyte", "synonyms": [], "top_markers": ["Pkhd1"]}],
                "suspected_contamination": [{"cluster_id": "8", "candidate_celltype": None, "top_markers": ["Nrxn1"]}],
                "context_dependent_state": [{"cluster_id": "18", "parent_celltype": "hepatocyte", "context": "DKO"}]}


@pytest.mark.asyncio
async def test_replay_terminates_triggers_focused_single_synthesis():
    lit = _Replay()
    res = await lit._multi_round_pubmed("Cell types in mouse liver", state=None, pubmed_max_results=50,
        initial_query="q1", evaluate_fn=lit._eval, organism="mouse", tissue="liver")
    assert "rare_unresolved" in lit.focused_buckets and "suspected_contamination" in lit.focused_buckets
    assert lit.synth_calls == 1
    assert res["focused_evidence"]["19"]["status"] == "limited"
    assert res["focused_evidence"]["22"]["status"] == "strong"
    assert "18" not in res["focused_evidence"]      # context_dependent_state 不做 focused
