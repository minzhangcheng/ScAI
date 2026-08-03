"""annotate_clusters_from_literature: 注入归桶 hint + focused 证据(含摘要),contradictory 按 bucket 区分。"""
import json
import pytest
from backend.agents.pi_agent import PIAgent


class _FakePI(PIAgent):
    def __init__(self):
        self.captured = {}
        async def _chat(message, temperature=None, **kwargs):
            self.captured["prompt"] = message
            return json.dumps({"annotations": {"18": {"cell_type": "Hepatocyte"}}, "tissue_summary": ""})
        self.chat_without_history = _chat
    def refresh_runtime_config_if_needed(self):
        pass


@pytest.mark.asyncio
async def test_context_and_focused_hints_with_abstract():
    pi = _FakePI()
    triage = {"context_dependent_state": [{"cluster_id": "18", "parent_celltype": "hepatocyte", "context": "DKO stress"}],
              "rare_unresolved": [{"cluster_id": "22", "candidate_celltype": "plasmacytoid dendritic cell"}],
              "suspected_contamination": []}
    focused = {"22": {"articles": [{"title": "pDC paper", "abstract": "Siglech Tcf4 define pDC"}],
                      "status": "strong", "tier": 1}}
    await pi.annotate_clusters_from_literature(
        cluster_markers={"18": ["Mt1"], "22": ["Siglech"]}, organism="mouse", tissue="liver",
        literature_report="REVIEW", triage=triage, focused_evidence=focused)
    p = pi.captured["prompt"]
    assert "context-dependent state" in p.lower() and "hepatocyte" in p.lower()
    assert "Focused evidence for Cluster 22" in p
    assert "pDC paper" in p and "define pDC" in p          # 含摘要片段


@pytest.mark.asyncio
async def test_database_priors_reach_the_actual_pi_matching_prompt():
    pi = _FakePI()
    priors = {"8": [{
        "cell_type": "pDC", "p_adj": 3.4e-4,
        "overlap": ["IRF8", "SIGLECH"],
        "marker_source": "provided_cluster_markers",
    }]}

    await pi.annotate_clusters_from_literature(
        cluster_markers={"8": ["Irf8", "Siglech"]},
        organism="mouse", tissue="liver", literature_report="REVIEW",
        database_priors=priors,
    )

    prompt = pi.captured["prompt"]
    assert "=== DATABASE PRIORS" in prompt
    assert "Cluster 8" in prompt and "pDC" in prompt
    assert "IRF8" in prompt and "SIGLECH" in prompt


@pytest.mark.asyncio
async def test_contradictory_hint_differs_by_bucket():
    pi = _FakePI()
    triage = {"rare_unresolved": [{"cluster_id": "5", "candidate_celltype": "fibroblast"}],
              "suspected_contamination": [{"cluster_id": "8", "candidate_celltype": None}],
              "context_dependent_state": []}
    focused = {"5": {"articles": [], "status": "contradictory", "tier": 2},
               "8": {"articles": [], "status": "contradictory", "tier": 2}}
    await pi.annotate_clusters_from_literature(
        cluster_markers={"5": ["A"], "8": ["Nrxn1"]}, organism="mouse", tissue="liver",
        literature_report="R", triage=triage, focused_evidence=focused)
    p = pi.captured["prompt"]
    assert "consider alternative" in p.lower()             # rare bucket 措辞
    assert "contamination" in p.lower()                    # suspected bucket 措辞


@pytest.mark.asyncio
async def test_exhausted_marker():
    pi = _FakePI()
    triage = {"rare_unresolved": [{"cluster_id": "8", "candidate_celltype": "x"}],
              "suspected_contamination": [], "context_dependent_state": []}
    focused = {"8": {"articles": [], "status": "exhausted", "tier": 2}}
    await pi.annotate_clusters_from_literature(
        cluster_markers={"8": ["Nrxn1"]}, organism="mouse", tissue="liver",
        literature_report="R", triage=triage, focused_evidence=focused)
    assert "Focused evidence exhausted" in pi.captured["prompt"]


@pytest.mark.asyncio
async def test_runtime_failure_hint_is_not_absence_of_literature():
    pi = _FakePI()
    triage = {"rare_unresolved": [{"cluster_id": "22", "candidate_celltype": "pDC"}],
              "suspected_contamination": [], "context_dependent_state": []}
    focused = {"22": {"articles": [], "status": "exhausted", "tier": 1,
                       "exhaustion_reason": "search_failed", "error": "NCBI unavailable"}}
    await pi.annotate_clusters_from_literature(
        cluster_markers={"22": ["Siglech"]}, organism="mouse", tissue="liver",
        literature_report="R", triage=triage, focused_evidence=focused)
    prompt = pi.captured["prompt"]
    assert "focused evidence unavailable" in prompt.lower()
    assert "runtime failure" in prompt.lower()
    assert "NCBI unavailable" in prompt
    assert "do not interpret as absence of literature" in prompt.lower()


@pytest.mark.asyncio
async def test_no_pubmed_results_hint_is_true_no_results():
    pi = _FakePI()
    triage = {"rare_unresolved": [{"cluster_id": "22", "candidate_celltype": "pDC"}],
              "suspected_contamination": [], "context_dependent_state": []}
    focused = {"22": {"articles": [], "status": "exhausted", "tier": 1,
                       "exhaustion_reason": "no_pubmed_results", "error": None}}
    await pi.annotate_clusters_from_literature(
        cluster_markers={"22": ["Siglech"]}, organism="mouse", tissue="liver",
        literature_report="R", triage=triage, focused_evidence=focused)
    prompt = pi.captured["prompt"]
    assert "focused search found no pubmed results" in prompt.lower()
    assert "evidence scarcity" in prompt.lower()
    # The phrase "not a runtime failure" technically contains "runtime failure",
    # so we check for the full contrastive clause instead.
    assert "not a runtime failure" in prompt.lower()


@pytest.mark.asyncio
async def test_no_relevant_evidence_hint_uses_exhausted_language():
    pi = _FakePI()
    triage = {"rare_unresolved": [{"cluster_id": "22", "candidate_celltype": "pDC"}],
              "suspected_contamination": [], "context_dependent_state": []}
    focused = {"22": {"articles": [{"title": "marker paper", "abstract": "generic"}],
                       "status": "exhausted", "tier": 1,
                       "exhaustion_reason": "no_relevant_evidence", "error": None}}
    await pi.annotate_clusters_from_literature(
        cluster_markers={"22": ["Siglech"]}, organism="mouse", tissue="liver",
        literature_report="R", triage=triage, focused_evidence=focused)
    prompt = pi.captured["prompt"]
    assert "retrieved articles were not relevant" in prompt.lower()
    assert "marker paper" in prompt
