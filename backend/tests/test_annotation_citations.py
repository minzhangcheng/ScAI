"""Structured citations for cluster annotations.

Measured on two live runs before this change: the matching prompt carried zero
PMIDs, all fifteen clusters cited nothing, and the per-cluster articles that the
annotation was actually based on were discarded when the call returned. Neither
model ever fabricated an identifier; they simply had none to name. So the record
has to capture two separate facts, and neither can come from prose:

- what was put in front of the matcher, which survives a matcher that names
  nothing;
- what the matcher said it used, validated against the first.

Every test here fails on the tree before the change.
"""
from backend.agents.pi.annotation import (
    AnnotationMixin as PIAnnotationMixin,
    _evidence_articles,
    _render_cluster_evidence,
)
from backend.models.schemas import AnnotationCitation, ClusterAnnotation
from backend.workflow.annotation import AnnotationMixin


def _article(pmid, title="A title", year="2024"):
    return {"pmid": pmid, "title": title, "abstract": "An abstract.", "year": year}


def _focused(pmid_list, status="retrieved"):
    return {"status": status, "articles": [_article(p) for p in pmid_list]}


# ------------------------------------------------------------- prompt renders


def test_rendered_articles_carry_their_identifier():
    rendered = _evidence_articles(_focused(["12345678"]), 3)

    assert "PMID 12345678" in rendered


def test_evidence_block_and_pool_describe_the_same_articles():
    triage = {"covered": ["1"], "covered_celltypes": {"1": "Kupffer cell"}}
    focused = {"1": _focused(["11111111", "22222222"])}

    text, pool = _render_cluster_evidence(["1"], triage, focused)

    assert "PMID 11111111" in text and "PMID 22222222" in text
    assert [a["pmid"] for a in pool["1"]] == ["11111111", "22222222"]


def test_a_cluster_that_was_searched_and_found_nothing_still_gets_a_pool_entry():
    triage = {"rare_unresolved": [{"cluster_id": "7", "candidate_celltype": "X"}]}
    focused = {"7": {"status": "exhausted", "exhaustion_reason": "no_pubmed_results"}}

    _text, pool = _render_cluster_evidence(["7"], triage, focused)

    assert pool["7"] == []


# ------------------------------------------------------------ citation capture


def _attach(parsed, pool):
    return PIAnnotationMixin._attach_annotation_citations(parsed, pool)


def test_a_named_article_is_recorded_as_used():
    parsed = {"annotations": {"1": {"cell_type": "Kupffer cell",
                                    "cited_pmids": ["11111111"]}}}

    out = _attach(parsed, {"1": [_article("11111111"), _article("22222222")]})
    info = out["annotations"]["1"]

    assert info["citation_status"] == "grounded"
    used = [c for c in info["citations"] if c["used_by_model"]]
    assert [c["pmid"] for c in used] == ["11111111"]
    assert "cited_pmids" not in info


def test_the_pool_is_kept_even_when_the_matcher_names_nothing():
    parsed = {"annotations": {"1": {"cell_type": "Kupffer cell"}}}

    info = _attach(parsed, {"1": [_article("11111111")]})["annotations"]["1"]

    assert info["citation_status"] == "pool_only"
    assert [c["pmid"] for c in info["citations"]] == ["11111111"]
    assert info["citations"][0]["used_by_model"] is False


def test_an_identifier_that_was_never_shown_is_dropped():
    parsed = {"annotations": {"1": {"cell_type": "Kupffer cell",
                                    "cited_pmids": ["11111111", "99999999"]}}}

    info = _attach(parsed, {"1": [_article("11111111")]})["annotations"]["1"]

    assert [c["pmid"] for c in info["citations"]] == ["11111111"]


def test_no_evidence_and_no_pool_entry_are_different_verdicts():
    parsed = {"annotations": {"1": {"cell_type": "A"}, "2": {"cell_type": "B"}}}

    out = _attach(parsed, {"1": []})["annotations"]

    assert out["1"]["citation_status"] == "no_evidence"
    assert out["2"]["citation_status"] == "unavailable"


def test_citations_carry_enough_to_render_without_a_second_lookup():
    parsed = {"annotations": {"1": {"cell_type": "A", "cited_pmids": ["11111111"]}}}

    citation = _attach(
        parsed, {"1": [_article("11111111", title="Kupffer cells in mouse liver")]}
    )["annotations"]["1"]["citations"][0]

    assert citation["title"] == "Kupffer cells in mouse liver"
    assert citation["year"] == "2024"
    assert citation["url"] == "https://pubmed.ncbi.nlm.nih.gov/11111111/"


# --------------------------------------------------------------- persistence


def test_citations_survive_the_save_and_reload_round_trip():
    # _annotations_from_cluster_annotations is the only path back from a saved
    # annotation. A field it omits is dropped on the first reload, and no type
    # check catches it.
    built = AnnotationMixin._build_cluster_annotations({
        "1": {
            "cell_type": "Kupffer cell",
            "citations": [{"pmid": "11111111", "title": "T", "used_by_model": True}],
            "citation_status": "grounded",
        }
    })
    assert [c.pmid for c in built[0].citations] == ["11111111"]

    back = AnnotationMixin._annotations_from_cluster_annotations(built)

    assert back["1"]["citation_status"] == "grounded"
    assert back["1"]["citations"][0]["pmid"] == "11111111"
    assert back["1"]["citations"][0]["used_by_model"] is True

    again = AnnotationMixin._build_cluster_annotations(back)
    assert [c.pmid for c in again[0].citations] == ["11111111"]


def test_an_annotation_without_citations_says_unavailable_not_unsupported():
    ca = ClusterAnnotation(cluster_id="1", cell_type="Kupffer cell")

    assert ca.citations == []
    assert ca.citation_status == "unavailable"


def test_citation_defaults_are_safe():
    c = AnnotationCitation(pmid="11111111")

    assert c.used_by_model is False
    assert c.stale is False
    assert c.source == "focused_search"
