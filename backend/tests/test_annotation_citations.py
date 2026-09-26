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
)
from backend.workflow.annotation import AnnotationMixin


def _article(pmid, title="A title", year="2024"):
    return {"pmid": pmid, "title": title, "abstract": "An abstract.", "year": year}


def _attach(parsed, pool):
    return PIAnnotationMixin._attach_annotation_citations(parsed, pool)


def test_an_identifier_that_was_never_shown_is_dropped():
    parsed = {"annotations": {"1": {"cell_type": "Kupffer cell",
                                    "cited_pmids": ["11111111", "99999999"]}}}

    info = _attach(parsed, {"1": [_article("11111111")]})["annotations"]["1"]

    assert [c["pmid"] for c in info["citations"]] == ["11111111"]


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
