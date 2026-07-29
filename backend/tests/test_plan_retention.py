"""Retention helpers must measure content loss without knowing any domain."""
import inspect

from backend.utils import plan_retention
from backend.utils.plan_retention import (
    extract_salient_terms,
    missing_salient_terms,
    retention_ratio,
    uncovered_items,
)


def test_stopwords_and_short_tokens_are_not_salient():
    terms = extract_salient_terms(
        "The plan should use the method that was expected for each step"
    )
    for noise in ("the", "that", "was", "for", "each", "step", "plan", "method", "expected"):
        assert noise not in terms


def test_identifier_shaped_tokens_survive_the_length_floor():
    terms = extract_salient_terms("Score Gpx4 and Mt1 across Slc7a11")
    assert {"gpx4", "mt1", "slc7a11"} <= terms


def test_dotted_paths_split_into_usable_tokens():
    terms = extract_salient_terms("scanpy.tl.score_genes on the object")
    assert "scanpy" in terms
    assert "score_genes" in terms


def test_missing_terms_are_exactly_what_the_revision_dropped():
    source = "Score modules for alpha beta gamma using the scoring routine"
    produced = "Score modules for alpha using the scoring routine"
    missing = missing_salient_terms(source, produced)
    assert "beta" in missing
    assert "gamma" in missing
    # Terms the revision kept are never reported, so generic vocabulary that
    # both texts share filters itself out.
    assert "alpha" not in missing
    assert "modules" not in missing
    assert "scoring" not in missing


def test_missing_terms_are_emphasis_first_and_deterministic():
    source = "central central central incidental peripheral"
    missing = missing_salient_terms(source, "")
    assert missing == missing_salient_terms(source, "")
    # Repetition in the source is emphasis, so the repeated term leads.
    assert missing[0] == "central"
    assert missing[1:] == ["incidental", "peripheral"]


def test_explicit_asks_outrank_incidental_prose():
    """A term cut by the limit is a term that never gets reinstated.

    Ranking by rarity surfaces one-off compounds from incidental prose and
    pushes the vocabulary the request was actually about past the cut.
    """
    source = (
        "measure alphaterm alphaterm alphaterm "
        "and note the incidentalcompound-observation once"
    )
    ranked = missing_salient_terms(source, "", priority_text="measure alphaterm")
    assert ranked[0] == "alphaterm"
    assert ranked.index("alphaterm") < ranked.index("incidentalcompound-observation")


def test_missing_terms_respect_the_limit():
    source = " ".join(f"term{i}word" for i in range(50))
    assert len(missing_salient_terms(source, "", limit=10)) == 10
    assert len(missing_salient_terms(source, "", limit=0)) == 50


def test_retention_ratio_bounds():
    assert retention_ratio("", "anything") == 1.0
    assert retention_ratio("alpha beta", "alpha beta") == 1.0
    assert retention_ratio("alpha beta", "alpha") == 0.5
    assert retention_ratio("alpha beta", "unrelated") == 0.0


def test_uncovered_items_flags_only_unaddressed_requirements():
    items = [
        "Score nuclei for alphastress and betastress programs separately",
        "Compare against public gammainjury reference datasets",
        "   ",
    ]
    produced = "Step 4 scores nuclei for alphastress and betastress programs separately"
    uncovered = uncovered_items(items, produced)
    assert len(uncovered) == 1
    assert "gammainjury" in uncovered[0]


def test_module_carries_no_domain_vocabulary():
    source = inspect.getsource(plan_retention).lower()
    for domain_term in (
        "ferroptos", "apoptos", "cuproptos", "necroptos", "nrf2", "hepatocyte",
        "liver", "mouse", "copper", "tumor", "cancer",
    ):
        assert domain_term not in source
