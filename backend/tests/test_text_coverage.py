"""Coverage helpers must work without knowing any domain."""
import inspect

from backend.utils import text_coverage
from backend.utils.text_coverage import extract_salient_terms, uncovered_items


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
    source = inspect.getsource(text_coverage).lower()
    for domain_term in (
        "ferroptos", "apoptos", "cuproptos", "necroptos", "nrf2", "hepatocyte",
        "liver", "mouse", "copper", "tumor", "cancer",
    ):
        assert domain_term not in source
