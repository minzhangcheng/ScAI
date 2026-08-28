"""fetch_gene_set must be usable without already knowing the resource's schema.

The first production run called it six times and never succeeded, so all five
scored panels fell back to hand-assembled. Every failure was a tool defect:

  - the model passed the release in the form the resource publishes it
    ("2024.1.Mm") and the URL builder appended the organism code again;
  - collection identifiers are organism-specific and nothing listed them, so the
    model guessed another organism's prefix;
  - errors returned a bare 404 with a URL, from which nothing can be recovered;
  - failed fetches wrote no AnalysisStep, so the whole episode was invisible in
    session.json and appeared only in the LLM transcript.

These tests are offline; the network paths are exercised separately.
"""
import inspect

import pytest

import backend.agents.bioinformatics_agent as bio_mod
from backend.agents.bioinformatics import executor_gene_sets
from backend.services.gene_set_resolver import (
    cache_path,
    msigdb_gmt_url,
    normalize_version,
    parse_release_index,
)


# ------------------------------------------------------- version tolerance

@pytest.mark.parametrize("raw,release,organism", [
    ("2025.1", "2025.1", ""),
    ("2025.1.Mm", "2025.1", "Mm"),      # the form the resource publishes
    ("v2025.1.Mm", "2025.1", "Mm"),
    ("2024.1.Hs", "2024.1", "Hs"),
])
def test_normalize_version_accepts_every_reasonable_form(raw, release, organism):
    assert normalize_version(raw) == (release, organism)


def test_stamped_version_does_not_double_the_organism_suffix():
    """The exact defect: '2024.1.Mm' became '.../2024.1.Mm.Mm/...'."""
    bare = msigdb_gmt_url("2025.1", "mh.all", "Mm")
    stamped = msigdb_gmt_url("2025.1.Mm", "mh.all", "Mm")
    assert bare == stamped
    assert ".Mm.Mm" not in stamped


def test_mismatched_organism_in_version_is_refused():
    """Asking for one organism's release while analysing another is a mistake."""
    with pytest.raises(ValueError, match="organism"):
        msigdb_gmt_url("2025.1.Hs", "mh.all", "Mm")


def test_cache_key_is_release_not_raw_version():
    """Otherwise the same file caches twice under two spellings."""
    assert cache_path("/tmp", "2025.1", "mh.all", "Mm") == \
           cache_path("/tmp", "2025.1.Mm", "mh.all", "Mm")


def test_unparseable_version_is_not_treated_as_a_release():
    release, organism = normalize_version("Enrichr_2024")
    assert release == "Enrichr_2024" and organism == ""


# ---------------------------------------------------------------- discovery

def test_parse_release_index_extracts_collection_ids():
    html = (
        '<a href="mh.all.v2025.1.Mm.symbols.gmt">x</a>'
        '<a href="m5.go.bp.v2025.1.Mm.symbols.gmt">y</a>'
        '<a href="mh.all.v2025.1.Mm.entrez.gmt">z</a>'   # wrong id space
        '<a href="mh.all.v2024.1.Mm.symbols.gmt">w</a>'  # wrong release
    )
    assert parse_release_index(html, "2025.1", "Mm") == ["m5.go.bp", "mh.all"]


def test_parse_release_index_accepts_a_stamped_version():
    html = '<a href="mh.all.v2025.1.Mm.symbols.gmt">x</a>'
    assert parse_release_index(html, "2025.1.Mm", "Mm") == ["mh.all"]


# ------------------------------------------------------------------- wiring

def _fetch_block() -> str:
    return inspect.getsource(executor_gene_sets.handle_fetch_gene_set)


def test_list_collections_mode_exists():
    """Without it the only route to a collection identifier is guessing."""
    schema = next(t["function"]["parameters"] for t in bio_mod.TOOLS
                  if t["function"]["name"] == "fetch_gene_set")
    assert "list_collections" in schema["properties"]
    assert "list_collections" in _fetch_block()


def test_only_version_is_unconditionally_required():
    """set_id and collection cannot be required when discovery is the first step."""
    schema = next(t["function"]["parameters"] for t in bio_mod.TOOLS
                  if t["function"]["name"] == "fetch_gene_set")
    assert schema["required"] == ["version"]


def test_schema_warns_that_collections_are_organism_specific():
    schema = next(t["function"]["parameters"] for t in bio_mod.TOOLS
                  if t["function"]["name"] == "fetch_gene_set")
    description = schema["properties"]["collection"]["description"]
    assert "ORGANISM-SPECIFIC" in description or "organism-specific" in description.lower()
    assert "list_collections" in description


def test_fetch_failures_are_recorded_in_the_analysis_log():
    """A failure invisible in session.json reads as 'never attempted'."""
    block = _fetch_block()
    assert block.count('step_type="fetch_gene_set"') >= 3, (
        "success, discovery and failure paths must each leave a trace"
    )
    assert "FAILED:" in block


def test_fetch_without_identifiers_directs_to_discovery():
    block = _fetch_block()
    assert "needs both collection and set_id" in block
    assert "list_collections=true" in block
