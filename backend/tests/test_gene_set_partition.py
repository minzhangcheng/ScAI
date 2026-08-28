"""An ontology process term contains its own regulators, so scoring it whole says little.

Two production runs registered 9 of 10 panels as hand-assembled, and the model's
stated reason was correct: "broad GO sets mix opposing regulators". A module
score over a set holding both the drivers and the restrainers of a process rises
whether the process is running or being held in check.

The ontology already separates them. Measured on the installed resource: the
biological-process collections publish hundreds of positive- and
negative-regulation terms, several hundred processes carry both, and the
regulator sets sit inside the base term. Splitting is therefore string work on
identifiers -- no biology is hardcoded here, and a test below enforces that.
"""
import inspect

import pytest

import backend.agents.bioinformatics_agent as bio_mod
from backend.agents.bioinformatics import executor_gene_sets
from backend.services.analysis_plan import canonical_panel, panel_hash
from backend.services.gene_set_provenance import parse_provenance
from backend.services.gene_set_resolver import (
    PARTITION_PARTS,
    describe_partition,
    direction_sibling_ids,
    partition_by_direction,
    partition_provenance,
    validate_set_id,
)


# ------------------------------------------------------- sibling derivation

def test_siblings_are_derived_by_name():
    ids = direction_sibling_ids("ONT_SOME_PROCESS")
    assert ids == {
        "positive": "ONT_POSITIVE_REGULATION_OF_SOME_PROCESS",
        "negative": "ONT_NEGATIVE_REGULATION_OF_SOME_PROCESS",
    }


def test_an_already_directional_set_has_nothing_to_split():
    assert direction_sibling_ids("ONT_POSITIVE_REGULATION_OF_X") == {}
    assert direction_sibling_ids("ONT_NEGATIVE_REGULATION_OF_X") == {}


@pytest.mark.parametrize("bad", ["", "   ", "NOPREFIX"])
def test_underivable_identifiers_return_nothing(bad):
    assert direction_sibling_ids(bad) == {}


def test_derivation_does_not_assert_existence():
    """Names are derived mechanically; only the fetch confirms they exist.

    Most collections publish no regulation terms at all, so a derived name that
    happens to be meaningless must not be treated as a promise.
    """
    assert direction_sibling_ids("HALLMARK_ANYTHING")


# --------------------------------------------------------------- the split

def _split(base, pos, neg):
    return partition_by_direction(base, {"positive": pos, "negative": neg})


def test_partition_is_lossless():
    base = ["A", "B", "C", "D", "E"]
    parts = _split(base, ["A", "B"], ["C"])
    union = {g for part in parts.values() for g in part}
    assert union == set(base)


def test_partition_is_mutually_exclusive():
    """A gene counted in two arms would argue for both directions at once."""
    base = ["A", "B", "C"]
    parts = _split(base, ["A", "B"], ["B", "C"])
    seen = []
    for part in parts.values():
        seen.extend(part)
    assert len(seen) == len(set(seen))


def test_genes_in_both_arms_become_ambiguous():
    """Context-dependent regulators cannot support a directional claim."""
    parts = _split(["A", "B", "C"], ["A", "B"], ["B", "C"])
    assert parts["ambiguous"] == ["B"]
    assert parts["positive"] == ["A"]
    assert parts["negative"] == ["C"]


def test_core_is_what_no_arm_claims():
    parts = _split(["A", "B", "C", "D"], ["A"], ["B"])
    assert set(parts["core"]) == {"C", "D"}


def test_siblings_outside_the_base_are_ignored():
    """A regulation term may name genes the base does not; those are not ours."""
    parts = _split(["A"], ["A", "ZZZ"], [])
    assert parts["positive"] == ["A"]
    assert "ZZZ" not in {g for part in parts.values() for g in part}


def test_empty_arms_yield_empty_parts_not_errors():
    parts = _split(["A", "B"], [], [])
    assert parts["core"] == ["A", "B"]
    assert parts["positive"] == [] and parts["negative"] == []


def test_partition_matches_the_plan_normalisation():
    """A part must hash the way a frozen plan hashes, or drift is spurious."""
    parts = _split(["b", "A"], ["A"], [])
    assert parts["core"] == canonical_panel(["b"])
    assert panel_hash(parts["positive"]) == panel_hash(["a"])


def test_all_declared_parts_are_produced():
    parts = _split(["A", "B", "C"], ["A", "B"], ["B", "C"])
    assert set(parts) == set(PARTITION_PARTS)


# ---------------------------------------------------------------- provenance

def test_part_provenance_stays_parseable():
    base = "database:msigdb:2025.1.Mm:ONT_SOME_PROCESS"
    for part in PARTITION_PARTS:
        prov, err = parse_provenance(partition_provenance(base, part))
        assert err is None
        assert prov.kind == "database"
        assert prov.set_id.endswith("#" + part)


def test_partitioned_set_id_passes_validation():
    assert validate_set_id("ONT_SOME_PROCESS#positive") is None


def test_describe_partition_reports_every_part():
    text = describe_partition(_split(["A", "B"], ["A"], ["B"]))
    for part in PARTITION_PARTS:
        assert part in text


# ------------------------------------------------------------------- wiring

def _fetch_block() -> str:
    return inspect.getsource(executor_gene_sets.handle_fetch_gene_set)


def test_partition_mode_is_exposed():
    schema = next(t["function"]["parameters"] for t in bio_mod.TOOLS
                  if t["function"]["name"] == "fetch_gene_set")
    assert "partition_by_direction" in schema["properties"]


def test_handler_binds_each_part_separately():
    block = _fetch_block()
    assert "resolve_partitioned_gene_set" in block
    assert "Score them SEPARATELY" in block


def test_whole_set_path_offers_the_split():
    """A caller fetching a splittable set should be told the option exists."""
    block = _fetch_block()
    assert "direction_sibling_ids(result.set_id)" in block
    assert "partition_by_direction=true" in block


def test_partition_failure_is_recorded():
    block = _fetch_block()
    assert "FAILED: partition" in block


# ------------------------------------------------------------- neutrality

def test_partition_logic_names_no_biology():
    import re

    from backend.services import gene_set_resolver

    src = inspect.getsource(gene_set_resolver.partition_by_direction)
    src += inspect.getsource(gene_set_resolver.direction_sibling_ids)
    lowered = src.lower()
    for term in ("ferroptos", "apoptos", "hallmark_", "gobp_", "inflamm"):
        assert term not in lowered
    for word in ("liver", "mouse", "human", "cancer"):
        assert not re.search(rf"\b{word}\b", lowered)
