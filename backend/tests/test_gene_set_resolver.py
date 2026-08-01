"""Gene sets come from a versioned resource, not from the model's memory.

Sandbox code cannot reach the network, so before this the agent had no way to
fetch a curated set and every panel it scored was remembered. These tests cover
the pure resolution logic offline; the network path is exercised separately.

The cross-organism behaviour is the part worth pinning: the upstream resource
publishes two organisms natively, everything else is projected, and a projection
that loses most of the set must say so rather than passing as equivalent.
"""
import inspect

import pytest

import backend.agents.bioinformatics_agent as bio_mod
from backend.services.gene_set_resolver import (
    DEFAULT_HOMOLOGY_TYPES,
    LOW_MAPPING_RATE,
    MSIGDB_ORGANISMS,
    GeneSetResult,
    build_provenance,
    cache_path,
    msigdb_gmt_url,
    normalize_organism,
    parse_gmt,
    suggest_similar,
    summarize_projection,
    validate_set_id,
)
from backend.services.gene_set_provenance import parse_provenance


# ------------------------------------------------------------------ organisms

@pytest.mark.parametrize("raw,expected", [
    ("Mus musculus", "mouse"), ("mouse", "mouse"), ("Mm", "mouse"),
    ("Homo sapiens", "human"), ("HUMAN", "human"), ("Hs", "human"),
    ("Danio rerio", "danio rerio"),   # unknown organisms pass through
])
def test_normalize_organism(raw, expected):
    assert normalize_organism(raw) == expected


def test_only_two_organisms_are_native():
    """Everything else must go through projection, not be silently mislabelled."""
    assert set(MSIGDB_ORGANISMS) == {"human", "mouse"}


# ----------------------------------------------------------------------- GMT

def test_parse_gmt_reads_name_and_genes():
    text = "SET_A\tdescription\tG1\tG2\tG3\nSET_B\tdesc\tG4\n"
    sets = parse_gmt(text)
    assert sets == {"SET_A": ["G1", "G2", "G3"], "SET_B": ["G4"]}


def test_parse_gmt_skips_blank_fields_and_short_lines():
    text = "SET_A\tdesc\tG1\t\tG2\n\nBAD_LINE\tdesc\n"
    sets = parse_gmt(text)
    assert sets == {"SET_A": ["G1", "G2"]}


def test_gmt_url_embeds_the_release():
    """The release in the path is what makes a citation re-fetchable."""
    url = msigdb_gmt_url("2025.1", "mh.all", "Mm")
    assert url.endswith("/2025.1.Mm/mh.all.v2025.1.Mm.symbols.gmt")


def test_cache_path_is_version_and_organism_scoped(tmp_path):
    a = cache_path(tmp_path, "2025.1", "mh.all", "Mm")
    b = cache_path(tmp_path, "2024.1", "mh.all", "Mm")
    c = cache_path(tmp_path, "2025.1", "mh.all", "Hs")
    assert a != b != c and a != c, "a version bump must not overwrite the old cache"


# --------------------------------------------------------------- identifiers

def test_validate_set_id_rejects_empty():
    assert validate_set_id("") is not None


def test_validate_set_id_rejects_injection_shaped_input():
    assert validate_set_id("../../etc/passwd") is not None


def test_validate_set_id_accepts_normal_identifiers():
    assert validate_set_id("SOME_SET_ID_1") is None


def test_suggest_similar_matches_on_shared_tokens():
    """A near-miss should be recoverable in one step, not a dead end."""
    available = ["ALPHA_BETA_RESPONSE", "ALPHA_GAMMA_SIGNAL", "UNRELATED_THING"]
    hits = suggest_similar("ALPHA_BETA", available)
    assert hits[0] == "ALPHA_BETA_RESPONSE"


def test_suggest_similar_returns_nothing_for_no_overlap():
    assert suggest_similar("ZZZ", ["ALPHA_BETA_RESPONSE"]) == []


# ---------------------------------------------------------------- provenance

def test_built_provenance_parses_under_the_controlled_vocabulary():
    """The resolver and the validator must agree on the format."""
    raw = build_provenance("msigdb", "2025.1.Mm", "SOME_SET")
    prov, err = parse_provenance(raw)
    assert err is None
    assert prov.kind == "database"
    assert prov.version == "2025.1.Mm"
    assert prov.set_id == "SOME_SET"


# ---------------------------------------------------------------- projection

def test_native_result_has_no_projection_warning():
    result = GeneSetResult(
        set_id="S", genes=["A", "B"], resource="msigdb", version="2025.1.Mm",
        collection="c", organism="mouse", source_organism="mouse", projected=False,
    )
    assert result.warnings() == []


def test_projected_result_always_warns():
    result = GeneSetResult(
        set_id="S", genes=["A"], resource="msigdb", version="2025.1.Mm",
        collection="c", organism="rattus norvegicus", source_organism="mouse",
        projected=True,
        projection={"n_source": 10, "n_mapped": 9, "mapping_rate": 0.9},
    )
    warnings = result.warnings()
    assert warnings
    assert "not equivalent" in warnings[0]


def test_low_mapping_rate_gets_an_extra_warning():
    """Losing most of a set is a different claim from losing a few members."""
    result = GeneSetResult(
        set_id="S", genes=["A"], resource="msigdb", version="2025.1.Mm",
        collection="c", organism="danio rerio", source_organism="mouse",
        projected=True,
        projection={"n_source": 100, "n_mapped": 12, "mapping_rate": 0.12},
    )
    warnings = result.warnings()
    assert len(warnings) == 2
    assert "not a random sample" in warnings[1]
    assert "exploratory" in warnings[1]


def test_mapping_rate_threshold_is_explicit():
    assert 0 < LOW_MAPPING_RATE < 1


def test_default_homology_is_one_to_one_only():
    """one2many introduces duplicate counting; it must be opt-in."""
    assert DEFAULT_HOMOLOGY_TYPES == ("ortholog_one2one",)


def test_summarize_projection_records_one_to_many():
    summary = summarize_projection(
        3, {"A": ["a1"], "B": ["b1", "b2"]}, ("ortholog_one2one",)
    )
    assert summary["n_source"] == 3
    assert summary["n_mapped"] == 2
    assert summary["mapping_rate"] == pytest.approx(0.6667, abs=1e-4)
    assert summary["one_to_many"] == ["B"]


# ------------------------------------------------------------------- wiring

def test_fetch_gene_set_is_a_registered_tool():
    names = {t["function"]["name"] for t in bio_mod.TOOLS}
    assert "fetch_gene_set" in names
    assert "fetch_gene_set" in bio_mod.KNOWN_TOOL_NAMES


def test_fetch_gene_set_has_a_dispatch_branch():
    src = inspect.getsource(bio_mod)
    assert 'elif tool_name == "fetch_gene_set":' in src


def test_fetch_runs_off_the_event_loop():
    """Network I/O in the dispatch loop would stall every other session."""
    src = inspect.getsource(bio_mod)
    start = src.index('elif tool_name == "fetch_gene_set":')
    end = src.index('elif tool_name == "register_analysis_plan":', start)
    block = src[start:end]
    assert "asyncio.to_thread" in block


def test_fetched_genes_are_injected_not_retyped():
    """Making the model retype the panel reintroduces exactly what this prevents."""
    src = inspect.getsource(bio_mod)
    start = src.index('elif tool_name == "fetch_gene_set":')
    end = src.index('elif tool_name == "register_analysis_plan":', start)
    block = src[start:end]
    assert "self._bind_gene_set(" in block
    bind = inspect.getsource(bio_mod.BioinformaticsAgent._bind_gene_set)
    assert "self.sandbox.execute" in bind
    assert "skip_safety_check=True" in bind


def test_a_bind_the_kernel_refused_reports_no_variable():
    """The kernel says no in the result far more often than it raises.

    A dead or restarting kernel, an execution timeout and a MemoryError on a
    large panel all come back as an ordinary result carrying .error. Returning
    the variable name anyway makes the caller suppress the inline gene list, so
    the next cell dies on NameError or the model retypes the panel from memory
    -- the fetched-panel-becomes-remembered-panel failure this helper exists to
    prevent.
    """
    import asyncio
    from types import SimpleNamespace

    from backend.utils.sandbox_parts.shared import CodeExecutionResult

    class _Kernel:
        def __init__(self, result):
            self.result = result

        async def execute(self, code, **kwargs):
            return self.result

    refused = CodeExecutionResult(
        success=False, output="",
        error="Kernel died during execution and restart failed.")
    bind = bio_mod.BioinformaticsAgent._bind_gene_set

    agent = SimpleNamespace(session_id="t", sandbox=_Kernel(refused))
    assert asyncio.run(bind(agent, "HALLMARK_X", ["TP53", "MYC"], "gene set")) is None

    agent = SimpleNamespace(
        session_id="t", sandbox=_Kernel(CodeExecutionResult(success=True)))
    assert asyncio.run(
        bind(agent, "HALLMARK_X", ["TP53", "MYC"], "gene set")
    ) == "gene_set_HALLMARK_X"


def test_a_refused_bind_falls_back_to_the_inline_gene_list():
    """None is the caller's signal to inline the genes, and it must be used."""
    src = inspect.getsource(bio_mod)
    start = src.index('elif tool_name == "fetch_gene_set":')
    end = src.index('elif tool_name == "register_analysis_plan":', start)
    block = src[start:end]
    assert "if injected:" in block and "Genes: " in block, (
        "the inline list is the documented fallback for a bind that did not take"
    )


def test_resolver_ships_no_curated_biology():
    """The resource is configuration; ScAI must not bundle gene lists."""
    import re

    from backend.services import gene_set_resolver

    src = inspect.getsource(gene_set_resolver).lower()
    for term in ("ferroptos", "apoptos", "cuproptos", "hallmark_", "gobp_"):
        assert term not in src
    for word in ("liver", "cancer", "tumor"):
        assert not re.search(rf"\b{word}\b", src)
