"""A gene panel must be frozen before its result is visible.

Curating a panel is legitimate; revising it after seeing the enrichment result
is selection on the outcome, and the final report cannot tell the two apart
unless the plan is on record. These tests pin the freeze, the drift detection,
and the typed edit vocabulary.

See backend/services/analysis_plan for what the mechanism does and does not
prove -- step ordering shows the plan came first, not that no intermediate
result was seen.
"""
import inspect

import pytest

import backend.agents.bioinformatics_agent as bio_mod
from backend.models.schemas import AnalysisState, ScoringPlan
from backend.services.analysis_plan import (
    EDIT_OPERATIONS,
    HIGH_RISK_OPERATIONS,
    canonical_panel,
    describe_drift,
    diff_panels,
    panel_hash,
    summarize_edits,
    validate_edit_operations,
)


# ------------------------------------------------------------------ hashing

def test_panel_hash_ignores_order_and_duplicates():
    """Reordering a panel is not tampering; membership change is."""
    assert panel_hash(["B", "A", "C"]) == panel_hash(["A", "B", "C", "A"])


def test_panel_hash_ignores_symbol_casing():
    """Casing is an organism convention, not panel content."""
    assert panel_hash(["Actb", "Gapdh"]) == panel_hash(["ACTB", "GAPDH"])


def test_panel_hash_changes_on_membership_change():
    assert panel_hash(["A", "B"]) != panel_hash(["A", "B", "C"])


def test_empty_panel_hashes_to_empty():
    assert panel_hash([]) == ""
    assert panel_hash(None) == ""


def test_canonical_panel_preserves_first_seen_casing():
    assert canonical_panel([" Actb ", "actb", "Gapdh"]) == ["Actb", "Gapdh"]


# --------------------------------------------------------------------- drift

def test_identical_panel_has_no_drift():
    diff = diff_panels(["A", "B"], ["B", "A"])
    assert diff["matches"]
    assert describe_drift(diff) == ""


def test_added_gene_is_reported():
    diff = diff_panels(["A", "B"], ["A", "B", "C"])
    assert not diff["matches"]
    assert diff["added"] == ["C"]
    assert "added" in describe_drift(diff)


def test_removed_gene_is_reported():
    """Dropping genes after seeing the result is the cherry-pick this catches."""
    diff = diff_panels(["A", "B", "C"], ["A", "B"])
    assert not diff["matches"]
    assert diff["removed"] == ["C"]
    assert "removed" in describe_drift(diff)


# ------------------------------------------------------------ edit operations

def test_low_risk_edit_needs_no_rationale():
    ops, err = validate_edit_operations([{"operation": "intersect"}])
    assert err is None
    assert ops[0]["risk"] == "low"


def test_partition_requires_rationale_but_is_low_risk():
    """Splitting a mixed-direction set is the high-value edit, not a suspect one."""
    assert EDIT_OPERATIONS["partition"]["risk"] == "low"
    ops, err = validate_edit_operations([{"operation": "partition"}])
    assert err is not None and "rationale" in err


@pytest.mark.parametrize("op", HIGH_RISK_OPERATIONS)
def test_high_risk_edits_require_rationale(op):
    ops, err = validate_edit_operations([{"operation": op, "genes": ["A"]}])
    assert err is not None
    assert "rationale" in err


def test_high_risk_edit_accepted_with_rationale():
    ops, err = validate_edit_operations([
        {"operation": "remove", "genes": ["A"],
         "rationale": "gene is not measured on this platform at all"}
    ])
    assert err is None
    assert ops[0]["risk"] == "high"


def test_unknown_operation_rejected():
    ops, err = validate_edit_operations([{"operation": "replace_everything"}])
    assert err is not None
    assert "must be one of" in err


def test_removing_genes_is_higher_risk_than_adding_curated_ones():
    """The asymmetry is deliberate: removal is almost always toward the hypothesis."""
    assert EDIT_OPERATIONS["remove"]["risk"] == "high"
    assert EDIT_OPERATIONS["union"]["risk"] == "low"


def test_summarize_edits_flags_high_risk():
    text = summarize_edits([
        {"operation": "remove", "risk": "high", "genes": ["A"]},
        {"operation": "union", "risk": "low", "genes": []},
    ])
    assert "high-risk" in text and "remove" in text


# ------------------------------------------------------------------- wiring

def test_register_analysis_plan_is_a_registered_tool():
    names = {t["function"]["name"] for t in bio_mod.TOOLS}
    assert "register_analysis_plan" in names
    assert "register_analysis_plan" in bio_mod.KNOWN_TOOL_NAMES


def test_register_analysis_plan_has_a_dispatch_branch():
    src = inspect.getsource(bio_mod)
    assert 'elif tool_name == "register_analysis_plan":' in src


def test_every_tool_is_named_in_the_system_prompt():
    """The enumeration had silently drifted to 8 of 10 tools.

    A tool the prompt never mentions is a tool the model will not call, so the
    enumeration has to stay in sync with TOOLS.
    """
    src = inspect.getsource(bio_mod)
    missing = [t["function"]["name"] for t in bio_mod.TOOLS
               if t["function"]["name"] not in src]
    assert not missing, f"tools absent from the agent module text: {missing}"


def test_register_score_requires_plan_id_for_panel_scores():
    import re

    src = inspect.getsource(bio_mod)
    start = src.index('elif tool_name == "register_score":')
    end = src.index('elif tool_name == "register_mechanism_programs":', start)
    block = src[start:end]
    # Long messages are written as implicitly concatenated literals across
    # lines, so match the joined text rather than the raw source; otherwise the
    # test breaks on reflow instead of on behaviour.
    joined = re.sub(r'"\s*\n\s*(?:f?)"', "", block)
    joined = re.sub(r"\s+", " ", joined)
    assert "plan_id" in block
    assert "must cite the plan_id" in joined
    assert "diff_panels" in block


def test_plan_schema_requires_the_panel_and_its_source():
    schema = next(t["function"]["parameters"] for t in bio_mod.TOOLS
                  if t["function"]["name"] == "register_analysis_plan")
    for field in ("score_name", "method", "panel_genes", "panel_source"):
        assert field in schema["required"]
    # Declared but optional: absence is reported back rather than blocking.
    for field in ("falsifier", "testing_family", "edit_operations", "baseline_genes"):
        assert field in schema["properties"]


def test_state_holds_frozen_plans():
    state = AnalysisState(id="s", user_id="u")
    assert state.scoring_plans == []
    plan = ScoringPlan(score_name="x", method="m", panel_genes=["A"],
                        panel_hash=panel_hash(["A"]))
    state.scoring_plans.append(plan)
    assert state.scoring_plans[0].panel_hash == panel_hash(["A"])


# ------------------------------------------------------------ neutrality

def test_plan_module_names_no_specific_biology():
    import re

    from backend.services import analysis_plan

    src = inspect.getsource(analysis_plan).lower()
    for term in ("ferroptos", "apoptos", "cuproptos", "necroptos", "nrf2",
                 "hepatocyte", "msigdb"):
        assert term not in src
    for word in ("liver", "mouse", "cancer", "tumor"):
        assert not re.search(rf"\b{word}\b", src)
