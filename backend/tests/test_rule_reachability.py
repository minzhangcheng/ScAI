"""Every rule must actually be able to reach the model.

The corpus header promises two things: that a `gate` is refused in code, and
that `guidance` is retrieved on demand. An independent review found neither was
fully true -- two gates had no implementation at all, and seven guidance rules
were unreachable because the only runtime path was a gate citing a rule id, and
no gate cites a guidance rule.

A rule the model never sees is worse than no rule: it makes the corpus look
larger than the system's real behaviour, and the gap is invisible from inside.
These tests close it.
"""
import inspect
import re
from pathlib import Path

import pytest

import backend.agents.bioinformatics_agent as bio_mod
from backend.agents.bioinformatics import executor_collaboration
from backend.services import statistical_gates
from backend.services.method_rules import (
    ENFORCEMENT_ADVISORY, ENFORCEMENT_GATE, ENFORCEMENT_GUIDANCE,
    ENFORCEMENT_VALUES, all_rules, gates,
)

_REPO = Path(__file__).resolve().parent.parent.parent
_BACKEND = _REPO / "backend"


def _production_sources() -> str:
    """Concatenated backend source, excluding tests."""
    chunks = []
    for path in _BACKEND.rglob("*.py"):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


_SOURCES = _production_sources()


#: Symbols that only appear where a call is actually refused. A rule id
#: interpolated into a message string reaches none of them.
_REFUSAL_SYMBOLS = ("GateFinding(", "rejected:", "_reject_count")


def _citation_windows(rule_id: str) -> list[str]:
    """Source around each place a rule id appears, one window per occurrence."""
    windows = []
    for path in _BACKEND.rglob("*.py"):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        start = 0
        while True:
            idx = text.find(rule_id, start)
            if idx == -1:
                break
            windows.append(text[max(0, idx - 700):idx + 700])
            start = idx + 1
    return windows


@pytest.mark.parametrize("rule", [r for r in all_rules() if r["enforcement"] == ENFORCEMENT_GATE],
                         ids=lambda r: r["id"])
def test_every_gate_is_cited_by_production_code(rule):
    """A gate that no code enforces is a false claim about the system."""
    assert rule["id"] in _SOURCES, (
        f"rule {rule['id']!r} is declared enforcement=gate but no non-test "
        f"backend file references it. Either implement the check, or reclassify "
        f"it as guidance -- the corpus header promises gates are refused in code."
    )


@pytest.mark.parametrize("rule", [r for r in all_rules() if r["enforcement"] == ENFORCEMENT_GATE],
                         ids=lambda r: r["id"])
def test_every_gate_reaches_the_model_through_a_refusal(rule):
    """enforcement=gate must mean the call is REFUSED, not annotated.

    Mentioning the id is not enforcing it. Two rules were interpolated into
    advisory notes -- one onto a plan-drift message, one onto a power caveat --
    and the weaker assertion passed them both. A rule that only advises belongs
    on the advisory or guidance channel, where the corpus now says so.
    """
    windows = _citation_windows(rule["id"])
    assert windows, f"{rule['id']} is not referenced in production code at all"
    assert any(any(sym in w for sym in _REFUSAL_SYMBOLS) for w in windows), (
        f"{rule['id']} is declared enforcement=gate, but every place it appears "
        f"is a message string rather than a refusal. Reclassify it as advisory "
        f"or guidance, or implement a check that rejects."
    )


@pytest.mark.parametrize(
    "rule", [r for r in all_rules() if r["enforcement"] == ENFORCEMENT_ADVISORY],
    ids=lambda r: r["id"])
def test_advisory_rules_are_emitted_without_blocking(rule):
    """An advisory must reach the model, and must not refuse the call."""
    from backend.services import statistical_gates

    src = inspect.getsource(statistical_gates)
    assert rule["id"] in src, f"{rule['id']} is advisory but no check emits it"
    assert 'severity="advise"' in src, (
        "an advisory rule requires at least one check emitting severity='advise'"
    )


def test_enforcement_vocabulary_is_closed():
    """Only the documented values may appear."""
    seen = {r["enforcement"] for r in all_rules()}
    assert seen <= set(ENFORCEMENT_VALUES), (
        f"undocumented enforcement values: {seen - set(ENFORCEMENT_VALUES)}"
    )


def test_corpus_header_documents_every_enforcement_value():
    """The header is the contract these tests check against."""
    from backend.services.method_rules import _RULES_PATH

    header = _RULES_PATH.read_text(encoding="utf-8").split("meta:")[0]
    for value in {r["enforcement"] for r in all_rules()}:
        assert value in header, f"enforcement value {value!r} is undocumented"


def test_guidance_is_reachable_through_a_retrieval_path():
    """Guidance must have a runtime route, or it is documentation nobody reads."""
    assert "search_rules" in _SOURCES, (
        "no production code calls search_rules, so guidance rules can never "
        "reach the model"
    )


def test_method_question_is_exposed_on_a_tool():
    """The retrieval route has to be callable by the agent, not just importable."""
    schema = next(t["function"]["parameters"] for t in bio_mod.TOOLS
                  if t["function"]["name"] == "lookup_documentation")
    assert "method_question" in schema["properties"]


def test_method_question_alone_is_a_valid_call():
    """Asking a method question must not require inventing a function path."""
    source = inspect.getsource(executor_collaboration.handle_lookup_documentation)
    assert "method_question and not function_path" in source


def test_gates_cite_only_existing_rules():
    """A gate citing a missing id would emit an empty explanation."""
    from backend.services.method_rules import get_rule

    src = inspect.getsource(statistical_gates)
    cited = set(re.findall(r'rule_id="([a-z0-9-]+)"', src))
    cited |= set(re.findall(r'also_see=\("([a-z0-9-]+)"', src))
    assert cited, "no rule ids found in the gate module"
    for rule_id in cited:
        assert get_rule(rule_id) is not None, f"gate cites unknown rule {rule_id!r}"


def test_gate_count_matches_the_corpus():
    """Sanity: the helper and the enforcement field cannot disagree."""
    assert {r["id"] for r in gates()} == {
        r["id"] for r in all_rules() if r["enforcement"] == ENFORCEMENT_GATE
    }


# ---------------------------------------------- the specific review findings

def test_zero_match_panel_is_the_most_underpowered_case():
    """A 200-gene panel with zero genes measured used to bypass the gate entirely."""
    from backend.services.gene_set_provenance import assess_panel_power

    power = assess_panel_power(200, 0)
    assert power["underpowered"], (
        "a panel whose genes are entirely absent from the matrix is the most "
        "underpowered case, not an exempt one"
    )
    assert not power["supports_negative_claim"]


def test_no_declared_panel_is_not_flagged_as_underpowered():
    """The distinction the fix has to preserve: no panel != empty panel."""
    from backend.services.gene_set_provenance import assess_panel_power

    power = assess_panel_power(0, None)
    assert not power["underpowered"]
    assert not power["supports_negative_claim"]


def test_zero_match_negative_claim_is_refused():
    from types import SimpleNamespace

    from backend.services.gene_set_provenance import assess_panel_power
    from backend.services.statistical_gates import check_underpowered_negative_claims

    score = SimpleNamespace(score_name="panel_zero", method="gseapy.prerank",
                            plan_id="p", power=assess_panel_power(200, 0))
    out = check_underpowered_negative_claims(
        ["panel_zero was not significant, arguing against this mechanism"], [score])
    assert len(out) == 1


def test_zero_match_caveat_names_the_likely_cause():
    """Zero overlap is nearly always an organism or naming mismatch, not biology."""
    from backend.services.gene_set_provenance import assess_panel_power, power_caveat

    text = power_caveat(assess_panel_power(200, 0))
    assert "NONE" in text
    assert "organism" in text or "naming" in text


def test_cell_level_inference_is_enforced():
    """biological-replicate-is-the-unit was a gate with no implementation."""
    from backend.services.statistical_gates import check_cell_level_inference

    out = check_cell_level_inference(
        ["a per-cell test shows the condition effect is significant"], {"A": 3, "B": 3})
    assert len(out) == 1
    assert out[0].rule_id == "biological-replicate-is-the-unit"


def test_pseudobulk_wording_is_not_flagged_as_cell_level():
    """The honest analysis reads almost identically; do not refuse it."""
    from backend.services.statistical_gates import check_cell_level_inference

    out = check_cell_level_inference(
        ["pseudobulk aggregation per sample shows a significant condition effect"],
        {"A": 3, "B": 3})
    assert out == []


def test_permutation_rule_is_not_conditioned_on_sample_size():
    """Gene-permutation anti-conservativeness is independent of n.

    The first draft scoped the rule to designs with few samples per group, which
    waved through exactly the same defect at larger n.
    """
    from backend.services.method_rules import get_rule

    rule = get_rule("gene-permutation-is-anticonservative")
    applies = rule["applies_when"].lower()
    assert "any preranked" in applies or "at any sample size" in applies
    assert "fewer than" not in applies


def test_min_size_rule_states_the_real_default():
    """The first draft said the default was not 10, implying it should be lowered."""
    from backend.services.method_rules import get_rule

    python = get_rule("minimum-gene-set-size")["python"]
    assert "min_size=15" in python
    assert "max_size=500" in python
    assert "not lower" in python.lower()


def test_post_clustering_rule_cites_selective_inference_work():
    from backend.services.method_rules import get_rule

    citations = " ".join(get_rule("post-clustering-inference-is-invalid")["citations"])
    assert "Grabski" in citations or "Neufeld" in citations
    assert "Zappia" not in citations, (
        "clustree is a resolution-visualisation paper, not selective inference"
    )


def test_trajectory_rule_cites_velocity_work():
    from backend.services.method_rules import get_rule

    citations = " ".join(get_rule("trajectory-direction-needs-external-evidence")["citations"])
    assert "Squair" not in citations, (
        "Squair 2021 is about pseudoreplication in single-cell DE, not trajectories"
    )
    assert "Bergen" in citations or "Saelens" in citations


def test_ora_universe_rule_exists_and_gives_a_working_route():
    """The rule must tell the model what to DO, not just what the null is.

    An earlier draft justified this rule with a claim that gseapy silently
    ignores the background argument for named libraries. That was true of
    gseapy <1.1 and is stale for the installed 1.3.0, whose Enrichr client
    forwards a supplied background to the Speedrichr endpoints. The rule was
    corrected; this test pins the working route rather than the old rationale.
    """
    from backend.services.method_rules import get_rule

    rule = get_rule("ora-universe-must-be-the-measured-genes")
    assert rule is not None
    assert rule["enforcement"] == ENFORCEMENT_GATE
    python = rule["python"]
    assert "get_library" in python, "the rule must name the controllable route"
    assert "background=" in python
    assert "IGNORES" not in python, (
        "the stale background is ignored claim must not come back"
    )


def test_composition_rule_admits_the_missing_python_tooling():
    from backend.services.method_rules import get_rule

    python = get_rule("composition-lives-on-a-simplex")["python"]
    assert "scCODA" in python
    assert "not installed" in python or "absent" in python
