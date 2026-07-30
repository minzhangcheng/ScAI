"""Citations must support the claim they are attached to, and APIs must exist.

Mis-citation has now recurred across three rounds of review: a clustering-
visualisation paper cited for selective inference, a single-cell DE paper cited
for trajectory direction, and an independent-filtering paper cited on four rules
about entirely different topics. The pattern is always the same -- a real paper,
correctly formatted, attached to a claim it does not make.

These tests cannot judge whether a citation is apt; only a reader can, against
the upstream source. What they CAN pin is the specific corrections already made,
plus every claim that IS mechanically checkable: whether a named API exists,
whether a documented call would raise, whether a stated default is attributed to
the function it belongs to.

A structural detector for mis-citation was attempted and removed. The idea was
to flag any paper cited across several unrelated source skills. It fails in both
directions: the Bourgon 2010 misattribution spanned only two skills, so the
heuristic would have missed the very defect that motivated it, while a paper
about reporting standards in enrichment analysis legitimately spans three. The
only reliable check is against the upstream reference lists, which this project
deliberately does not vendor -- so that check belongs to review, not CI, and the
pins below record what review found.
"""
import inspect
import re

import pytest

from backend.services.method_rules import all_rules, corpus_meta

_RULES = all_rules()
_BY_ID = {r["id"]: r for r in _RULES}


# --------------------------------------------- corrections already established

def test_bourgon_stays_only_on_the_two_rules_it_supports():
    """Pin the specific correction, since it was wrong on four rules."""
    expected = {"adjusted-p-na-is-a-flag-not-a-gap",
                "filter-must-be-independent-under-the-null"}
    actual = {
        rule["id"] for rule in _RULES
        if any("bourgon" in str(c).lower() for c in (rule.get("citations") or []))
    }
    assert actual == expected, (
        f"Bourgon 2010 is about independent filtering before FDR adjustment. It "
        f"belongs on {sorted(expected)} and nowhere else; found on {sorted(actual)}"
    )


def test_uncited_rules_say_so_explicitly():
    """A rule with no upstream support must not imply it has one.

    read-the-set-before-scoring-it makes a claim bioSkills never makes. Keeping
    the rule is fine; attributing it upstream is not.
    """
    for rule in _RULES:
        citations = rule.get("citations") or []
        assert citations, f"{rule['id']} has no citations field at all"
        if any("no upstream citation" in str(c).lower() for c in citations):
            joined = " ".join(str(c) for c in citations).lower()
            assert "scai" in joined or "own guidance" in joined, (
                f"{rule['id']} declares no upstream source but does not say whose "
                f"guidance it is"
            )


# ----------------------------------------------------------------- API claims

_API_PATTERN = re.compile(r"\b((?:sc|scanpy|gseapy|gp)\.[a-z_]+\.[a-z_0-9]+)\b")


def test_python_fields_reference_only_real_apis():
    """A corpus whose reason for existing is Python translation must compile.

    scanpy.pp.neighbours (British spelling) shipped in one rule and does not
    exist; the risk is that such a name reads as authoritative.
    """
    import gseapy  # noqa: F401
    import scanpy  # noqa: F401

    modules = {"sc": scanpy, "scanpy": scanpy, "gp": gseapy, "gseapy": gseapy}
    missing = []
    for rule in _RULES:
        for dotted in _API_PATTERN.findall(str(rule.get("python", ""))):
            root, *rest = dotted.split(".")
            obj = modules.get(root)
            if obj is None:
                continue
            for part in rest:
                obj = getattr(obj, part, None)
                if obj is None:
                    missing.append((rule["id"], dotted))
                    break
    assert not missing, f"python fields name APIs that do not exist: {missing}"


def test_lfc_shrink_is_documented_with_its_required_argument():
    """The documented call was argument-less; the real signature requires coeff."""
    from pydeseq2.ds import DeseqStats

    signature = inspect.signature(DeseqStats.lfc_shrink)
    assert "coeff" in signature.parameters
    assert signature.parameters["coeff"].default is inspect.Parameter.empty, (
        "coeff is required, so any rule showing lfc_shrink() must show the argument"
    )
    for rule in _RULES:
        text = str(rule.get("python", ""))
        if "lfc_shrink" not in text:
            continue
        assert "lfc_shrink()" not in text, (
            f"{rule['id']} documents an argument-less lfc_shrink call, which raises "
            f"TypeError"
        )


def test_no_rule_claims_a_default_without_naming_the_function():
    """Defaults differ per entry point, so an unqualified default misleads.

    gseapy.prerank caps at 500 and gseapy.gsva at 1000; stating "the default is
    500" without saying which call is read as a general fact.
    """
    for rule in _RULES:
        text = str(rule.get("python", ""))
        for match in re.finditer(r"(min_size|max_size|weight)\s*=\s*[\d.]+", text):
            window = text[max(0, match.start() - 120):match.start()]
            assert re.search(r"(gseapy|gp|sc|scanpy)\.\w+", window), (
                f"{rule['id']} states {match.group(0)!r} without naming the function "
                f"it belongs to"
            )


# ------------------------------------------------------------------- contract

def test_corpus_still_claims_upstream_citations():
    """If this promise is dropped the tests above lose their basis."""
    assert corpus_meta().get("source") == "bioSkills"
