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


from backend.services.method_rules import all_rules

_RULES = all_rules()
_BY_ID = {r["id"]: r for r in _RULES}





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
