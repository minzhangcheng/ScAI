# Freezing a scoring plan before its result is known.
"""Pre-registration for gene-panel scoring.

The problem
-----------
Letting the model curate a gene panel is genuinely useful -- an ontology term
often mixes drivers and suppressors of the same process, and splitting it is a
judgement a curated database cannot make. But if the panel can be revised
*after* the enrichment result is visible, the revision is no longer a judgement:
it is selection on the outcome. The same run then reports whichever panel gave
the desired answer, with nothing in the record showing that other panels were
tried. That is the garden of forking paths, and no amount of downstream
statistics repairs it.

What this module enforces
-------------------------
A plan freezes ``(score_name, method, panel)`` and returns a ``plan_id``.
``register_score`` must cite one, and the handler re-hashes the panel it was
actually given: if it differs from the frozen one, the score is recorded with an
explicit, itemised diff rather than silently accepted.

What it does NOT prove
----------------------
Step ordering shows the plan was declared before the score was registered. It
does *not* prove no intermediate result was seen -- the agent runs arbitrary
code in between, and nothing here can tell which execution produced an
enrichment statistic. The honest claim is narrower and still worth having: the
panel is fixed in advance, and any later change to it is visible. The count of
executions between plan and score is recorded so a reader can judge.

Domain neutrality
-----------------
Nothing here names an organism, tissue, disease or pathway, and nothing may be
added that does. ScAI is a general-purpose platform (see CLAUDE.md).
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional, Sequence

#: Typed edits a curator may apply to a database panel. Free-form replacement is
#: deliberately absent: every edit has to name its shape so the report can say
#: what was done. Risk is about how easily the edit can bend a result toward a
#: hypothesis, not about how hard it is to perform.
EDIT_OPERATIONS = {
    "partition": {
        "risk": "low",
        "summary": "Split a set into direction-coherent subsets scored separately.",
        "rationale_required": True,
    },
    "union": {
        "risk": "low",
        "summary": "Combine two or more versioned sets.",
        "rationale_required": False,
    },
    "intersect": {
        "risk": "low",
        "summary": "Keep only genes shared by two or more versioned sets.",
        "rationale_required": False,
    },
    "restrict_to_measured": {
        "risk": "low",
        "summary": "Drop genes absent from the analysed matrix.",
        "rationale_required": False,
    },
    "remove": {
        "risk": "high",
        "summary": "Drop specific genes by hand.",
        "rationale_required": True,
    },
    "add_bespoke": {
        "risk": "high",
        "summary": "Add genes not present in any cited versioned set.",
        "rationale_required": True,
    },
}

HIGH_RISK_OPERATIONS = tuple(
    name for name, spec in EDIT_OPERATIONS.items() if spec["risk"] == "high"
)

#: Floor on the rationale a rationale-requiring edit must carry. Long enough
#: that a reason has to be given rather than a word typed.
MIN_RATIONALE_CHARS = 20


def canonical_panel(genes: Optional[Iterable[str]]) -> list[str]:
    """Order- and duplicate-insensitive form of a gene panel.

    Two panels differing only in ordering are the same panel; hashing the sorted
    unique set means a reordering does not read as tampering, while a genuine
    membership change always does. Comparison is case-insensitive because symbol
    casing is an organism convention, not panel content.
    """
    if not genes:
        return []
    seen: dict[str, str] = {}
    for gene in genes:
        text = str(gene).strip()
        if not text:
            continue
        seen.setdefault(text.lower(), text)
    return [seen[k] for k in sorted(seen)]


def panel_hash(genes: Optional[Iterable[str]]) -> str:
    """Stable digest of a panel's membership."""
    canonical = canonical_panel(genes)
    if not canonical:
        return ""
    joined = "\n".join(g.lower() for g in canonical)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def diff_panels(frozen: Sequence[str], actual: Sequence[str]) -> dict:
    """Itemise how a scored panel departs from its frozen plan."""
    frozen_map = {g.lower(): g for g in canonical_panel(frozen)}
    actual_map = {g.lower(): g for g in canonical_panel(actual)}
    added = [actual_map[k] for k in sorted(set(actual_map) - set(frozen_map))]
    removed = [frozen_map[k] for k in sorted(set(frozen_map) - set(actual_map))]
    return {
        "matches": not added and not removed,
        "added": added,
        "removed": removed,
        "n_frozen": len(frozen_map),
        "n_actual": len(actual_map),
    }


def describe_drift(diff: dict) -> str:
    """One-line description of a plan/score panel mismatch, or '' when clean."""
    if diff.get("matches"):
        return ""
    bits = []
    if diff.get("added"):
        added = diff["added"]
        shown = ", ".join(added[:8]) + ("..." if len(added) > 8 else "")
        bits.append(f"{len(added)} gene(s) added after the plan was frozen ({shown})")
    if diff.get("removed"):
        removed = diff["removed"]
        shown = ", ".join(removed[:8]) + ("..." if len(removed) > 8 else "")
        bits.append(f"{len(removed)} gene(s) removed after the plan was frozen ({shown})")
    return "; ".join(bits)


def validate_edit_operations(operations) -> tuple[list[dict], Optional[str]]:
    """Check a list of typed panel edits.

    Returns ``(normalised, error)``. ``error`` is a corrective message for the
    model; on error the normalised list is empty.
    """
    if operations in (None, ""):
        return [], None
    if not isinstance(operations, list):
        return [], "edit_operations must be a list of objects."

    normalised: list[dict] = []
    for i, raw in enumerate(operations):
        if not isinstance(raw, dict):
            return [], f"edit_operations[{i}] must be an object."
        op = str(raw.get("operation", "")).strip().lower()
        if op not in EDIT_OPERATIONS:
            return [], (
                f"edit_operations[{i}].operation must be one of "
                f"{sorted(EDIT_OPERATIONS)}; got {op!r}."
            )
        spec = EDIT_OPERATIONS[op]
        rationale = str(raw.get("rationale", "")).strip()
        if spec["rationale_required"] and len(rationale) < MIN_RATIONALE_CHARS:
            return [], (
                f"edit_operations[{i}] is a {spec['risk']}-risk '{op}' and needs a "
                f"rationale of at least {MIN_RATIONALE_CHARS} characters saying why. "
                f"Removing genes from, or adding genes to, a versioned set moves "
                f"the result toward whatever the edit was chosen to show."
            )
        genes = [str(g).strip() for g in (raw.get("genes") or []) if str(g).strip()]
        normalised.append({
            "operation": op,
            "risk": spec["risk"],
            "rationale": rationale,
            "genes": genes[:200],
        })
    return normalised, None


def summarize_edits(operations: Sequence[dict]) -> str:
    """Short human-readable summary of applied edits, or '' when none."""
    if not operations:
        return ""
    parts = []
    for op in operations:
        label = op.get("operation", "?")
        n = len(op.get("genes") or [])
        parts.append(f"{label}({n})" if n else label)
    high = [op["operation"] for op in operations if op.get("risk") == "high"]
    text = "edits: " + ", ".join(parts)
    if high:
        text += f" -- includes high-risk {', '.join(sorted(set(high)))}"
    return text
