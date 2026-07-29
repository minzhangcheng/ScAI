"""Resolve the full payload behind a selected research direction.

Direction selection stores only ``{direction, notes, is_custom}``.  Everything
the exploration round derived about that direction — its open uncertainties,
competing explanations, the follow-up analyses it asked for, the entities it
implicated — stays behind in
``exploration_report.research_direction_candidates`` and never reaches the
round that is supposed to act on it.

Rather than widen the stored selection (which would not help sessions already
in flight), resolve the candidate on demand by matching the direction text.
The helpers here are domain-neutral: they read whatever the candidate carries
and never assume a subject area.
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Candidate fields that state what this direction still has to establish, in
# the order a planner should see them. Values may be strings or lists.
REQUIREMENT_FIELDS: Sequence[str] = (
    "key_uncertainties",
    "suggested_validation_analysis",
    "alternative_hypotheses",
)

# Nested under ``validation_plan``.
VALIDATION_PLAN_FIELDS: Sequence[str] = ("next_analysis", "decision_criteria")

# Entities the exploration round already implicated; a plan that no longer
# mentions any of them has drifted off the direction it claims to serve.
ENTITY_FIELDS: Sequence[str] = ("key_genes", "key_pathways", "key_cell_types")


def normalize_direction_text(direction: Any) -> str:
    """Whitespace- and case-insensitive key for direction comparison."""
    return " ".join(str(direction or "").strip().lower().split())


def find_direction_candidate(
    candidates: Iterable[Dict[str, Any]],
    direction: str,
) -> Optional[Dict[str, Any]]:
    """The candidate whose ``direction`` matches ``direction``.

    Falls back to a containment match because a stored selection is sometimes a
    lightly edited copy of the candidate text.
    """
    key = normalize_direction_text(direction)
    if not key:
        return None
    pool = [c for c in (candidates or []) if isinstance(c, dict)]
    for candidate in pool:
        if normalize_direction_text(candidate.get("direction")) == key:
            return candidate
    for candidate in pool:
        other = normalize_direction_text(candidate.get("direction"))
        if other and (other in key or key in other):
            return candidate
    return None


def candidate_for_state(state: Any, direction: str = "") -> Optional[Dict[str, Any]]:
    """Resolve the active round's direction candidate from an AnalysisState."""
    report = getattr(state, "exploration_report", None)
    if report is None:
        return None
    target = direction or getattr(state, "current_round_direction", "") or ""
    candidates = list(getattr(report, "research_direction_candidates", None) or [])
    candidates += list(getattr(report, "raw_direction_candidates", None) or [])
    return find_direction_candidate(candidates, target)


def _as_items(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        return []
    if isinstance(value, Iterable):
        items = []
        for entry in value:
            if isinstance(entry, str) and entry.strip():
                items.append(entry.strip())
        return items
    return []


def direction_requirements(candidate: Optional[Dict[str, Any]]) -> List[str]:
    """Statements this direction must still resolve, de-duplicated in order."""
    if not isinstance(candidate, dict):
        return []
    items: List[str] = []
    for field in REQUIREMENT_FIELDS:
        items.extend(_as_items(candidate.get(field)))
    plan = candidate.get("validation_plan")
    if isinstance(plan, dict):
        for field in VALIDATION_PLAN_FIELDS:
            items.extend(_as_items(plan.get(field)))
    seen = set()
    unique = []
    for item in items:
        marker = item.lower()
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(item)
    return unique


def direction_entities(candidate: Optional[Dict[str, Any]]) -> List[str]:
    """Genes, pathways, and cell types the exploration round implicated."""
    if not isinstance(candidate, dict):
        return []
    items: List[str] = []
    for field in ENTITY_FIELDS:
        items.extend(_as_items(candidate.get(field)))
    seen = set()
    unique = []
    for item in items:
        marker = item.lower()
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(item)
    return unique


def build_direction_context_block(
    candidate: Optional[Dict[str, Any]],
    *,
    max_items: int = 12,
    max_entities: int = 40,
) -> str:
    """Prompt block carrying the direction's own unresolved questions.

    Empty string when no candidate resolves, so callers can interpolate it
    unconditionally.
    """
    if not isinstance(candidate, dict):
        return ""
    requirements = direction_requirements(candidate)[:max_items]
    entities = direction_entities(candidate)[:max_entities]
    hypothesis = str(candidate.get("main_hypothesis") or "").strip()
    if not (requirements or entities or hypothesis):
        return ""

    lines = ["\n## Direction Payload (from the exploration round that raised this direction)"]
    if hypothesis:
        lines.append(f"Main hypothesis: {hypothesis}")
    if requirements:
        lines.append(
            "Open questions and follow-up analyses this direction was selected to resolve. "
            "Every one of them must be addressed by a step, or explicitly recorded as out of scope:"
        )
        lines.extend(f"- {item}" for item in requirements)
    if entities:
        lines.append(
            "Entities the exploration round implicated (carry these into the relevant steps "
            "rather than replacing them with generic category names):"
        )
        lines.append("- " + ", ".join(entities))
    return "\n".join(lines)
