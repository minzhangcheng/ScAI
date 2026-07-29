"""Resolve the full payload behind a selected research direction.

The entries in ``user_chosen_directions`` carry only ``{direction, notes,
is_custom}``. Everything the exploration round derived about a direction — its
open uncertainties, competing explanations, the follow-up analyses it asked
for, the entities it implicated — stays behind in the exploration report and
never reaches the round that is supposed to act on it.

The link is not actually lost, though: a selection gate persists its frozen
candidate list and the index each selection came from, so resolution goes
through that record first and only falls back to matching the direction text.
Resolving on demand rather than widening the stored selection also means
sessions already in flight get the payload without a migration.

The helpers here are domain-neutral: they read whatever the candidate carries
and never assume a subject area.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import re

from ..models.schema_parts.enums import AgentType
from .formatting import as_list

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


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a model instance or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def candidate_from_selection_records(state: Any, direction: str) -> Optional[Dict[str, Any]]:
    """The candidate a selection gate already bound to this direction.

    Each gate event persists its frozen candidate list alongside the selections
    made against it, and a selection records the index of the candidate it came
    from. That link is exact, so prefer it over matching the direction text —
    text matching cannot tell two similarly-worded candidates apart, and a
    user-edited direction has no exact match at all.
    """
    key = normalize_direction_text(direction)
    if not key:
        return None
    for record in reversed(list(getattr(state, "direction_selection_records", None) or [])):
        candidates = list(_get(record, "candidates") or [])
        if not candidates:
            continue
        for selection in list(_get(record, "selections") or []):
            index = _get(selection, "candidate_index")
            if index is None or not 0 <= index < len(candidates):
                continue
            if normalize_direction_text(_get(selection, "direction")) != key:
                continue
            candidate = candidates[index]
            if isinstance(candidate, dict):
                return candidate
    return None


def candidate_for_state(state: Any, direction: str = "") -> Optional[Dict[str, Any]]:
    """Resolve the active round's direction candidate from an AnalysisState."""
    target = direction or getattr(state, "current_round_direction", "") or ""
    exact = candidate_from_selection_records(state, target)
    if exact is not None:
        return exact
    # Fallback for selections made before the index was recorded, and for the
    # PI-decide path, which writes bare {direction, notes} entries.
    report = getattr(state, "exploration_report", None)
    if report is None:
        return None
    candidates = list(getattr(report, "research_direction_candidates", None) or [])
    candidates += list(getattr(report, "raw_direction_candidates", None) or [])
    return find_direction_candidate(candidates, target)


def _as_items(value: Any) -> List[str]:
    """Non-empty strings from a scalar or list-shaped LLM value."""
    if isinstance(value, dict):
        return []
    return [item.strip() for item in as_list(value) if isinstance(item, str) and item.strip()]


def _dedupe(items: Iterable[str]) -> List[str]:
    """Case-insensitively unique, order preserved."""
    seen, unique = set(), []
    for item in items:
        text = str(item or "").strip()
        marker = text.lower()
        if not text or marker in seen:
            continue
        seen.add(marker)
        unique.append(text)
    return unique


def _collect(candidate: Optional[Dict[str, Any]], fields: Sequence[str]) -> List[str]:
    if not isinstance(candidate, dict):
        return []
    items: List[str] = []
    for name in fields:
        items.extend(_as_items(candidate.get(name)))
    return items


def direction_requirements(candidate: Optional[Dict[str, Any]]) -> List[str]:
    """Statements this direction must still resolve, de-duplicated in order."""
    items = _collect(candidate, REQUIREMENT_FIELDS)
    plan = (candidate or {}).get("validation_plan") if isinstance(candidate, dict) else None
    if isinstance(plan, dict):
        items.extend(_collect(plan, VALIDATION_PLAN_FIELDS))
    return _dedupe(items)


def direction_entities(candidate: Optional[Dict[str, Any]]) -> List[str]:
    """Genes, pathways, and cell types the exploration round implicated."""
    return _dedupe(_collect(candidate, ENTITY_FIELDS))


# --- resolved context, with a fallback for later rounds ---------------------

# Only the innovation rounds right after exploration analyse a direction that
# came from the exploration report. Later on, a direction is generated from the
# previous round's next steps or from an integration gate and has no candidate
# object at all — measured over 85 planning events across 22 sessions (the
# corpus on disk as of 2026-07-29), a candidate resolved for 45. Everything
# downstream that reads the payload therefore has to fall back to the round
# result, or it goes silent for nearly half of a long run.

_AGENT_PREFIX_RE = re.compile(r"^\s*\[([^\]]+)\]\s*")
# Next-step scopes a computational plan can actually execute. Items scoped to a
# wet-lab agent are carried as context but never checked for coverage: a plan
# cannot run a stain, so checking one manufactures a permanent false positive.
COMPUTATIONAL_SCOPES = frozenset({AgentType.BIOINFORMATICS.value})


@dataclass(frozen=True)
class DirectionContext:
    """What the active direction obliges this round to address.

    ``requirements`` are checked for coverage and shown to the planner.
    ``notes`` are shown only — open questions this round may legitimately be
    unable to close.  ``entities`` are named things the plan should keep
    naming.
    """
    requirements: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    hypothesis: str = ""
    source: str = "none"

    def __bool__(self) -> bool:
        return bool(self.requirements or self.entities or self.notes or self.hypothesis)


def _split_scoped_steps(steps: Iterable[Any]) -> Tuple[List[str], List[str]]:
    """Partition ``[Agent]``-prefixed next steps into executable and context."""
    executable: List[str] = []
    context: List[str] = []
    for raw in steps or []:
        text = str(raw or "").strip()
        if not text:
            continue
        match = _AGENT_PREFIX_RE.match(text)
        if not match:
            executable.append(text)
            continue
        body = text[match.end():].strip()
        if match.group(1).strip().lower() in COMPUTATIONAL_SCOPES:
            executable.append(body or text)
        else:
            context.append(text)
    return executable, context


def context_from_candidate(candidate: Dict[str, Any]) -> DirectionContext:
    return DirectionContext(
        requirements=direction_requirements(candidate),
        entities=direction_entities(candidate),
        hypothesis=str(candidate.get("main_hypothesis") or "").strip(),
        source="exploration_candidate",
    )


def context_from_round_result(result: Any) -> DirectionContext:
    """Derive the payload from the previous round when no candidate exists.

    A round result already states what the next round owes: its next steps are
    the asks, and its mechanism programs name the entities that were implicated
    and the evidence still missing.  Nothing new has to be generated for this,
    so it costs no LLM call and changes no agent contract.
    """
    if result is None:
        return DirectionContext()
    requirements, wet_lab = _split_scoped_steps(_get(result, "next_steps") or [])
    entities: List[str] = []
    notes: List[str] = list(wet_lab)
    for program in (_get(result, "data_driven_mechanism_programs") or []):
        label = str(_get(program, "label", "") or "").strip()
        if label:
            entities.append(label)
        entities.extend(_as_items(_get(program, "supporting_genes")))
        entities.extend(_as_items(_get(program, "supporting_pathways")))
        # Missing evidence is dominated by orthogonal assays and histology, so
        # it informs the planner without being held against the plan.
        notes.extend(_as_items(_get(program, "missing_evidence")))
        notes.extend(_as_items(_get(program, "alternative_explanations")))
    ctx = DirectionContext(
        requirements=_dedupe(requirements),
        entities=_dedupe(entities),
        notes=_dedupe(notes),
        source="previous_round",
    )
    return ctx if ctx else DirectionContext()


def resolve_direction_context(state: Any, direction: str = "") -> DirectionContext:
    """Best available payload for the active direction.

    Prefers the exploration candidate, which is richer and states the direction's
    own uncertainties; falls back to the previous round result so later rounds
    are not left without one.
    """
    candidate = candidate_for_state(state, direction)
    if isinstance(candidate, dict):
        ctx = context_from_candidate(candidate)
        if ctx:
            return ctx
    # Walk backwards rather than taking the last result outright: an
    # integration round produces a RoundResult with no next steps and no
    # mechanism programs, so keying on the immediate predecessor goes blank for
    # every direction round that follows one. Skipping an empty result loses
    # nothing, because there was nothing in it to lose.
    for result in reversed(list(getattr(state, "round_results", None) or [])):
        ctx = context_from_round_result(result)
        if ctx:
            return ctx
    return DirectionContext()


_SOURCE_HEADINGS = {
    "exploration_candidate": (
        "\n## Direction Payload (from the exploration round that raised this direction)"
    ),
    "previous_round": "\n## Direction Payload (carried from the previous round's results)",
}


MAX_BLOCK_REQUIREMENTS = 12
MAX_BLOCK_ENTITIES = 40
MAX_BLOCK_NOTES = 6


def build_direction_context_block(context: Any) -> str:
    """Prompt block carrying what the active direction still has to resolve.

    Accepts a :class:`DirectionContext` or a raw candidate dict. Returns an
    empty string when nothing resolves, so callers can interpolate it
    unconditionally.
    """
    if isinstance(context, dict):
        context = context_from_candidate(context)
    if not isinstance(context, DirectionContext) or not context:
        return ""

    lines = [_SOURCE_HEADINGS.get(context.source, _SOURCE_HEADINGS["previous_round"])]
    if context.hypothesis:
        lines.append(f"Main hypothesis: {context.hypothesis}")
    if context.requirements:
        lines.append(
            "Open questions and follow-up analyses this direction was selected to resolve. "
            "Every one of them must be addressed by a step, or explicitly recorded as out of scope:"
        )
        lines.extend(f"- {item}" for item in context.requirements[:MAX_BLOCK_REQUIREMENTS])
    if context.entities:
        lines.append(
            "Entities already implicated (carry these into the relevant steps rather than "
            "replacing them with generic category names):"
        )
        lines.append("- " + ", ".join(context.entities[:MAX_BLOCK_ENTITIES]))
    if context.notes:
        lines.append(
            "Known gaps and competing explanations, for context. These may be outside what "
            "this plan can settle; say so rather than silently ignoring them:"
        )
        lines.extend(f"- {item}" for item in context.notes[:MAX_BLOCK_NOTES])
    return "\n".join(lines)
