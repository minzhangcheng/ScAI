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
import json
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


# --- what earlier gates already decided --------------------------------------

# A gate can leave many candidates behind, and every carried candidate costs a
# novelty search and a slot in the scoring prompt. Carry the most promising few
# rather than the whole backlog.
MAX_CARRIED_CANDIDATES = 5


def _record_label(record: Any) -> str:
    """Human-readable name of the gate a candidate came from."""
    phase = str(_get(record, "phase", "") or "")
    cycle = _get(record, "integration_cycle")
    if phase == "integration" and cycle is not None:
        return f"Integration Cycle {cycle} - integration"
    return f"Round {_get(record, 'round_number', 0)} - {phase or 'exploration'}"


def _record_selections(record: Any) -> List[Any]:
    """Both shapes a gate writes its choices in: PI-decided and user-picked."""
    return list(_get(record, "pi_selected") or []) + list(_get(record, "selections") or [])


def investigated_direction_texts(state: Any) -> List[str]:
    """Every direction a gate has committed to, oldest first.

    ``user_chosen_directions`` deliberately holds only the most recent batch, so
    an integration gate deduplicating against it cannot see what earlier cycles
    already ran and re-proposes those directions as new. The selection records
    are the durable trail, and the legacy field remains the fallback for
    sessions that predate them.
    """
    texts: List[str] = []
    for record in getattr(state, "direction_selection_records", None) or []:
        for entry in _record_selections(record):
            text = str(_get(entry, "direction", "") or "").strip()
            if text:
                texts.append(text)
    for entry in getattr(state, "user_chosen_directions", None) or []:
        text = str(_get(entry, "direction", "") or "").strip()
        if text:
            texts.append(text)
    return _dedupe(texts)


# Containment only means something once a direction text is long enough to be
# one. Below this a stored placeholder like "A" is a substring of every
# candidate, and containment would swallow the whole list.
MIN_CONTAINMENT_MATCH_CHARS = 12


def _covers_same_direction(text: str, investigated: Sequence[str]) -> bool:
    """Whether an investigated direction already covers ``text``.

    Exact matching is not enough. A gate restates the same question in its own
    words -- "What initiates the interferon state?" and "Identify the source of
    the interferon response" are one direction with two names -- and an exact
    comparison re-offers work that has already run. This is the containment rule
    ``find_direction_candidate`` already uses for the same reason, with a length
    floor so a short placeholder cannot match everything.
    """
    key = normalize_direction_text(text)
    if not key:
        return False
    for other in investigated:
        if not other:
            continue
        if key == other:
            return True
        shorter, longer = (key, other) if len(key) <= len(other) else (other, key)
        if len(shorter) >= MIN_CONTAINMENT_MATCH_CHARS and shorter in longer:
            return True
    return False


def _declined_reason(record: Any, text: str) -> Optional[Dict[str, str]]:
    """What an earlier gate said when it passed this direction over."""
    key = normalize_direction_text(text)
    for entry in _get(record, "declined") or []:
        if normalize_direction_text(_get(entry, "direction", "")) == key:
            return {
                "gate": _record_label(record),
                "reason": str(_get(entry, "reason_not_selected", "") or "")[:400],
                "follow_up_value": str(_get(entry, "follow_up_value", "") or ""),
            }
    return None


def unpursued_prior_candidates(state: Any) -> List[Dict[str, Any]]:
    """Candidates an earlier gate proposed that no round ever ran.

    Each gate persists its frozen candidate list, but nothing reads it back: the
    integration gate builds candidates from the integration report alone, so a
    direction deferred at an earlier gate returns only if the model reinvents
    it. A prerequisite finishing does not bring it back either, because
    prerequisite handling runs once, at the gate that deferred it.

    Records are walked newest first, so a direction that was re-proposed and
    re-scored at a later gate is carried in that later, better-informed form
    rather than in its original one.

    A direction an earlier gate explicitly declined is still carried, annotated
    with what was said, rather than suppressed: the results that arrived since
    are exactly what could change the answer, and a gate that cannot see the
    direction cannot revisit it. The stored record is never modified.
    """
    investigated = [normalize_direction_text(t) for t in investigated_direction_texts(state)]
    seen: set = set()
    carried: List[Dict[str, Any]] = []
    records = list(getattr(state, "direction_selection_records", None) or [])
    for order, record in enumerate(reversed(records)):
        candidates = [c for c in (_get(record, "candidates") or []) if isinstance(c, dict)]
        # Prerequisite ids are only meaningful among the candidates of the gate
        # that issued them; the same id is reused by every later gate.
        text_by_id = {}
        for candidate in candidates:
            candidate_id = str(candidate.get("direction_id") or "").strip()
            if candidate_id and candidate_id not in text_by_id:
                text_by_id[candidate_id] = str(candidate.get("direction") or "").strip()
        label = _record_label(record)
        for candidate in candidates:
            text = str(candidate.get("direction") or "")
            key = normalize_direction_text(text)
            if not key or key in seen or _covers_same_direction(text, investigated):
                continue
            seen.add(key)
            enriched = dict(candidate)
            # Resolve prerequisites once, here, and keep the resolved text. A
            # carried candidate re-stored by a later gate keeps ids minted by the
            # gate that first issued it, so re-resolving them against a later
            # gate's id space names an unrelated direction.
            if "unmet_prerequisites" not in enriched:
                unmet: List[str] = []
                for prereq_id in as_list(candidate.get("prerequisite_dependencies")):
                    prereq_text = text_by_id.get(str(prereq_id), "")
                    if not prereq_text:
                        # A dangling id is an unknown prerequisite, not a
                        # direction named after the id token.
                        continue
                    if not _covers_same_direction(prereq_text, investigated):
                        unmet.append(prereq_text)
                enriched["unmet_prerequisites"] = unmet
            enriched["prerequisites_satisfied"] = not enriched.get("unmet_prerequisites")
            enriched.setdefault("carried_from_gate", label)
            declined = _declined_reason(record, text)
            if declined:
                enriched["previously_declined"] = declined
            enriched["_carry_recency"] = order
            carried.append(enriched)
    return carried


def merge_unpursued_candidates(
    state: Any,
    fresh_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Append earlier gates' unpursued candidates to a freshly generated batch.

    Freshly generated candidates keep their order and precedence: one that
    restates a carried direction wins, because it was written against the
    current evidence. Carried candidates follow, those whose prerequisites have
    since been satisfied first and, within that, the most recently deferred
    first -- the cap must not spend every slot on the oldest gate's backlog and
    cut the directions the last gate just held back.
    """
    fresh = [c for c in (fresh_candidates or []) if isinstance(c, dict)]
    covered = [normalize_direction_text(c.get("direction")) for c in fresh]
    carried = [
        candidate
        for candidate in unpursued_prior_candidates(state)
        if not _covers_same_direction(str(candidate.get("direction") or ""), covered)
    ]
    carried.sort(
        key=lambda c: (not c.get("prerequisites_satisfied", False), c.get("_carry_recency", 0))
    )
    for candidate in carried:
        candidate.pop("_carry_recency", None)
    return fresh + carried[:MAX_CARRIED_CANDIDATES]


# --- fitting a candidate list into a prompt ---------------------------------

# Fields a gate reasons over, most decisive first. A candidate that cannot have
# all of them loses its least decisive ones, never its place in the list. The
# first two are the identity and are always emitted: a candidate the model
# cannot name is one it can neither score nor reject.
CANDIDATE_PROMPT_FIELDS: Sequence[str] = (
    "direction_id",
    "direction",
    "core_observation",
    "main_hypothesis",
    "rationale",
    "functional_impact_hypothesis",
    "key_genes",
    "key_pathways",
    "key_cell_types",
    "supporting_data",
    "alternative_hypotheses",
    "artifact_risks",
    "prerequisite_dependencies",
    "unmet_prerequisites",
    "carried_from_gate",
    "previously_declined",
    "risk_flags",
    "novelty_score",
    "literature_novelty",
    "evidence_strength",
    "confidence",
    "priority_rank",
    "report_recommendation",
)

# The novelty record's verdict and its caveats. The per-article dumps and the
# full query strings are the bulk of the record and none of them change a
# scoring decision, so they are not sent.
EVIDENCE_PROMPT_FIELDS: Sequence[str] = (
    "direction_id",
    "novelty_status",
    "prior_art_level",
    "support_level",
    "contradiction_level",
    "direct_evidence_count",
    "analogous_evidence_count",
    "contradictory_evidence_count",
    "species_tissue_qualification",
    "gap_assessment",
    "summary",
    "limitations",
    "scheduler_limitation",
    "fallback_tier",
    "timed_out",
)


def _shorten(value: Any, budget: int) -> Any:
    """A JSON-serialisable copy of ``value`` that fits in about ``budget`` chars."""
    if budget <= 0:
        return None
    if isinstance(value, str):
        if len(value) <= budget:
            return value
        return value[: max(1, budget - 3)].rstrip() + "..."
    if isinstance(value, list):
        kept: List[Any] = []
        used = 2
        for item in value:
            piece = _shorten(item, budget - used)
            if piece is None:
                break
            used += len(json.dumps(piece, ensure_ascii=False)) + 1
            if used > budget:
                break
            kept.append(piece)
        return kept
    if isinstance(value, dict):
        kept_map: Dict[str, Any] = {}
        used = 2
        for key, item in value.items():
            piece = _shorten(item, budget - used - len(str(key)) - 6)
            if piece is None:
                break
            used += len(json.dumps({str(key): piece}, ensure_ascii=False))
            if used > budget:
                break
            kept_map[str(key)] = piece
        return kept_map
    return value


def budgeted_prompt_block(
    items: Optional[Sequence[Dict[str, Any]]],
    fields: Sequence[str],
    budget: int,
    *,
    floor: int = 320,
) -> str:
    """Render every item as JSON, shortening each to its share of ``budget``.

    ``json.dumps(items)[:budget]`` keeps the first entry or two whole and removes
    the rest without a trace. The model cannot keep, score, or reject an entry it
    never received, so a prompt built that way silently decides the outcome for
    everything past the cut. Here each item gets a slot and what does not fit is
    dropped from inside that item, tail-first, leaving a count behind.

    The block may exceed ``budget`` when there are more items than ``floor``
    allows for; every item appearing is the property worth keeping.
    """
    entries = [item for item in (items or []) if isinstance(item, dict)]
    if not entries:
        return "[]"
    share = max(floor, budget // len(entries))
    rendered: List[Dict[str, Any]] = []
    for item in entries:
        kept: Dict[str, Any] = {}
        omitted = 0
        used = 2
        for field in fields:
            value = item.get(field)
            if value is None or value == "" or value == [] or value == {}:
                continue
            piece = _shorten(value, max(0, share - used - len(field) - 6))
            if piece is None or piece == "" or piece == [] or piece == {}:
                omitted += 1
                continue
            kept[field] = piece
            used = len(json.dumps(kept, ensure_ascii=False))
        for field in fields[:2]:
            if field not in kept and item.get(field) not in (None, ""):
                kept[field] = item.get(field)
        if omitted:
            kept["fields_omitted_for_prompt_budget"] = omitted
        rendered.append(kept)
    return json.dumps(rendered, ensure_ascii=False, indent=1)


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
    #
    # Carrying a *predecessor* round's asks forward is deliberate — a later
    # direction is often generated from the previous round's next steps. A
    # *sibling* of the same batch is different: it was chosen alongside this
    # direction, never asked to resolve its questions, so adopting its asks
    # manufactures requirements this direction was never given and then reports
    # them as coverage gaps.
    key = normalize_direction_text(
        direction or _get(state, "current_round_direction", "") or ""
    )
    # With no direction to be a sibling *of* -- the exploration round clears
    # current_round_direction -- there is no sibling relationship to reason
    # about, and filtering on an empty key would discard the whole batch.
    siblings: set = set()
    if key:
        siblings = {
            normalize_direction_text(_get(entry, "direction", ""))
            for entry in (_get(state, "active_direction_batch", None) or [])
        } - {"", key}
    for result in reversed(list(getattr(state, "round_results", None) or [])):
        result_key = normalize_direction_text(
            _get(result, "direction_key", "") or _get(result, "direction", "")
        )
        if result_key and result_key in siblings:
            continue
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
