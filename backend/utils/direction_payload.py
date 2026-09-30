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
import hashlib

from ..models.schema_parts.enums import AgentType
from .formatting import as_list
from .bioinformatics_evidence import evidence_brief_references, render_evidence_briefs

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

# The new policy's execution facts, PI assessment, source-linked observations,
# and limitations lead the prompt. Legacy count/novelty fields remain for old
# records and are descriptive when policy_version is set.
EVIDENCE_PROMPT_FIELDS: Sequence[str] = (
    "direction_id",
    "policy_version",
    "plan_status",
    "assessment_status",
    "assessment_uncertainty",
    "assessment_limitations",
    "pi_reviews",
    "article_inventory",
    "extractions",
    "task_executions",
    "unexecuted_task_ids",
    "tasks",
    "source_passage",
    "cited_article_ids",
    # Legacy summary fields are retained below for historical evidence.
    "novelty_status",
    "prior_art_level",
    "support_level",
    "contradiction_level",
    "direct_evidence_count",
    "analogous_evidence_count",
    "contradictory_evidence_count",
    "species_tissue_qualification",
    # Ahead of the prose fields on purpose: budgeted_prompt_block drops
    # tail-first, and the article identities are what let a reader check the
    # counts rather than take them on faith.
    "key_evidence",
    "gap_assessment",
    "summary",
    "limitations",
    "scheduler_limitation",
    "fallback_tier",
    "timed_out",
)

_DISCLOSED_EVIDENCE_LIST_FIELDS: Sequence[str] = (
    "article_inventory",
    "extractions",
    "task_executions",
)

_EVIDENCE_LIST_IDENTITY_FIELDS = {
    "article_inventory": "pmid",
    "extractions": "pmid",
    "task_executions": "task_id",
}


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
        disclosed_counts: Dict[str, Any] = {}
        for field in fields:
            value = item.get(field)
            if value is None or value == "" or value == [] or value == {}:
                if item.get("policy_version") and field in _DISCLOSED_EVIDENCE_LIST_FIELDS:
                    disclosed_counts[f"{field}_source_count"] = 0
                    disclosed_counts[f"{field}_retained_count"] = 0
                    disclosed_counts[f"{field}_omitted_count"] = 0
                    disclosed_counts[f"{field}_detail_truncated_count"] = 0
                    disclosed_counts[f"{field}_detail_truncated"] = False
                continue
            piece = _shorten(value, max(0, share - used - len(field) - 6))
            if item.get("policy_version") and field in _DISCLOSED_EVIDENCE_LIST_FIELDS:
                source_count = len(value) if isinstance(value, list) else 0
                identity_field = _EVIDENCE_LIST_IDENTITY_FIELDS[field]
                retained_items: List[Dict[str, Any]] = []
                detail_truncated_count = 0
                if isinstance(value, list) and isinstance(piece, list):
                    for index, projected in enumerate(piece):
                        if not isinstance(projected, dict) or not projected.get(identity_field):
                            continue
                        retained_items.append(projected)
                        if index >= len(value) or projected != value[index]:
                            detail_truncated_count += 1
                    piece = retained_items
                retained_count = len(retained_items)
                disclosed_counts[f"{field}_source_count"] = source_count
                disclosed_counts[f"{field}_retained_count"] = retained_count
                disclosed_counts[f"{field}_omitted_count"] = max(
                    0, source_count - retained_count
                )
                disclosed_counts[f"{field}_detail_truncated_count"] = (
                    detail_truncated_count
                )
                disclosed_counts[f"{field}_detail_truncated"] = bool(
                    detail_truncated_count
                )
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
        kept.update(disclosed_counts)
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


# --- bounded direction evidence and follow-up handoff -----------------------

# These limits intentionally live with the projection.  Planning, reporting,
# and Chat should see the same bounded source view rather than each inventing a
# different truncation rule for the same scientific record.
MAX_DIRECTION_ROUNDS = 12
MAX_DIRECTION_FINDINGS = 8
MAX_DIRECTION_MECHANISMS = 8
MAX_DIRECTION_REVIEWS = 8
MAX_DIRECTION_CHAT_OUTCOMES = 8
MAX_DIRECTION_FOLLOWUPS = 12
MAX_DIRECTION_TEXT = 1200
MAX_DIRECTION_CONTEXT_TEXT = 16000

FOLLOWUP_DISPOSITIONS = frozenset({
    "deepen_now", "offer_in_chat", "needs_new_data", "closed",
})


def _model_dict(value: Any) -> Dict[str, Any]:
    """Return a detached JSON-shaped mapping for a model or mapping."""
    if isinstance(value, dict):
        return dict(value)
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        try:
            return dict(dumper(mode="json"))
        except Exception:
            try:
                return dict(dumper())
            except Exception:
                return {}
    return {}


def _lineage(value: Any) -> str:
    return str(_get(value, "lineage_id", None) or "main").strip() or "main"


def _round_number(value: Any) -> int:
    try:
        return max(0, int(_get(value, "round_number", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _direction_key_for(value: Any) -> str:
    return normalize_direction_text(
        _get(value, "direction_key", "") or _get(value, "direction", "")
    )


def _bounded_text(value: Any, limit: int = MAX_DIRECTION_TEXT) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _bounded_strings(values: Any, limit: int = MAX_DIRECTION_FINDINGS) -> List[str]:
    if isinstance(values, str):
        values = [values]
    elif not isinstance(values, (list, tuple)):
        return []
    out: List[str] = []
    for value in values or []:
        text = _bounded_text(value)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def direction_pass_identity(
    lineage_id: str,
    round_number: int,
    direction_key: str,
    *,
    phase: str = "innovation",
) -> str:
    """Build the stable server-owned identity for one direction pass.

    The workflow owns the same identity components as its result ledger.  This
    helper intentionally does not import workflow code, keeping the utility
    below the workflow dependency boundary.
    """
    lineage = str(lineage_id or "main").strip() or "main"
    phase_text = str(phase or "innovation").strip() or "innovation"
    key = normalize_direction_text(direction_key)
    return f"{lineage}:{phase_text}:{int(round_number or 0)}:{key}"


def _reference_values(value: Any, limit: int = MAX_DIRECTION_FINDINGS) -> List[str]:
    """Keep only compact source references, never client file paths/content."""
    values = value if isinstance(value, (list, tuple)) else ([] if value is None else [value])
    references: List[str] = []
    for raw in values:
        if isinstance(raw, dict):
            raw = (
                raw.get("execution_id")
                or raw.get("evidence_id")
                or raw.get("source")
                or raw.get("id")
            )
        text = _bounded_text(raw, 160)
        if text and text not in references:
            references.append(text)
        if len(references) >= limit:
            break
    return references


def stable_followup_id(
    pass_identity: str,
    ordinal: int,
    item: Dict[str, Any],
) -> str:
    """Derive a repeatable follow-up reference from a server-owned pass."""
    identity_payload = {
        "question": _bounded_text(item.get("question"), 800),
        "why_it_matters": _bounded_text(item.get("why_it_matters"), 800),
        "disposition": _bounded_text(item.get("disposition"), 80),
        "reason": _bounded_text(item.get("reason"), 800),
        "suggested_instruction": _bounded_text(item.get("suggested_instruction"), 800),
        "required_input": _bounded_text(item.get("required_input"), 800),
        "evidence_references": _reference_values(item.get("evidence_references")),
    }
    payload = json.dumps(
        {"pass": str(pass_identity or ""), "ordinal": int(ordinal), "item": identity_payload},
        ensure_ascii=False,
        sort_keys=True,
    )
    return "fu-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sanitize_followup_items(
    values: Any,
    *,
    pass_identity: str = "",
    source_round: int = 0,
    direction: str = "",
    lineage_id: str = "main",
    limit: int = MAX_DIRECTION_FOLLOWUPS,
) -> List[Dict[str, Any]]:
    """Normalize PI-authored remaining questions into bounded handoff records.

    This is a structural sanitizer. It preserves the PI's disposition and
    uncertainty, but never decides whether a question is scientifically useful.
    The workflow calls this once while persisting the PI decision.  It supplies
    the server-owned pass identity, and this function assigns the resulting
    ``followup_id`` once.  Persisted records are read by
    ``_stored_followup_items`` below, which keeps that ID unchanged even when an
    assessment adds an outcome or the item order changes.
    """
    if not isinstance(values, (list, tuple)):
        return []
    lineage = str(lineage_id or "main").strip() or "main"
    default_identity = pass_identity or direction_pass_identity(
        lineage, source_round, direction,
    )
    output: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, raw in enumerate(values):
        if not isinstance(raw, dict):
            continue
        question = _bounded_text(raw.get("question"), 1000)
        if not question:
            continue
        disposition = str(raw.get("disposition") or "").strip().lower()
        if disposition not in FOLLOWUP_DISPOSITIONS:
            continue
        evidence = _reference_values(raw.get("evidence_references"))
        # Source ownership is supplied by the workflow.  A PI response may
        # describe the question, but it cannot move the follow-up to another
        # round or direction by writing those fields in its payload.
        try:
            item_round = max(0, int(source_round or 0))
        except (TypeError, ValueError):
            item_round = 0
        normalized = {
            "question": question,
            "why_it_matters": _bounded_text(raw.get("why_it_matters"), 1200),
            "disposition": disposition,
            "reason": _bounded_text(raw.get("reason"), 1200),
            "evidence_references": evidence,
            "suggested_instruction": _bounded_text(raw.get("suggested_instruction"), 1400),
            "required_input": _bounded_text(raw.get("required_input"), 1200),
            "source_round": item_round,
            "direction": _bounded_text(direction, 800),
            "lineage_id": lineage,
        }
        # The ID is server-owned. A raw PI response cannot choose the source
        # record it will later resolve in Chat.
        normalized["followup_id"] = stable_followup_id(default_identity, ordinal, normalized)
        if normalized["followup_id"] in seen:
            continue
        seen.add(normalized["followup_id"])
        output.append(normalized)
        if len(output) >= max(0, int(limit)):
            break
    return output


def _review_followup_values(review: Any) -> List[Any]:
    """Read the one canonical persisted review location.

    New records keep follow-ups at the review-record root.  Old records have no
    follow-up decision and therefore intentionally produce an empty list rather
    than trying to infer one from an assessment or free-text field.
    """
    if not isinstance(review, dict) or not isinstance(review.get("followups"), list):
        return []
    return list(review["followups"])


def _stored_followup_items(values: Any, *, limit: int = MAX_DIRECTION_FOLLOWUPS) -> List[Dict[str, Any]]:
    """Bound canonical persisted follow-ups without changing their IDs.

    This is deliberately separate from ``sanitize_followup_items``.  A stored
    decision is an addressable record: recomputing its hash from mutable text or
    ordinal would invalidate an already rendered report after assessment merge.
    """
    if not isinstance(values, (list, tuple)):
        return []
    output: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, dict):
            continue
        followup_id = str(raw.get("followup_id") or "").strip()
        question = _bounded_text(raw.get("question"), 1000)
        disposition = str(raw.get("disposition") or "").strip().lower()
        if not followup_id or not question or disposition not in FOLLOWUP_DISPOSITIONS:
            continue
        if followup_id in seen:
            continue
        item = {
            "followup_id": followup_id[:160],
            "question": question,
            "why_it_matters": _bounded_text(raw.get("why_it_matters"), 1200),
            "disposition": disposition,
            "reason": _bounded_text(raw.get("reason"), 1200),
            "evidence_references": _reference_values(raw.get("evidence_references")),
            "suggested_instruction": _bounded_text(raw.get("suggested_instruction"), 1400),
            "required_input": _bounded_text(raw.get("required_input"), 1200),
            "source_round": _round_number(raw),
            "direction": _bounded_text(raw.get("direction"), 800),
            "lineage_id": _lineage(raw),
        }
        for field_name in ("execution_outcome", "execution_reference"):
            value = raw.get(field_name)
            if value not in (None, "", [], {}):
                item[field_name] = _bounded_text(value, 1200)
        output.append(item)
        seen.add(followup_id)
        if len(output) >= max(0, int(limit)):
            break
    return output


def _result_projection(result: Any) -> Dict[str, Any]:
    """Project one RoundResult without copying arbitrary notebook payloads."""
    round_number = _round_number(result)
    programs = []
    for program in (_get(result, "data_driven_mechanism_programs") or [])[:MAX_DIRECTION_MECHANISMS]:
        data = _model_dict(program)
        if not data:
            continue
        programs.append({
            "id": _bounded_text(data.get("id"), 120),
            "label": _bounded_text(data.get("label"), 300),
            "description": _bounded_text(data.get("description"), 1000),
            "round_number": _round_number(data) or round_number,
            "cell_types": _bounded_strings(data.get("cell_types"), 8),
            "conditions_compared": _bounded_text(data.get("conditions_compared"), 500),
            "triggering_observations": _bounded_strings(
                data.get("triggering_observations"), MAX_DIRECTION_FINDINGS,
            ),
            "evidence_strength": _bounded_text(data.get("evidence_strength"), 80),
            "specificity": _bounded_text(data.get("specificity"), 80),
            "bulk_support": _bounded_text(data.get("bulk_support"), 80),
            "supporting_genes": _bounded_strings(data.get("supporting_genes"), 12),
            "supporting_pathways": _bounded_strings(data.get("supporting_pathways"), 12),
            "supporting_scores": _bounded_strings(data.get("supporting_scores"), 8),
            "supporting_tables": _bounded_strings(data.get("supporting_tables"), 8),
            "evidence_execution_ids": _reference_values(data.get("evidence_execution_ids")),
            "alternative_explanations": _bounded_strings(data.get("alternative_explanations"), 6),
            "missing_evidence": _bounded_strings(data.get("missing_evidence"), 6),
            "interpretation": _bounded_text(data.get("interpretation"), 1200),
        })
    conflicts = []
    for conflict in (_get(result, "conflicts") or [])[:MAX_DIRECTION_FINDINGS]:
        if isinstance(conflict, dict):
            conflicts.append({
                "description": _bounded_text(conflict.get("description"), 800),
                "resolution": _bounded_text(conflict.get("resolution"), 800),
                "agents": _bounded_strings(conflict.get("agents"), 6),
            })
        else:
            text = _bounded_text(conflict)
            if text:
                conflicts.append({"description": text, "resolution": "", "agents": []})
    return {
        "round_number": round_number,
        "phase": _bounded_text(_get(result, "phase", "innovation"), 40),
        "direction": _bounded_text(_get(result, "direction", ""), 800),
        "direction_key": _direction_key_for(result),
        "result_identity_key": _bounded_text(_get(result, "result_identity_key", ""), 240),
        "round_status": _bounded_text(_get(result, "round_status", ""), 40),
        "evidence_status": _bounded_text(_get(result, "evidence_status", ""), 40),
        "summary": _bounded_text(_get(result, "pi_summary", "") or _get(result, "summary", ""), 1600),
        "findings": {
            "pathology": _bounded_strings(_get(result, "pathology_findings") or [], MAX_DIRECTION_FINDINGS),
            "bioinformatics": _bounded_strings(_get(result, "bioinformatics_findings") or [], MAX_DIRECTION_FINDINGS),
            "biochemistry": _bounded_strings(_get(result, "biochemistry_findings") or [], MAX_DIRECTION_FINDINGS),
            "literature": _bounded_strings(_get(result, "literature_findings") or [], MAX_DIRECTION_FINDINGS),
        },
        "conflicts": conflicts,
        "next_questions": _bounded_strings(_get(result, "next_steps") or [], MAX_DIRECTION_FINDINGS),
        "evidence_execution_ids": _reference_values(_get(result, "evidence_execution_ids") or []),
        "evidence_notes": _bounded_strings(_get(result, "evidence_notes") or [], MAX_DIRECTION_FINDINGS),
        "evidence_gate_findings": _bounded_strings(
            _get(result, "evidence_gate_findings") or [], MAX_DIRECTION_FINDINGS,
        ),
        "mechanism_programs": programs,
        "evidence_briefs": render_evidence_briefs(_get(result, "bioinformatics_finding_records")),
        "evidence_brief_references": evidence_brief_references(_get(result, "bioinformatics_finding_records")),
    }


def _round_is_persisted_finished(result: Any) -> bool:
    """Whether a persisted innovation result can contribute direction context.

    This is an evidence-presence check, not a scientific completion verdict.
    A result may be partial or marked failed after writing useful observations;
    those observations must remain available to the PI. Empty legacy records are
    harmless and are represented only when they carry an explicit direction.
    """
    phase = str(_get(result, "phase", "") or "").strip().lower()
    if phase != "innovation":
        return False
    return bool(_direction_key_for(result))


def _selected_direction_keys(state: Any, lineage_id: str) -> List[str]:
    keys: List[str] = []
    for result in getattr(state, "round_results", None) or []:
        if _lineage(result) != lineage_id or not _round_is_persisted_finished(result):
            continue
        key = _direction_key_for(result)
        if key and key not in keys:
            keys.append(key)
    # Include selected directions that have not reached a RoundResult only as a
    # context placeholder; callers can then explain that no result was retained.
    for entry in (
        list(getattr(state, "active_direction_batch", None) or [])
        + list(getattr(state, "user_chosen_directions", None) or [])
    ):
        key = normalize_direction_text(_get(entry, "direction", ""))
        if key and key not in keys:
            keys.append(key)
    return keys


def build_direction_projection(
    state: Any,
    direction: str = "",
    *,
    lineage_id: Optional[str] = None,
    max_rounds: int = MAX_DIRECTION_ROUNDS,
) -> Dict[str, Any]:
    """Collect bounded evidence and follow-up records for one direction."""
    target_lineage = str(
        lineage_id or getattr(state, "restore_branch_id", None) or "main"
    ).strip() or "main"
    requested_key = normalize_direction_text(direction)
    keys = [requested_key] if requested_key else _selected_direction_keys(state, target_lineage)
    key = requested_key or (keys[0] if len(keys) == 1 else "")
    matching_results = []
    for raw in getattr(state, "round_results", None) or []:
        if _lineage(raw) != target_lineage or not _round_is_persisted_finished(raw):
            continue
        result_key = _direction_key_for(raw)
        if key and result_key != key:
            continue
        if not key and result_key not in keys:
            continue
        matching_results.append(_result_projection(raw))
    matching_results.sort(key=lambda item: item.get("round_number", 0))
    matching_results = matching_results[-max(1, int(max_rounds)):]
    if not direction and not key and matching_results:
        key = matching_results[0].get("direction_key", "")
    resolved_direction = direction or next(
        (str(item.get("direction") or "") for item in matching_results if item.get("direction")),
        "",
    )
    if not resolved_direction and key:
        resolved_direction = key

    candidate = candidate_for_state(state, resolved_direction) if resolved_direction else None
    candidate_projection = _model_dict(candidate) if candidate else {}
    if candidate_projection:
        # Keep candidate identity and all direction-specific evidence, but bound
        # large arrays before the projection crosses into an LLM prompt.
        candidate_projection = {
            str(name): (
                _bounded_strings(value, 12)
                if isinstance(value, list)
                else _bounded_text(value, 1200)
                if isinstance(value, str)
                else value
            )
            for name, value in candidate_projection.items()
            if name not in {"articles", "queries", "raw_results"}
        }

    pass_identities = {
        direction_pass_identity(target_lineage, item.get("round_number", 0), item.get("direction_key", ""))
        for item in matching_results
        if item.get("direction_key")
    }
    reviews: List[Dict[str, Any]] = []
    followups: List[Dict[str, Any]] = []
    for raw_review in reversed(list(getattr(state, "post_execution_reviews", None) or [])):
        if not isinstance(raw_review, dict) or _lineage(raw_review) != target_lineage:
            continue
        review_key = normalize_direction_text(
            raw_review.get("direction_key") or raw_review.get("direction", "")
        )
        review_round = _round_number(raw_review)
        review_identity = str(
            raw_review.get("direction_pass_id")
            or raw_review.get("direction_pass_identity")
            or raw_review.get("result_identity_key")
            or ""
        )
        matches = bool(
            (key and review_key == key)
            or (review_identity and review_identity in pass_identities)
            or (not key and review_round in {r.get("round_number") for r in matching_results})
        )
        if not matches:
            continue
        review_copy = {
            "review_status": _bounded_text(raw_review.get("review_status"), 80),
            "review_reason": _bounded_text(raw_review.get("review_reason"), 1200),
            "round_number": review_round,
            "direction": _bounded_text(raw_review.get("direction") or resolved_direction, 800),
            "direction_key": review_key or key,
            "review_attempt_id": _bounded_text(raw_review.get("review_attempt_id"), 120),
            "assessment": _model_dict(raw_review.get("assessment")),
        }
        reviews.append(review_copy)
        pass_identity = review_identity or direction_pass_identity(
            target_lineage, review_round, review_key or key,
        )
        items = _review_followup_values(raw_review)
        # Review records are already sanitized at decision persistence. Keep
        # their IDs and mutable outcome fields unchanged across projection.
        followups.extend(_stored_followup_items(items))
        if len(reviews) >= MAX_DIRECTION_REVIEWS:
            break

    unique_followups: List[Dict[str, Any]] = []
    seen_followups: set[str] = set()
    for item in followups:
        if item["followup_id"] in seen_followups:
            continue
        seen_followups.add(item["followup_id"])
        unique_followups.append(item)
        if len(unique_followups) >= MAX_DIRECTION_FOLLOWUPS:
            break

    chat_outcomes: List[Dict[str, Any]] = []
    followup_ids = {item["followup_id"] for item in unique_followups}
    for message in reversed(list(getattr(state, "chat_messages", None) or [])):
        if not isinstance(message, dict):
            continue
        if str(message.get("branch_id") or "main") != target_lineage:
            continue
        message_followup = str(message.get("followup_id") or "")
        if not message_followup or message_followup not in followup_ids:
            continue
        chat_outcomes.append({
            "id": _bounded_text(message.get("id"), 120),
            "role": _bounded_text(message.get("role"), 40),
            "followup_id": message_followup,
            "branch_id": target_lineage,
            "reply_to": _bounded_text(message.get("reply_to"), 120),
            "content": _bounded_text(message.get("content"), 1600),
            "timestamp": _bounded_text(message.get("timestamp"), 80),
        })
        if len(chat_outcomes) >= MAX_DIRECTION_CHAT_OUTCOMES:
            break

    available_data = ""
    try:
        available_data = _bounded_text(state.build_experimental_design_summary(), 2400)
    except Exception:
        available_data = ""
    omissions = []
    total_results = sum(
        1 for raw in getattr(state, "round_results", None) or []
        if _lineage(raw) == target_lineage and _round_is_persisted_finished(raw)
        and (not key or _direction_key_for(raw) == key)
    )
    if total_results > len(matching_results):
        omissions.append(f"{total_results - len(matching_results)} earlier direction result(s) omitted by the context cap.")
    return {
        "lineage_id": target_lineage,
        "direction": resolved_direction,
        "direction_key": key,
        "pass_identities": sorted(pass_identities),
        "candidate": candidate_projection,
        "rounds": matching_results,
        "reviews": reviews,
        "followups": unique_followups,
        "chat_outcomes": chat_outcomes,
        "available_data": available_data,
        "omissions": omissions,
    }


def collect_direction_projections(
    state: Any,
    *,
    lineage_id: Optional[str] = None,
    max_directions: int = MAX_DIRECTION_ROUNDS,
) -> List[Dict[str, Any]]:
    """Return one projection for each retained innovation direction."""
    target = str(lineage_id or getattr(state, "restore_branch_id", None) or "main").strip() or "main"
    keys = _selected_direction_keys(state, target)
    try:
        direction_cap = max(1, int(max_directions))
    except (TypeError, ValueError):
        direction_cap = MAX_DIRECTION_ROUNDS
    projections = [
        build_direction_projection(state, key, lineage_id=target)
        for key in keys[:direction_cap]
    ]
    projections = [
        item for item in projections
        if item.get("direction_key") or item.get("direction")
    ]
    omitted = max(0, len(keys) - direction_cap)
    if omitted and projections:
        omissions = projections[0].setdefault("omissions", [])
        omissions.append(
            f"{omitted} selected innovation direction(s) omitted by the "
            f"{direction_cap}-direction context cap; persisted records remain available."
        )
    return projections


def build_direction_projection_block(
    projection: Any,
    *,
    max_chars: int = MAX_DIRECTION_CONTEXT_TEXT,
) -> str:
    """Format a projection for PI planning, review, synthesis, or Chat.

    The formatter keeps source references and a visible omission note when its
    caller supplies a smaller context budget.  A raw prefix slice is unsafe:
    it can remove every limiting observation and execution join key while still
    looking like a complete context block.
    """
    if not isinstance(projection, dict):
        return ""

    def _values_text(values: Any, limit: int = MAX_DIRECTION_FINDINGS) -> str:
        if isinstance(values, str):
            values = [values]
        elif not isinstance(values, (list, tuple)):
            return ""
        return "; ".join(_bounded_text(value, 600) for value in (values or [])[:limit])

    lines = ["## Selected Direction Evidence Context"]
    direction = _bounded_text(projection.get("direction"), 800)
    if direction:
        lines.append(f"Direction: {direction}")

    candidate = projection.get("candidate") or {}
    if candidate:
        lines.append("Frozen candidate details (observations and PI notes):")
        for field_name in (
            "core_observation", "main_hypothesis", "rationale", "key_uncertainties",
            "alternative_hypotheses", "supporting_data", "artifact_risks",
        ):
            value = candidate.get(field_name)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, list):
                value = _values_text(value, 8)
            elif isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False)[:1000]
            lines.append(f"- {field_name}: {_bounded_text(value, 1200)}")

    source_references: List[str] = []

    def _add_refs(values: Any) -> None:
        for value in values or []:
            ref = _bounded_text(value, 160)
            if ref and ref not in source_references:
                source_references.append(ref)

    for round_data in projection.get("rounds") or []:
        round_number = round_data.get("round_number", "?")
        lines.append(f"\n### Source Round {round_number}")
        if round_data.get("result_identity_key"):
            lines.append(f"Result identity: {round_data['result_identity_key']}")
            _add_refs([round_data["result_identity_key"]])
        if round_data.get("round_status") or round_data.get("evidence_status"):
            lines.append(
                "Persisted status: "
                f"{round_data.get('round_status') or 'unknown'} / "
                f"evidence {round_data.get('evidence_status') or 'unknown'}"
            )
        if round_data.get("evidence_execution_ids"):
            lines.append("Execution evidence IDs: " + _values_text(round_data["evidence_execution_ids"], 12))
            _add_refs(round_data["evidence_execution_ids"])
        if round_data.get("summary"):
            lines.append(f"PI summary: {round_data['summary']}")
        if round_data.get("evidence_briefs"):
            _add_refs(round_data.get("evidence_brief_references") or [])
            lines.extend(round_data["evidence_briefs"].splitlines())
        findings = round_data.get("findings") or {}
        for modality, values in findings.items():
            if values:
                lines.append(f"{modality.title()} observations: " + _values_text(values))
        for conflict in round_data.get("conflicts") or []:
            description = conflict.get("description")
            resolution = conflict.get("resolution")
            if description:
                lines.append(
                    f"Contrary/conflicting evidence: {description}"
                    + (f" Resolution: {resolution}" if resolution else "")
                )
        if round_data.get("evidence_notes"):
            lines.append("Evidence limitations: " + _values_text(round_data["evidence_notes"]))
        if round_data.get("evidence_gate_findings"):
            lines.append("Measured evidence observations: " + _values_text(round_data["evidence_gate_findings"]))
        if round_data.get("next_questions"):
            lines.append("Open questions recorded by the PI: " + _values_text(round_data["next_questions"]))
        for program in round_data.get("mechanism_programs") or []:
            program_id = program.get("id") or ""
            label = program.get("label") or "Mechanism record"
            if program_id:
                _add_refs([program_id])
            if program.get("evidence_execution_ids"):
                _add_refs(program["evidence_execution_ids"])
            lines.append(f"Mechanism evidence: {label}")
            if program.get("triggering_observations"):
                lines.append("Triggering observations: " + _values_text(program["triggering_observations"]))
            evidence = "; ".join(
                (program.get("supporting_genes") or [])[:8]
                + (program.get("supporting_pathways") or [])[:8]
            )
            if evidence:
                lines.append("Supporting genes/pathways: " + evidence)
            if program.get("evidence_execution_ids"):
                lines.append("Mechanism execution IDs: " + _values_text(program["evidence_execution_ids"], 12))
            if program.get("interpretation"):
                lines.append("PI interpretation: " + _bounded_text(program["interpretation"], 1200))
            if program.get("alternative_explanations"):
                lines.append("Alternative explanations: " + _values_text(program["alternative_explanations"], 4))
            if program.get("missing_evidence"):
                lines.append("Missing/limiting evidence: " + _values_text(program["missing_evidence"], 4))

    for review in (projection.get("reviews") or [])[:MAX_DIRECTION_REVIEWS]:
        if review.get("review_attempt_id"):
            _add_refs([review["review_attempt_id"]])
        if review.get("assessment"):
            assessment = review["assessment"]
            summary = assessment.get("summary") if isinstance(assessment, dict) else ""
            if summary:
                lines.append(
                    f"Follow-up assessment (round {review.get('round_number', '?')}): "
                    f"{_bounded_text(summary, 1200)}"
                )
            if isinstance(assessment, dict) and assessment.get("remaining_gaps"):
                lines.append("Assessment limitations: " + _values_text(assessment["remaining_gaps"]))

    if projection.get("followups"):
        lines.append("Remaining questions and PI dispositions:")
        for item in projection["followups"][:MAX_DIRECTION_FOLLOWUPS]:
            line = f"- [{item.get('disposition')}] {item.get('question')}"
            if item.get("why_it_matters"):
                line += f" Why it matters: {item['why_it_matters']}"
            if item.get("reason"):
                line += f" Reason: {item['reason']}"
            if item.get("evidence_references"):
                line += " Evidence refs: " + _values_text(item["evidence_references"], 8)
                _add_refs(item["evidence_references"])
            if item.get("execution_reference"):
                line += " Outcome ref: " + _bounded_text(item["execution_reference"], 160)
                _add_refs([item["execution_reference"]])
            lines.append(line)

    if projection.get("chat_outcomes"):
        lines.append("Chat outcomes linked to these follow-ups:")
        for outcome in projection["chat_outcomes"][:MAX_DIRECTION_CHAT_OUTCOMES]:
            role = _bounded_text(outcome.get("role"), 40) or "message"
            content = _bounded_text(outcome.get("content"), 1000)
            identity = _bounded_text(outcome.get("id"), 120)
            if identity:
                _add_refs([identity])
            lines.append(f"- {role}: {content}" + (f" (message {identity})" if identity else ""))
    if projection.get("available_data"):
        lines.append("Available-data context:\n" + _bounded_text(projection["available_data"], 2400))

    existing_omissions = [str(value) for value in (projection.get("omissions") or []) if str(value).strip()]
    references_line = "Source references: " + (", ".join(source_references[:MAX_DIRECTION_FINDINGS * 4]) or "none recorded")
    max_len = max(1, int(max_chars or MAX_DIRECTION_CONTEXT_TEXT))
    full = "\n".join(lines + (["Context omissions: " + "; ".join(existing_omissions)] if existing_omissions else []) + [references_line])
    if len(full) <= max_len:
        return full

    # Keep a small, explicit trailer in every capped block.  The source IDs in
    # this trailer let a reviewer join the omitted prose back to persisted
    # records, while the omission message prevents a shortened context from
    # being mistaken for complete evidence.
    omitted_note = "Context omissions: direction context was capped; omitted material remains in persisted records."
    if existing_omissions:
        omitted_note += " " + "; ".join(existing_omissions)
    trailer = "\n" + omitted_note + "\n" + references_line
    if len(trailer) >= max_len:
        # Preserve a machine-readable indication even for an unusually tiny
        # caller budget.  Normal report/Chat budgets are large enough for the
        # full trailer and source IDs.
        compact = "Context omissions: capped."
        if max_len <= len(compact):
            return compact[:max_len]
        return compact + ("\n" + references_line)[: max_len - len(compact)]
    body_budget = max_len - len(trailer)
    body_lines: List[str] = []
    used = 0
    omitted_count = 0
    for line in lines:
        extra = len(line) + (1 if body_lines else 0)
        if used + extra > body_budget:
            omitted_count += 1
            continue
        body_lines.append(line)
        used += extra
    if omitted_count:
        while True:
            omitted_note = (
                "Context omissions: direction context was capped; "
                f"{omitted_count} rendered item(s) omitted. Persisted records and source references remain available."
            )
            if existing_omissions:
                omitted_note += " " + "; ".join(existing_omissions)
            trailer = "\n" + omitted_note + "\n" + references_line
            if len(trailer) > max_len:
                trailer = trailer[:max_len]
                body_lines = []
                break
            if len("\n".join(body_lines)) + len(trailer) <= max_len:
                break
            body_lines.pop()
            omitted_count += 1
    return "\n".join(body_lines) + trailer


def resolve_followup_reference(
    state: Any,
    followup_id: str,
    *,
    lineage_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a stable follow-up only inside the owned active lineage."""
    requested = str(followup_id or "").strip()
    if not requested or len(requested) > 160:
        return None
    target_lineage = str(lineage_id or getattr(state, "restore_branch_id", None) or "main").strip() or "main"
    for raw_review in reversed(list(getattr(state, "post_execution_reviews", None) or [])):
        if not isinstance(raw_review, dict) or _lineage(raw_review) != target_lineage:
            continue
        round_number = _round_number(raw_review)
        direction = str(raw_review.get("direction") or "").strip()
        key = normalize_direction_text(raw_review.get("direction_key") or direction)
        items = _stored_followup_items(_review_followup_values(raw_review))
        for item in items:
            if item.get("followup_id") != requested:
                continue
            projection = build_direction_projection(
                state, item.get("direction") or direction, lineage_id=target_lineage,
            )
            return {
                **item,
                "direction_key": projection.get("direction_key") or key,
                "source_context": projection,
            }
    return None
