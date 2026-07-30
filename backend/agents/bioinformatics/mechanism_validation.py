# Validation and normalization of mechanism-program payloads.
# Moved verbatim out of backend/agents/bioinformatics_agent.py.
"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
With step-level checkpointing for recovery from interruptions
"""
import logging
import re


# Configure logger
logger = logging.getLogger(__name__)


_OVERCLAIM_WORDS = ("confirmed", "proven", "causal", "demonstrates", "establishes")

_NEGATION_WORDS = {"not", "no", "never", "unlikely", "cannot", "can't", "doesn't",
                    "don't", "didn't", "won't", "wouldn't", "shouldn't", "isn't",
                    "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"}

_MAX_PROGRAMS = 8

_MAX_GENES = 50

_MAX_OBS_LEN = 500

_DEFAULT_MISSING = "Direct causal validation is not available from the current data."

_REQUIRED_FIELDS = ("label", "triggering_observations", "evidence_strength",
                    "specificity", "interpretation")

def _ensure_list(val):
    """Normalize a value to a list — pass through lists, wrap strings, skip None/other."""
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        return val
    return []

def _overclaim_unnegated(text):
    """Check if text contains overclaim words not negated within the prior 3 words."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    for i, w in enumerate(words):
        if w in _OVERCLAIM_WORDS:
            start = max(0, i - 3)
            if not any(n in words[start:i] for n in _NEGATION_WORDS):
                return True
    return False

#: Grades a claim may carry, weakest first. Used to clamp rather than to refuse:
#: the strength of a claim is bounded by the reproducibility of the instruments
#: it rests on, and lowering it costs nothing, whereas refusing costs a rejection,
#: a retry, and a step from a bounded analysis budget.
_STRENGTH_ORDER = ("weak", "moderate", "strong")

#: Provenance kinds that another investigator cannot retrieve. A panel built for
#: this analysis may be the best instrument available -- for a concept no public
#: resource covers, it is the ONLY one -- so this is not a defect. It does mean
#: a result resting entirely on such panels cannot also be the strongest grade
#: on offer, because nobody outside this run can re-derive the instrument.
_UNREPRODUCIBLE_PROVENANCE_KINDS = ("bespoke",)

_MAX_STRENGTH_ON_UNREPRODUCIBLE_SOURCES = "moderate"


def _clamp_strength(declared, ceiling):
    """The weaker of a declared grade and a ceiling."""
    try:
        return declared if _STRENGTH_ORDER.index(declared) <= _STRENGTH_ORDER.index(ceiling) \
            else ceiling
    except ValueError:
        return ceiling


def _provenance_ceiling(supporting_scores, score_provenance):
    """Ceiling implied by the sources a program cites, and why.

    Returns (ceiling, note) with ceiling None when nothing constrains the grade.
    Only resolvable citations count: a score name that matches no registration
    says nothing about reproducibility either way, and guessing from it would
    penalise a program for a bookkeeping gap rather than for its evidence.
    """
    if not score_provenance:
        return None, ""
    resolved = []
    for name in supporting_scores or ():
        entry = score_provenance.get(str(name).strip())
        if isinstance(entry, dict):
            resolved.append((str(name).strip(), str(entry.get("kind", "") or "").lower()))
    if not resolved:
        return None, ""
    unreproducible = [n for n, kind in resolved if kind in _UNREPRODUCIBLE_PROVENANCE_KINDS]
    if len(unreproducible) != len(resolved):
        return None, ""
    return _MAX_STRENGTH_ON_UNREPRODUCIBLE_SOURCES, (
        "Every supporting instrument was constructed for this analysis and cannot be "
        "retrieved independently (" + ", ".join(sorted(unreproducible)[:6]) + "), so the "
        "evidence grade is capped at "
        f"'{_MAX_STRENGTH_ON_UNREPRODUCIBLE_SOURCES}'. Re-testing against a versioned "
        "public resource, or against an independently constructed panel, is what would "
        "raise it."
    )


def validate_mechanism_program_payload(payload, *, round_number=0, phase="", stage="",
                                       score_provenance=None):
    """Validate a register_mechanism_programs payload conservatively.

    Returns (programs, error_message). On any structural error, programs is []
    and error_message is a corrective string to return to the agent. An empty
    programs list is valid ONLY when no_program_reason is non-empty.

    `score_provenance` maps a registered score name to its provenance dict. When
    supplied, a program whose every resolvable supporting score was constructed
    for this analysis has its `evidence_strength` clamped and the reason appended
    to `missing_evidence`. This stays a clamp on purpose: the platform records
    provenance meticulously and then acts on none of it, which is how a claim
    grounded in instruments nobody else can obtain came to read the same as one
    grounded in versioned public resources.
    """
    from ...models.schemas import DataDrivenMechanismProgram

    if not isinstance(payload, dict):
        return [], "Invalid payload: expected an object with a 'programs' list."

    raw_programs = payload.get("programs") or []
    if not isinstance(raw_programs, list):
        return [], "'programs' must be a list."
    no_program_reason = (payload.get("no_program_reason") or "").strip()

    if not raw_programs:
        if not no_program_reason:
            return [], ("Empty programs list requires a non-empty no_program_reason "
                        "explaining why no data-driven mechanism program was registered.")
        return [], None

    raw_programs = raw_programs[:_MAX_PROGRAMS]
    programs = []
    for idx, rp in enumerate(raw_programs):
        if not isinstance(rp, dict):
            return [], f"Program {idx} is not an object."
        for field in _REQUIRED_FIELDS:
            val = rp.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                return [], f"Program {idx} is missing required field '{field}'."
        raw_obs = rp.get("triggering_observations") or []
        if isinstance(raw_obs, str):  # tolerate a single string instead of a list
            raw_obs = [raw_obs]
        if not isinstance(raw_obs, list):
            return [], f"Program {idx} triggering_observations must be a list."
        observations = [o for o in raw_obs if str(o).strip()]
        if not observations:
            return [], f"Program {idx} has empty triggering_observations."
        observations = [str(o)[:_MAX_OBS_LEN] for o in observations]
        evidence_execution_ids = [
            str(value).strip()
            for value in _ensure_list(rp.get("evidence_execution_ids"))
            if str(value).strip()
        ]
        if not evidence_execution_ids:
            return [], f"Program {idx} requires at least one evidence_execution_id."

        alternatives = [a for a in _ensure_list(rp.get("alternative_explanations")) if str(a).strip()]
        missing = [m for m in _ensure_list(rp.get("missing_evidence")) if str(m).strip()]
        if not missing:
            missing = [_DEFAULT_MISSING]

        interpretation = str(rp.get("interpretation", ""))
        if _overclaim_unnegated(interpretation):
            missing = missing + [
                "Interpretation may overclaim (uses confirmatory/causal wording); "
                "treat as candidate, not confirmed."
            ]

        bulk_support = rp.get("bulk_support", "unknown")
        if bulk_support not in ("supported", "not_supported", "mixed", "not_available", "unknown"):
            bulk_support = "unknown"
        evidence_strength = rp.get("evidence_strength")
        if evidence_strength not in _STRENGTH_ORDER:
            evidence_strength = "weak"
        supporting_scores = [str(s) for s in _ensure_list(rp.get("supporting_scores"))]
        ceiling, ceiling_note = _provenance_ceiling(supporting_scores, score_provenance)
        if ceiling:
            clamped = _clamp_strength(evidence_strength, ceiling)
            if clamped != evidence_strength:
                missing = missing + [ceiling_note]
                logger.info(
                    "mechanism program %r: evidence_strength %s -> %s "
                    "(all supporting instruments are analysis-specific)",
                    str(rp.get("label", ""))[:80], evidence_strength, clamped,
                )
            evidence_strength = clamped
        specificity = rp.get("specificity")
        if specificity not in ("high", "medium", "low"):
            specificity = "low"

        programs.append(DataDrivenMechanismProgram(
            round_number=round_number,
            phase=phase,
            stage=stage,
            label=str(rp.get("label", ""))[:200],
            description=str(rp.get("description", "")),
            cell_types=[str(c) for c in _ensure_list(rp.get("cell_types"))],
            conditions_compared=str(rp.get("conditions_compared", "")),
            triggering_observations=observations,
            supporting_genes=[str(g) for g in _ensure_list(rp.get("supporting_genes"))][:_MAX_GENES],
            supporting_pathways=[str(p) for p in _ensure_list(rp.get("supporting_pathways"))],
            supporting_scores=supporting_scores,
            supporting_tables=[str(t) for t in _ensure_list(rp.get("supporting_tables"))],
            evidence_execution_ids=evidence_execution_ids,
            bulk_support=bulk_support,
            evidence_strength=evidence_strength,
            specificity=specificity,
            alternative_explanations=alternatives,
            missing_evidence=missing,
            interpretation=interpretation,
            downstream_literature_queries=[str(q) for q in _ensure_list(rp.get("downstream_literature_queries"))],
        ))
    return programs, None
