"""Pure, sandbox-free helpers for the post-execution review round.

A round ends when the bioinformatics agent elects to call report_findings, and
nothing downstream compares what was approved against what actually ran:
`PlanStep.status` has three readers and no writer, so every persisted plan stays
`pending` and a dropped method family leaves no trace anywhere.

These helpers close that gap by auditing the approved plan against EXECUTION
EVIDENCE — successful code executions, registered scores, mechanism programs,
saved figures — rather than against the agent's prose. Free-text overlap scoring
was tried and removed (see the deleted `utils/plan_retention.py`); it reported
paraphrase as absence. What survives here is narrower and binary: a step that
names a concrete tool or function is unmet only when that identifier appears in
no successful execution of the round. Two exemptions keep that honest — a step
the plan itself marked conditional, and a family the kernel demonstrably lacks.

Deterministic logic lives here so it is unit-testable without an LLM or a live
Jupyter kernel. Agent-coupled logic lives on the agents; orchestration lives in
`PostExecutionReviewMixin`.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from backend.services.figure_contract import (
    audit_figure_contract,
    collect_plan_figure_requirements,
    legacy_figure_opt_out,
    valid_text_only_exception,
)
from backend.services.execution_evidence import execution_supports_evidence
from backend.services.report.artifacts import (
    archival_render_available,
    is_report_eligible,
)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def compute_review_attempt_id(
    current_round: int,
    plan_id: str,
    planning_stage: str,
    evidence_execution_ids: Sequence[str],
    result_status: str,
) -> str:
    """Deterministic id for one post-execution review attempt.

    Stable across websocket reconnect, node retry (`@with_retry`, 3 attempts) and
    Snapshot V2 restore for the same execution; changes when the round, the
    approved plan, the stage, the cited evidence, or the terminal status changes.
    Evidence id order is normalized because the agent cites them unordered.
    """
    payload = json.dumps(
        {
            "round": int(current_round or 0),
            "plan_id": str(plan_id or ""),
            "stage": str(planning_stage or "full"),
            "evidence": sorted(str(e) for e in (evidence_execution_ids or [])),
            "status": str(result_status or ""),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Method-family detection
#
# Families are TOOLS, never biology: the platform is general-purpose, so no
# tissue, disease, organism or pathway may appear here. A family is worth listing
# only when a plan step that names it cannot be satisfied by anything else — that
# non-substitutability is what makes its absence a finding rather than a style
# difference.
#
# Matched as SUBSTRINGS, because the code and error text this is read against
# carry identifiers inside larger expressions -- "dc.run_", "li.mt.", "milo.",
# "sc.external.pp.harmony" are not words and tokenising the table would drop
# them. The cost is that a bare token which is also English over-matches the one
# prose input, plan-step text; the answer is to qualify that token, not to change
# the matcher for all three inputs.
#
# NOT the same taxonomy as the frozensets behind
# services.statistical_gates.infer_method_kind, and not a drifted copy of them:
# this asks whether a planned STEP ran, that asks what kind of statistical claim
# a registered score licenses. Ten of the families here have no counterpart
# there, and the distinction this collapses into one "enrichment" bucket is
# exactly the one that side must keep -- gseapy.prerank, gseapy.enrichr and
# gseapy.ssgsea are three inference kinds from one package. The tables are also
# right to disagree about members: betabinom and quasibinomial belong in
# composition_model here, because the step ran, and must stay out of _DA there,
# because they fit one population at a time. Do not unify them.
# ---------------------------------------------------------------------------

METHOD_SIGNATURES: Dict[str, Tuple[str, ...]] = {
    "bulk_de": ("pydeseq2", "deseqdataset", "deseqstats", "deseq2", "edger",
                "exacttest", "glmqlfit", "limma", "voom"),
    "marker_de": ("rank_genes_groups",),
    "pseudobulk": ("pseudobulk", "get_pseudobulk", "aggregate_and_filter",
                   "sum_by", "groupby_sum"),
    "cell_communication": ("liana", "li.mt.", "rank_aggregate", "cellphonedb",
                           "cellchat", "ligrec", "nichenet"),
    "pathway_activity": ("decoupler", "dc.run_", "dc.mt.", "run_ulm", "run_mlm",
                         "run_wsum", "run_consensus", "progeny"),
    "enrichment": ("gseapy", "prerank", "enrichr", "ssgsea", "gsva",
                   "enrich_analysis", "over_representation"),
    # Tool identifiers only. "regulon" itself is prose — decoupler's input network
    # is called a regulon set, so matching the bare word made "run decoupler on the
    # collected regulons" demand pySCENIC and report a satisfied step as unmet.
    "regulon": ("pyscenic", "grnboost", "aucell"),
    "trajectory": ("sc.tl.dpt", "diffmap", "sc.tl.paga", "palantir",
                   "slingshot", "pseudotime", "monocle"),
    "rna_velocity": ("scvelo", "velocyto", "velocity_graph", "cellrank"),
    "composition_model": ("sccoda", "scanpro", "propeller", "milopy", "milo.",
                          "pertpy", "dirichlet", "betabinom", "quasibinomial",
                          "glmmtmb"),
    "deconvolution": ("cell2location", "cibersort", "bayesprism", "tangram",
                      "spotlight", "scaden", "music"),
    "gene_scoring": ("score_genes", "addmodulescore", "sc.tl.score_genes"),
    "integration": ("harmonypy", "sc.external.pp.harmony", "scanorama", "bbknn",
                    "scvi", "combat"),
    # Qualified for the same reason "regulon" was removed above: bare "ingest" is
    # the ordinary verb for loading data, which is step one of nearly every plan,
    # so "Ingest the filtered matrix" read as a commitment nothing ran -- and an
    # unmet step is what sends the supplementary pass. It was also the only token
    # here claimed by two families, so it charged both at once. Dropped from
    # integration outright rather than qualified twice: sc.tl.ingest projects
    # labels from a reference, which is this family, not batch correction.
    "reference_mapping": ("tl.ingest", ".ingest(", "scarches", "symphony",
                          "label_transfer", "transfer_anchors"),
    "doublet_detection": ("scrublet", "doubletdetection", "solo"),
    "copy_number": ("infercnv", "copykat", "numbat"),
    "survival": ("lifelines", "coxph", "kaplan"),
}

# A step the plan itself hedged is not a dropped step. These were the dominant
# false-positive source when auditing real runs: plans routinely write
# "scCODA if available" or "propeller if assumptions are reasonable", and the
# executor is entitled to decline.
_CONDITIONAL_MARKERS: Tuple[str, ...] = (
    "if available", "if present", "if applicable", "if feasible", "if possible",
    "if assumptions", "if data quality", "if the data support", "if supported",
    "if metadata", "if replicate", "where available", "where possible",
    "when available", "subject to", "optional", "optionally", "if installed",
    "if the package", "as available", "if sufficient", "permitting",
)

# Evidence that the kernel cannot run a family at all. A missing package is an
# environment fact, not a compliance failure.
_UNAVAILABLE_ERROR_MARKERS: Tuple[str, ...] = (
    "modulenotfounderror", "importerror", "no module named",
    "cannot import name", "is not installed",
)

#: The module an import failure actually named. Python states it directly in the
#: common forms; when it does not -- "cannot import name 'x'" carries the symbol
#: but not always its module -- there is nothing to extract and the fallback
#: applies.
_FAILED_MODULE_PATTERNS: Tuple[Any, ...] = (
    re.compile(r"no module named ['\"]?([\w.]+)", re.I),
    re.compile(r"cannot import name ['\"][\w.]+['\"] from ['\"]?([\w.]+)", re.I),
    re.compile(r"['\"]?([\w.]+)['\"]? is not installed", re.I),
)

#: An import statement, which is the only part of a cell that can be the subject
#: of an import failure.
_IMPORT_LINE = re.compile(r"^\s*(?:import|from)\s+", re.M)

_MAX_TEXT = 4000


# ---------------------------------------------------------------------------
# Planned figure detection
#
# Figure TYPE inference and matching belong to services.figure_contract. This
# layer only adds two legacy plan semantics that contract intentionally does not
# invent: explicit text/no-figure waivers, and a generic "make a plot" promise
# whose geometry was never named. Keeping that boundary avoids a second figure
# taxonomy beside the structured PlanStep.figure_requirements contract.
# ---------------------------------------------------------------------------

_FIGURE_GENERIC_RE = re.compile(
    r"\b(?:publication[- ](?:quality|ready)\s+)?"
    r"(?:figure|figures|plot|plots|plotted|plotting|visuali[sz]e|visuali[sz]ed|"
    r"visuali[sz]ing|visuali[sz]ation|visuali[sz]ations|chart|charts|charting)\b",
    re.I,
)

# These clauses are explicit waivers, not merely an absence of plotting words.
# If a plan author says the deliverable is text/table only, or explicitly says
# not to create a figure, the audit must not silently turn it back into one.
_FIGURE_OPTOUT_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:text|prose|narrative|table|tabular)[ -]?only\b", re.I),
    re.compile(r"\bonly\s+(?:a\s+)?(?:text|prose|narrative|table|tabular)\b", re.I),
    re.compile(r"\b(?:no|without)\s+(?:publication[- ](?:quality|ready)\s+)?"
               r"(?:figure|figures|plot|plots|visuali[sz]ation|visuali[sz]ations|chart|charts)\b",
               re.I),
    re.compile(r"\bdo\s+not\s+(?:generate|create|produce|render|include|make|save|draw)"
               r"\b.{0,60}\b(?:figure|figures|plot|plots|visuali[sz]ation|"
               r"visuali[sz]ations|chart|charts)\b", re.I),
    re.compile(r"\bdo\s+not\s+(?:plot|visuali[sz]e|chart)\b", re.I),
)

# Optionality is scoped to the artifact channel. In a mixed step such as
# "run DE and optionally draw a volcano plot", the plot is waived but the DE
# method is still audited below.
_OPTIONAL_FIGURE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\boptional(?:ly)?\b[^.!?;\n]{0,60}\b(?:figure|figures|plot|plots|"
               r"visuali[sz]ation|visuali[sz]ations|chart|charts)\b", re.I),
    re.compile(r"\b(?:figure|figures|plot|plots|visuali[sz]ation|visuali[sz]ations|"
               r"chart|charts)\b[^.!?;\n]{0,60}\boptional\b", re.I),
    re.compile(r"\b(?:figure|figures|plot|plots|visuali[sz]ation|visuali[sz]ations|"
               r"chart|charts)\b[^.!?;\n]{0,40}\bif\s+(?:useful|helpful|needed|warranted|"
               r"informative|space permits|time permits)\b", re.I),
)

_OPTIONAL_QUALIFIER_RE = re.compile(
    r"\b(?:optional(?:ly)?|if\s+(?:useful|helpful|needed|warranted|informative|"
    r"space permits|time permits))\b",
    re.I,
)


def _step_text(step: Any) -> str:
    """Searchable text of one plan step: title, method and expected output."""
    parts = [
        str(getattr(step, "title", "") or ""),
        str(getattr(step, "method", "") or ""),
        str(getattr(step, "expected_output", "") or ""),
        str(getattr(step, "input", "") or ""),
    ]
    return " ".join(parts)[:_MAX_TEXT].lower()


def _figure_step_text(step: Any) -> str:
    """Plan text relevant to a deliverable, including its rationale."""
    return (
        _step_text(step) + " " + str(getattr(step, "rationale", "") or "").lower()
    )[:_MAX_TEXT]


def _step_figure_requirements(
    step: Any,
    stage: str = "",
    plan_id: str = "",
    plan_created_at: Any = None,
    plan_revision_of: Optional[str] = None,
    plan_phase: str = "",
    inherited_plan_ids_by_step: Optional[Mapping[int, Sequence[str]]] = None,
) -> List[Any]:
    """Read structured or legacy-named requirements from the shared contract."""
    return collect_plan_figure_requirements(SimpleNamespace(
        id=plan_id,
        created_at=plan_created_at,
        revision_of=plan_revision_of,
        steps=[step],
        stage=stage,
        phase=plan_phase,
    ), inherited_plan_ids_by_step=inherited_plan_ids_by_step)


def _figure_requirement_is_optional(step: Any, requirements: Sequence[Any]) -> bool:
    text = _figure_step_text(step)
    if any(pattern.search(text) for pattern in _OPTIONAL_FIGURE_PATTERNS):
        return True
    return bool(requirements and _OPTIONAL_QUALIFIER_RE.search(text))


def _figure_requirement_is_waived(
    step: Any,
    requirements: Sequence[Any],
    *,
    plan_phase: str = "",
) -> bool:
    """Whether the plan explicitly makes its visual output unnecessary."""
    # Structured deliverables are authoritative.  Neither a contradictory
    # boolean nor stray prose such as "optional" may erase them.
    if requirements:
        return False
    if valid_text_only_exception(step, plan_phase=plan_phase):
        return True
    # Prose opt-outs exist solely for persisted plans that predate the
    # structured visual-policy fields. Keep that compatibility decision in the
    # shared contract so execution and post-execution cannot diverge.
    return legacy_figure_opt_out(step)


def extract_planned_figure_kinds(step: Any) -> Set[str]:
    """Return required figure types using the shared figure contract."""
    requirements = _step_figure_requirements(step)
    if _figure_requirement_is_waived(step, requirements):
        return set()
    if requirements:
        return {str(requirement.figure_type) for requirement in requirements}
    return {"figure"} if _FIGURE_GENERIC_RE.search(_figure_step_text(step)) else set()


def _artifact_value(artifact: Any, name: str, default: Any = None) -> Any:
    """Read a model field from both live Pydantic objects and legacy dicts."""
    if isinstance(artifact, Mapping):
        return artifact.get(name, default)
    return getattr(artifact, name, default)


def _artifact_for_eligibility(artifact: Any) -> Any:
    """Give the shared report predicate attribute access to legacy dict state."""
    if isinstance(artifact, Mapping):
        return SimpleNamespace(**dict(artifact))
    return artifact


def _is_publication_figure_record(artifact: Any) -> bool:
    """Distinguish figure records from report-eligible table-like artifacts."""
    sentinel = object()
    return all(
        _artifact_value(artifact, field_name, sentinel) is not sentinel
        for field_name in ("figure_type", "png_path", "pdf_path")
    )


def eligible_current_round_figures(
    publication_figures: Optional[Sequence[Any]],
    round_number: int,
    *,
    target_lineage_id: str = "main",
) -> List[Any]:
    """Current-round ``PublicationFigure`` records fit for a final report.

    Round zero is not treated as a wildcard. Older sessions deserialize figures
    without round provenance as zero; accepting those in a later round would let
    a stale figure complete a new plan step. Dict-shaped legacy records remain
    readable, but an unattributed record cannot prove current-round work.
    """
    current = int(round_number or 0)
    target_lineage = str(target_lineage_id or "main")
    eligible: List[Any] = []
    for figure in publication_figures or []:
        if not _is_publication_figure_record(figure):
            continue
        if str(_artifact_value(figure, "lineage_id", None) or "main") != target_lineage:
            continue
        try:
            artifact_round = int(_artifact_value(figure, "round_number", 0) or 0)
        except (TypeError, ValueError):
            artifact_round = 0
        if artifact_round != current:
            continue
        candidate = _artifact_for_eligibility(figure)
        if is_report_eligible(candidate) and archival_render_available(candidate):
            eligible.append(figure)
    return eligible


def _artifact_label(artifact: Any) -> str:
    return str(
        _artifact_value(artifact, "name", "")
        or _artifact_value(artifact, "id", "")
        or "unnamed figure"
    )[:300]


def _legacy_figure_not_older_than_plan(artifact: Any, plan_created_at: Any) -> bool:
    """Apply the shared contract's bounded no-plan-id compatibility rule."""
    artifact_created_at = _artifact_value(artifact, "created_at", None)
    if not isinstance(plan_created_at, datetime) or not isinstance(
        artifact_created_at, datetime
    ):
        return True
    try:
        return artifact_created_at >= plan_created_at
    except TypeError:
        return artifact_created_at.replace(tzinfo=None) >= plan_created_at.replace(
            tzinfo=None
        )


def _generic_figure_evidences_step(
    step: Any,
    eligible_figures: Sequence[Any],
    *,
    plan_id: str = "",
    plan_created_at: Any = None,
    inherited_plan_ids: Sequence[str] = (),
    allow_unattributed_legacy: bool = True,
) -> bool:
    """Match a geometry-free legacy promise without stealing another step's figure.

    Once any current-round producer supplies ``plan_step_id``, attribution is an
    available evidence channel and only an exact match counts. The permissive
    any-figure fallback is retained solely for collections where every figure
    predates that field.
    """
    accepted_plan_ids = {
        str(value) for value in (plan_id, *inherited_plan_ids) if str(value or "")
    }
    result_figures = [
        figure for figure in eligible_figures
        if not bool(_artifact_value(figure, "diagnostic", False))
        and str(_artifact_value(figure, "purpose", "result") or "result").lower()
        not in {"baseline", "diagnostic"}
        and str(_artifact_value(figure, "figure_type", "") or "").lower()
        not in {"qc", "diagnostic", "baseline"}
        and (
            str(_artifact_value(figure, "plan_id", "") or "") in accepted_plan_ids
            or (
                not str(_artifact_value(figure, "plan_id", "") or "")
                and allow_unattributed_legacy
                and _legacy_figure_not_older_than_plan(figure, plan_created_at)
            )
        )
    ]
    attributed = [
        figure for figure in eligible_figures
        if _artifact_value(figure, "plan_step_id", None) is not None
        and figure in result_figures
    ]
    if not attributed:
        return bool(result_figures)
    step_id = int(getattr(step, "id", 0) or 0)
    for figure in attributed:
        try:
            if int(_artifact_value(figure, "plan_step_id")) == step_id:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _audit_step_figure_requirements(
    step: Any,
    eligible_figures: Sequence[Any],
    round_number: int,
    planning_stage: str = "",
    plan_id: str = "",
    plan_created_at: Any = None,
    plan_revision_of: Optional[str] = None,
    plan_phase: str = "",
    inherited_plan_ids_by_step: Optional[Mapping[int, Sequence[str]]] = None,
    target_lineage_id: str = "main",
) -> Tuple[Set[str], List[str]]:
    """Return (planned types, missing types) from the shared contract."""
    requirements = _step_figure_requirements(
        step,
        planning_stage,
        plan_id,
        plan_created_at,
        plan_revision_of,
        plan_phase,
        inherited_plan_ids_by_step,
    )
    if _figure_requirement_is_waived(
        step, requirements, plan_phase=plan_phase
    ):
        return set(), []
    if not requirements:
        planned = {"figure"} if _FIGURE_GENERIC_RE.search(_figure_step_text(step)) else set()
        step_id = int(getattr(step, "id", 0) or 0)
        inherited = tuple((inherited_plan_ids_by_step or {}).get(step_id, ()))
        evidenced = _generic_figure_evidences_step(
            step,
            eligible_figures,
            plan_id=plan_id,
            plan_created_at=plan_created_at,
            inherited_plan_ids=inherited,
            allow_unattributed_legacy=not bool(plan_revision_of),
        )
        return planned, ([] if not planned or evidenced else ["figure"])

    # Dict-shaped restored figures are converted only for this read; live model
    # objects and persisted state are untouched. The strict current-round/report
    # filter above runs before the shared matcher, so its legacy round-zero
    # compatibility cannot let an older artifact satisfy this round's plan.
    state = SimpleNamespace(
        current_round=int(round_number or 0),
        publication_figures=[_artifact_for_eligibility(fig) for fig in eligible_figures],
        restore_branch_id=str(target_lineage_id or "main"),
    )
    contract = audit_figure_contract(
        state,
        plan=SimpleNamespace(
            id=plan_id,
            created_at=plan_created_at,
            revision_of=plan_revision_of,
            steps=[step],
            stage=planning_stage,
            phase=plan_phase,
        ),
        include_stage_baseline=False,
        inherited_plan_ids_by_step=inherited_plan_ids_by_step,
    )
    planned = {str(requirement.figure_type) for requirement in contract.requirements}
    missing = sorted({
        str(item.requirement.figure_type) for item in contract.missing
    })
    return planned, missing


def step_is_conditional(step: Any) -> bool:
    """True when the step's own wording licenses skipping it."""
    text = _step_text(step)
    return any(marker in text for marker in _CONDITIONAL_MARKERS)


def _method_requirement_is_conditional(step: Any, planned_families: Set[str]) -> bool:
    """Apply optionality to methods only when it actually precedes a method.

    ``step_is_conditional`` remains the broad public description of the whole
    step. The audit needs a narrower answer because "run DE and optionally plot"
    waives the artifact, not the DE computation.
    """
    text = _step_text(step)
    non_optional_markers = (
        marker for marker in _CONDITIONAL_MARKERS
        if marker not in ("optional", "optionally")
    )
    if any(marker in text for marker in non_optional_markers):
        return True
    optional_matches = list(re.finditer(r"\boptional(?:ly)?\b", text, re.I))
    if not optional_matches:
        return False
    # "Optionally run scCODA and plot ..." makes the computation optional too;
    # "run scCODA and optionally plot ..." does not. Family signatures are the
    # same exact identifiers used to establish the method requirement.
    method_positions = [
        match.start()
        for family in planned_families
        for token in METHOD_SIGNATURES.get(family, ())
        for match in re.finditer(re.escape(token), text)
    ]

    def _governs(optional_match: re.Match[str], method_position: int) -> bool:
        if not 0 <= method_position - optional_match.end() <= 60:
            return False
        between = text[optional_match.end():method_position]
        if re.search(r"[.!?;\n]", between):
            return False
        return not re.search(
            r"\b(?:then|but|however|afterwards?|subsequently)\b",
            between,
            re.I,
        )

    return bool(method_positions) and all(
        any(
            _governs(optional_match, method_position)
            for optional_match in optional_matches
        )
        for method_position in method_positions
    )


def extract_planned_methods(step: Any) -> Set[str]:
    """Method families a step commits to, by identifier match on its own text."""
    text = _step_text(step)
    return {
        family
        for family, tokens in METHOD_SIGNATURES.items()
        if any(token in text for token in tokens)
    }


def executed_method_families(code_history: Sequence[Any], round_number: int) -> Set[str]:
    """Families evidenced by a SUCCESSFUL execution in this round.

    A failed cell proves the family was attempted, not that it produced anything,
    so only error-free executions count. Round filtering keeps a family that ran
    three rounds ago from vouching for this round's plan.
    """
    found: Set[str] = set()
    for execution in code_history or []:
        if getattr(execution, "error", None):
            continue
        exec_round = int(getattr(execution, "round_number", 0) or 0)
        if round_number and exec_round and exec_round != int(round_number):
            continue
        code = str(getattr(execution, "code", "") or "").lower()
        if not code:
            continue
        for family, tokens in METHOD_SIGNATURES.items():
            if family in found:
                continue
            if any(token in code for token in tokens):
                found.add(family)
    return found


def unavailable_method_families(code_history: Sequence[Any]) -> Set[str]:
    """Families whose absence the kernel itself explained (import failure).

    Not round-filtered: a package missing in round 5 was missing in round 2, and
    an earlier import error is still valid evidence about the environment.

    Scoped to what the failure was actually about. Matching against the whole
    cell source let one import failure condemn every family named anywhere in
    that cell -- and a cell that imports several packages before doing anything
    is the ordinary shape of a first cell. The verdict is permanent and the PI is
    instructed never to propose an unavailable method, so a family swept in by
    co-location is retired from the entire analysis on evidence that never
    mentioned it.

    Two narrowing steps, in order. Python usually names the module it could not
    load, and that name alone is the evidence. When it does not -- "cannot import
    name 'x'" need not carry a module -- fall back to the cell's import lines,
    which is still far narrower than its body: the subject of an import failure
    is always an import statement, never a call three lines further down.
    """
    unavailable: Set[str] = set()
    for execution in code_history or []:
        error = str(getattr(execution, "error", "") or "").lower()
        if not error or not any(m in error for m in _UNAVAILABLE_ERROR_MARKERS):
            continue
        named = [m.group(1) for p in _FAILED_MODULE_PATTERNS
                 for m in [p.search(error)] if m]
        if named:
            haystack = error + " " + " ".join(named)
        else:
            code = str(getattr(execution, "code", "") or "").lower()
            imports = [ln for ln in code.splitlines() if _IMPORT_LINE.match(ln)]
            haystack = error + " " + " ".join(imports)
        for family, tokens in METHOD_SIGNATURES.items():
            if any(token in haystack for token in tokens):
                unavailable.add(family)
    return unavailable


# ---------------------------------------------------------------------------
# Plan audit  (feature A: the only writer of PlanStep.status)
# ---------------------------------------------------------------------------

STATUS_COMPLETED = "completed"
STATUS_SKIPPED = "skipped"
STATUS_PENDING = "pending"
