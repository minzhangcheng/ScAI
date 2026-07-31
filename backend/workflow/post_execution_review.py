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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


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


def _step_text(step: Any) -> str:
    """Searchable text of one plan step: title, method and expected output."""
    parts = [
        str(getattr(step, "title", "") or ""),
        str(getattr(step, "method", "") or ""),
        str(getattr(step, "expected_output", "") or ""),
        str(getattr(step, "input", "") or ""),
    ]
    return " ".join(parts)[:_MAX_TEXT].lower()


def step_is_conditional(step: Any) -> bool:
    """True when the step's own wording licenses skipping it."""
    text = _step_text(step)
    return any(marker in text for marker in _CONDITIONAL_MARKERS)


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


@dataclass
class StepAudit:
    """Verdict for one plan step."""
    step_id: int
    title: str
    status: str
    reason: str
    planned_families: List[str] = field(default_factory=list)
    missing_families: List[str] = field(default_factory=list)
    verifiable: bool = True


@dataclass
class PlanAudit:
    """Aggregate plan-versus-evidence verdict for one execution."""
    steps: List[StepAudit] = field(default_factory=list)
    executed_families: List[str] = field(default_factory=list)
    unavailable_families: List[str] = field(default_factory=list)

    @property
    def unmet(self) -> List[StepAudit]:
        """Steps that named a family, could be checked, and show no evidence."""
        return [s for s in self.steps if s.status == STATUS_PENDING and s.verifiable]

    @property
    def completed(self) -> List[StepAudit]:
        return [s for s in self.steps if s.status == STATUS_COMPLETED]

    @property
    def skipped(self) -> List[StepAudit]:
        return [s for s in self.steps if s.status == STATUS_SKIPPED]

    @property
    def unverifiable(self) -> List[StepAudit]:
        """Steps with no machine-checkable identifier. Never counted as gaps —
        that conflation is what made free-text coverage scoring unusable."""
        return [s for s in self.steps if not s.verifiable]

    def summary(self) -> Dict[str, Any]:
        return {
            "n_steps": len(self.steps),
            "n_completed": len(self.completed),
            "n_skipped": len(self.skipped),
            "n_unmet": len(self.unmet),
            "n_unverifiable": len(self.unverifiable),
            "unmet_titles": [s.title for s in self.unmet][:20],
            "unmet_families": sorted({f for s in self.unmet for f in s.missing_families}),
            "executed_families": sorted(self.executed_families),
            "unavailable_families": sorted(self.unavailable_families),
        }


def audit_plan_execution(
    plan: Any,
    code_history: Sequence[Any],
    round_number: int = 0,
    structured_families: Optional[Set[str]] = None,
) -> PlanAudit:
    """Audit an approved plan against this round's execution evidence.

    `structured_families` lets the caller vouch for families from structured
    state instead of code text (e.g. a registered score proves `gene_scoring`),
    which is the more reliable channel where it exists.
    """
    audit = PlanAudit()
    if plan is None or not getattr(plan, "steps", None):
        return audit

    executed = set(executed_method_families(code_history, round_number))
    executed |= set(structured_families or set())
    unavailable = unavailable_method_families(code_history)
    audit.executed_families = sorted(executed)
    audit.unavailable_families = sorted(unavailable)

    for step in plan.steps:
        planned = extract_planned_methods(step)
        step_id = int(getattr(step, "id", 0) or 0)
        title = str(getattr(step, "title", "") or "")[:300]

        if not planned:
            audit.steps.append(StepAudit(
                step_id=step_id, title=title, status=STATUS_PENDING,
                reason="No machine-checkable method identifier in the step text.",
                verifiable=False,
            ))
            continue

        missing = sorted(planned - executed)
        if not missing:
            audit.steps.append(StepAudit(
                step_id=step_id, title=title, status=STATUS_COMPLETED,
                reason="Evidenced by a successful execution of: " + ", ".join(sorted(planned)),
                planned_families=sorted(planned),
            ))
            continue

        blocked = [f for f in missing if f in unavailable]
        if blocked and len(blocked) == len(missing):
            audit.steps.append(StepAudit(
                step_id=step_id, title=title, status=STATUS_SKIPPED,
                reason="Required tooling unavailable in this kernel: " + ", ".join(blocked),
                planned_families=sorted(planned), missing_families=missing,
            ))
            continue

        if step_is_conditional(step):
            audit.steps.append(StepAudit(
                step_id=step_id, title=title, status=STATUS_SKIPPED,
                reason="Step is conditional in the approved plan; not evidenced: "
                       + ", ".join(missing),
                planned_families=sorted(planned), missing_families=missing,
            ))
            continue

        audit.steps.append(StepAudit(
            step_id=step_id, title=title, status=STATUS_PENDING,
            reason="No successful execution evidences: " + ", ".join(missing),
            planned_families=sorted(planned), missing_families=missing,
        ))

    return audit


def apply_audit_to_plan(plan: Any, audit: PlanAudit, now: Optional[datetime] = None) -> int:
    """Write the audit onto the plan's steps — the missing writer of
    `PlanStep.status` / `executed_at`. Returns the number of steps updated.

    `executed_at` is stamped only for `completed`, so a timestamp always means
    "evidence was found", never "we looked".
    """
    if plan is None or not getattr(plan, "steps", None) or not audit.steps:
        return 0
    stamp = now or datetime.now()
    verdicts = {s.step_id: s for s in audit.steps}
    updated = 0
    for step in plan.steps:
        verdict = verdicts.get(int(getattr(step, "id", 0) or 0))
        if verdict is None:
            continue
        step.status = verdict.status
        if verdict.status == STATUS_COMPLETED and getattr(step, "executed_at", None) is None:
            step.executed_at = stamp
        updated += 1
    return updated


# ---------------------------------------------------------------------------
# Failure admissions in the agent's own output
# ---------------------------------------------------------------------------

_ADMISSION_PATTERNS: Tuple[str, ...] = (
    r"\bcould not (?:be )?(?:run|complete|execute|perform)",
    r"\bwas not (?:run|completed|executed|performed|possible)",
    r"\bnot (?:executed|performed|available|installed|possible)\b",
    r"\bfailed to (?:run|complete|execute|converge|import)",
    r"\bfell back\b|\bfallback (?:was )?used\b|\busing a fallback\b",
    r"\bskipped\b",
    r"\bunavailable\b",
    r"\bbudget exhausted\b|\bstep limit\b|\btime(?:d)? out\b",
    r"\bcould not be validated\b",
)
_ADMISSION_RE = re.compile("|".join(_ADMISSION_PATTERNS), re.IGNORECASE)
_MAX_ADMISSIONS = 20


def collect_failure_admissions(
    findings: Sequence[Any],
    analysis_log: Sequence[Any],
) -> List[str]:
    """Explicit "this did not run" statements the agent itself recorded.

    Read from findings and from `report_findings` summaries only. This is the
    agent's own admission, so it needs no interpretation — unlike its silence,
    which is exactly what the plan audit is for.
    """
    admissions: List[str] = []
    seen: Set[str] = set()

    def _add(text: str) -> None:
        for sentence in re.split(r"(?<=[.;\n])\s+", str(text or "")):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 12:
                continue
            if not _ADMISSION_RE.search(sentence):
                continue
            key = sentence.lower()[:160]
            if key in seen:
                continue
            seen.add(key)
            admissions.append(sentence[:400])
            if len(admissions) >= _MAX_ADMISSIONS:
                return

    for finding in findings or []:
        if len(admissions) >= _MAX_ADMISSIONS:
            break
        _add(finding if isinstance(finding, str) else json.dumps(finding, ensure_ascii=False))

    for step in analysis_log or []:
        if len(admissions) >= _MAX_ADMISSIONS:
            break
        if isinstance(step, dict):
            if step.get("step_type") != "report_findings":
                continue
            _add(step.get("findings_summary", ""))
        else:
            if getattr(step, "step_type", "") != "report_findings":
                continue
            _add(getattr(step, "findings_summary", "") or "")

    return admissions


# ---------------------------------------------------------------------------
# Structured evidence gaps  (observation only — see should_run_post_execution_review)
#
# A mechanism program is the closest thing this platform has to a stated claim:
# `evidence_execution_ids` is required by the tool schema, and `missing_evidence`
# (not required, but populated in every program of every run measured so far) is
# the program's own account of what it lacks. That makes it a candidate trigger
# for an EXPANSION pass — supplementary analysis on a topic whose plan completed
# cleanly — which the repair-oriented gate deliberately does not cover.
#
# It is deliberately NOT wired into the gate. `missing_evidence` was populated for
# 146 of 146 programs across the 9 sessions on disk, so a gate keyed on "a gap
# exists" would fire every round, and this project has twice had to rebuild a
# threshold designed from first principles (0.36.8-rc1, 0.36.8-rc5).
#
# These helpers LABEL TEXT. They do not decide whether a gap is closable, and the
# labels must not be read that way — measured on all 542 real items, a
# feasibility reading of them is wrong most of the time. 60.5% match no method
# vocabulary at all, and of the 65 that matched computational vocabulary only
# roughly 8-12 were actually runnable on data in hand: the rest are dominated by
# "replicate-aware"/"replicated pseudobulk" requests, which need per-sample or
# per-animal metadata the run may simply not have. Whether such an item is
# closable depends on the dataset, not on its wording, so no phrase list can
# decide it. Closing that gap properly means having the PRODUCER say so — e.g. a
# structured `missing_evidence` entry in the register_mechanism_programs schema
# that marks whether the item is closable with data already loaded — rather than
# inferring it afterwards.
# ---------------------------------------------------------------------------

# Method vocabulary, never biology: naming an assay class is domain-neutral in the
# sense the platform requires (no tissue, disease, organism or pathway). It marks
# what the text MENTIONS, nothing more.
_NEEDS_NEW_DATA_MARKERS: Tuple[str, ...] = (
    # direct measurement
    "assay", "western", "qpcr", "rt-pcr", "elisa", "immunohisto", "immunostain",
    "immunofluor", "ihc", "rnascope", "in situ", "flow cytometry", "facs",
    "mass spec", "proteomic", "metabolomic", "lipidomic", "icp-ms",
    "elemental analysis", "respirometry", "seahorse", "oxygen consumption",
    "enzyme activity", "electron microscop", "histolog", "histopatholog",
    "staining", "stain ", "biochemical measurement", "protein level",
    # new material
    "additional sample", "new sample", "independent cohort", "validation cohort",
    "replicate librar", "replicated librar", "biological replicate",
    "additional animal", "more animal", "new sequencing", "additional sequencing",
    "additional section", "orthogonal validation", "orthogonal protein",
    # intervention
    "knockout", "knockdown", "overexpress", "rescue experiment", "inhibitor",
    "lineage tracing", "transgenic", "supplementation",
    # another modality or an external corpus
    "spatial transcriptom", "atac", "chip-seq", "cut&run", "methylation",
    "public dataset", "external dataset", "published dataset", "reference atlas",
)

# Positive signals that the gap is answerable by more analysis of data in hand.
_IN_SILICO_MARKERS: Tuple[str, ...] = (
    "concordance", "correlat", "pseudobulk", "subcluster", "module score",
    "enrichment", "deconvolut", "trajector", "composition", "adjust for",
    "control for", "stratif", "re-analy", "reanaly", "sensitivity analysis",
    "ambient", "doublet", "batch", "normali", "cross-validat", "bulk validation",
    "same-sign", "effect size", "confidence interval", "multiple testing",
    "permutation", "null model", "regress",
)

# Co-occurrence rules, for requests a contiguous phrase cannot catch. "Replicated
# snRNA libraries per condition for formal composition testing" asks for new
# sequencing, but the words are separated and the sentence also names a
# computation, so a phrase list scores it in-silico. Each tuple must ALL appear.
_NEEDS_NEW_DATA_PAIRS: Tuple[Tuple[str, ...], ...] = (
    ("replicat", "librar"),
    ("replicat", "sample"),
    ("replicat", "cohort"),
    ("replicat", "animal"),
    ("additional", "librar"),
    ("independent", "librar"),
    ("independent", "measurement"),
    ("matched", "tissue"),
    ("paired", "tissue"),
)

# Labels describe the VOCABULARY the item uses, not what it would take to close it.
# Named so that no caller can mistake one for a feasibility verdict.
GAP_MENTIONS_NEW_MEASUREMENT = "mentions_new_measurement"
GAP_MENTIONS_COMPUTATION = "mentions_computation"
GAP_UNLABELLED = "unlabelled"

_MAX_GAP_ITEMS = 60


def label_evidence_gap(text: str) -> str:
    """Label one `missing_evidence` item by the method vocabulary it uses.

    NOT a feasibility judgement — see the section comment. Measurement vocabulary
    wins over computational vocabulary, because "orthogonal protein validation of
    the correlation" is a bench request that merely mentions a computation. An item
    labelled `mentions_computation` is a candidate to look at by hand, not a task
    that can be dispatched.
    """
    lowered = str(text or "").lower()
    if not lowered.strip():
        return GAP_UNLABELLED
    if any(marker in lowered for marker in _NEEDS_NEW_DATA_MARKERS):
        return GAP_MENTIONS_NEW_MEASUREMENT
    if any(all(part in lowered for part in pair) for pair in _NEEDS_NEW_DATA_PAIRS):
        return GAP_MENTIONS_NEW_MEASUREMENT
    if any(marker in lowered for marker in _IN_SILICO_MARKERS):
        return GAP_MENTIONS_COMPUTATION
    return GAP_UNLABELLED


def _program_field(program: Any, name: str) -> Any:
    if isinstance(program, dict):
        return program.get(name)
    return getattr(program, name, None)


def collect_structured_evidence_gaps(programs: Sequence[Any]) -> Dict[str, Any]:
    """Record this round's stated evidence gaps with their vocabulary labels.

    Reads the mechanism programs the round registered and keeps the items verbatim
    — the raw text is the part worth having, since the labels are not a feasibility
    verdict. Bounded for storage. Purely descriptive: nothing in the trigger path
    reads this.
    """
    items: List[Dict[str, str]] = []
    programs_with_gaps = 0
    programs_total = 0
    unbound_programs: List[str] = []

    for program in programs or []:
        programs_total += 1
        label = str(_program_field(program, "label") or "")[:200]
        # `evidence_execution_ids` is required by the tool schema; a program without
        # one is a claim with no citable execution, which is worth seeing separately.
        if not (_program_field(program, "evidence_execution_ids") or []):
            unbound_programs.append(label)
        gaps = _program_field(program, "missing_evidence") or []
        if gaps:
            programs_with_gaps += 1
        for gap in gaps:
            text = str(gap or "").strip()
            if not text:
                continue
            items.append({
                "program": label,
                "gap": text[:400],
                "label": label_evidence_gap(text),
            })
            if len(items) >= _MAX_GAP_ITEMS:
                break
        if len(items) >= _MAX_GAP_ITEMS:
            break

    counts = {
        GAP_MENTIONS_COMPUTATION: sum(1 for i in items if i["label"] == GAP_MENTIONS_COMPUTATION),
        GAP_MENTIONS_NEW_MEASUREMENT: sum(1 for i in items if i["label"] == GAP_MENTIONS_NEW_MEASUREMENT),
        GAP_UNLABELLED: sum(1 for i in items if i["label"] == GAP_UNLABELLED),
    }
    return {
        "programs_total": programs_total,
        "programs_with_gaps": programs_with_gaps,
        "programs_without_evidence_ids": unbound_programs[:20],
        "counts": counts,
        "items": items,
    }


# ---------------------------------------------------------------------------
# Trigger gate
# ---------------------------------------------------------------------------

def should_run_post_execution_review(
    audit: PlanAudit,
    admissions: Sequence[str],
    result_status: str,
    is_last_direction_of_batch: bool,
    has_runtime: bool = True,
) -> Tuple[bool, str]:
    """Decide whether the bounded supplementary pass is warranted.

    Returns (run, reason). Deliberately NOT gated on the PI's opinion: across
    real runs `synthesize_round` returned `should_stop=False` in 39 of 40 calls
    and never proposed zero follow-ups, so "does the PI think anything is open?"
    fires every round and cannot terminate anything. Only hard structural
    signals qualify:

      1. the executor errored or stopped short of its own completion path;
      2. the agent admitted in writing that something did not run;
      3. the plan audit found a non-conditional step with no execution evidence;
      4. this is the last direction of the active batch, so analysis may stop
         after it — the round whose follow-up proposals are otherwise the most
         likely to be discarded.

    Signal 4 used to claim the round was TERMINAL. It cannot: whether another
    analysis round follows depends on `integration_synthesis` incrementing
    `integration_cycle` and the PI then choosing new directions, both of which
    happen after this decision. "Last direction of the batch" is the strongest
    decidable form, and it fires once per batch rather than once per session.
    """
    if not has_runtime:
        return (False, "No live analysis runtime available for a supplementary pass.")

    status = str(result_status or "").strip().lower()
    if status in ("error", "timeout", "max_steps_reached", "completed_with_fallback"):
        return (True, f"Execution ended as '{status}' rather than a clean completion.")

    if admissions:
        return (True, f"Agent recorded {len(admissions)} explicit not-run/failure admission(s).")

    unmet = audit.unmet
    if unmet:
        families = sorted({f for s in unmet for f in s.missing_families})
        return (True,
                f"{len(unmet)} approved step(s) show no execution evidence "
                f"({', '.join(families[:6])}).")

    if is_last_direction_of_batch:
        return (True,
                "Last direction of the active batch: analysis may stop after this "
                "round, so its follow-up work would be discarded.")

    return (False, "Plan fully evidenced, no failure admissions, and another direction remains.")


# ---------------------------------------------------------------------------
# Bounding the PI's review output and the executor's result
# ---------------------------------------------------------------------------

_MAX_TASK_LEN = 600
_MAX_GAP_LEN = 500
_MAX_LIST = 12


def _clean_list(values: Any, limit: int, item_limit: int) -> List[str]:
    """Drop blanks BEFORE applying the limit, so the cap counts real items.

    Slicing first would let a model's empty strings silently consume the task
    budget, which is the one number in this feature that has to mean what it says.
    """
    out: List[str] = []
    for value in (values or []):
        text = (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)).strip()
        if not text:
            continue
        out.append(text[:item_limit])
        if len(out) >= limit:
            break
    return out


def sanitize_review_plan(plan: Any, max_tasks: int) -> Dict[str, Any]:
    """Bound the PI's review output. The task cap is the cost ceiling: an LLM PI
    asked what else could be done answers with 1-15 items every single time, so
    the number of tasks admitted is a hard constant here, not the PI's choice."""
    if not isinstance(plan, dict):
        return {"gaps": [], "supplementary_tasks": [], "out_of_scope": [],
                "verdict": "no_action", "rationale": "Review returned no structured plan."}
    cap = max(0, int(max_tasks))
    tasks = _clean_list(plan.get("supplementary_tasks"), cap, _MAX_TASK_LEN)
    verdict = str(plan.get("verdict", "") or "").strip().lower()
    if verdict not in ("supplement", "no_action"):
        verdict = "supplement" if tasks else "no_action"
    return {
        "gaps": _clean_list(plan.get("gaps"), _MAX_LIST, _MAX_GAP_LEN),
        "supplementary_tasks": tasks,
        "out_of_scope": _clean_list(plan.get("out_of_scope"), _MAX_LIST, _MAX_GAP_LEN),
        "verdict": verdict,
        "rationale": str(plan.get("rationale", "") or "")[:2000],
    }


_MAX_FINDINGS = 40
_MAX_FIGURES = 12


def sanitize_review_result(result: Any) -> Dict[str, Any]:
    """Bound a supplementary-execution result for storage and transport.

    Returns ONLY JSON-serializable keys. The executor also hands back
    `code_history` / `analysis_log` as pydantic objects for the caller to merge
    into evidence provenance; carrying them through here would put non-serializable
    values into a dict that gets prompted to the PI, sent over the websocket and
    written into the session file.
    """
    if not isinstance(result, dict):
        return {"status": "failed", "summary": "Supplementary pass returned no structured result.",
                "findings": [], "figures": [], "warnings": ["non-dict result"]}
    return {
        "status": str(result.get("status", "completed") or "completed"),
        "summary": str(result.get("summary", "") or "Supplementary analysis completed.")[:4000],
        "findings": _clean_list(result.get("findings"), _MAX_FINDINGS, _MAX_TASK_LEN),
        "figures": [str(f) for f in list(result.get("figures") or [])[:_MAX_FIGURES]],
        "warnings": _clean_list(result.get("warnings"), _MAX_LIST, _MAX_GAP_LEN),
    }


def empty_review_meta(status: str, reason: str, attempt_id: str = "") -> Dict[str, Any]:
    """A review record for the paths that decline to run one."""
    return {
        "review_attempt_id": attempt_id,
        "review_status": status,
        "review_reason": reason,
        "review_plan": {},
        "review_result": {},
        "plan_audit": {},
        "supplementary_findings": [],
    }
