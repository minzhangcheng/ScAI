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
                    "scvi", "combat", "ingest"),
    "reference_mapping": ("ingest", "scarches", "symphony", "label_transfer",
                          "transfer_anchors"),
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
    """
    unavailable: Set[str] = set()
    for execution in code_history or []:
        error = str(getattr(execution, "error", "") or "").lower()
        if not error or not any(m in error for m in _UNAVAILABLE_ERROR_MARKERS):
            continue
        haystack = error + " " + str(getattr(execution, "code", "") or "").lower()
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

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "status": self.status,
            "reason": self.reason,
            "planned_families": list(self.planned_families),
            "missing_families": list(self.missing_families),
            "verifiable": self.verifiable,
        }


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

    def as_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "steps": [s.as_dict() for s in self.steps]}


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
# Trigger gate
# ---------------------------------------------------------------------------

def should_run_post_execution_review(
    audit: PlanAudit,
    admissions: Sequence[str],
    result_status: str,
    is_terminal_round: bool,
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
      4. this is the round after which no further analysis can ever happen —
         the terminal round, whose follow-up proposals are otherwise discarded
         with certainty rather than probability.
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

    if is_terminal_round:
        return (True, "Terminal round: no later round can carry follow-up work.")

    return (False, "Plan fully evidenced, no failure admissions, and a later round remains.")


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
