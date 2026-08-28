"""Plan audit records, evidence gaps, review triggers, and sanitizers."""
from __future__ import annotations

from . import methods as _methods
globals().update(
    {name: value for name, value in vars(_methods).items() if not name.startswith("__")}
)

@dataclass
class StepAudit:
    """Verdict for one plan step."""
    step_id: int
    title: str
    status: str
    reason: str
    planned_families: List[str] = field(default_factory=list)
    missing_families: List[str] = field(default_factory=list)
    planned_figure_kinds: List[str] = field(default_factory=list)
    missing_figure_kinds: List[str] = field(default_factory=list)
    verifiable: bool = True


@dataclass
class PlanAudit:
    """Aggregate plan-versus-evidence verdict for one execution."""
    steps: List[StepAudit] = field(default_factory=list)
    executed_families: List[str] = field(default_factory=list)
    unavailable_families: List[str] = field(default_factory=list)
    eligible_figures: List[str] = field(default_factory=list)

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
        figure_steps = [s for s in self.steps if s.planned_figure_kinds]
        figure_unmet = [
            s for s in self.unmet if s.missing_figure_kinds
        ]
        return {
            "n_steps": len(self.steps),
            "n_completed": len(self.completed),
            "n_skipped": len(self.skipped),
            "n_unmet": len(self.unmet),
            "n_unverifiable": len(self.unverifiable),
            "n_figure_steps": len(figure_steps),
            "n_figure_unmet": len(figure_unmet),
            "unmet_titles": [s.title for s in self.unmet][:20],
            "unmet_families": sorted({f for s in self.unmet for f in s.missing_families}),
            "unmet_figure_kinds": sorted({
                kind for s in figure_unmet for kind in s.missing_figure_kinds
            }),
            "executed_families": sorted(self.executed_families),
            "unavailable_families": sorted(self.unavailable_families),
            "eligible_figures": sorted(self.eligible_figures),
        }


def audit_plan_execution(
    plan: Any,
    code_history: Sequence[Any],
    round_number: int = 0,
    structured_families: Optional[Set[str]] = None,
    publication_figures: Optional[Sequence[Any]] = None,
    inherited_plan_ids_by_step: Optional[Mapping[int, Sequence[str]]] = None,
    target_lineage_id: str = "main",
) -> PlanAudit:
    """Audit an approved plan against this round's execution evidence.

    `structured_families` lets the caller vouch for families from structured
    state instead of code text (e.g. a registered score proves `gene_scoring`),
    which is the more reliable channel where it exists.

    `publication_figures` is the only evidence channel for visual deliverables.
    Successful plotting code is not enough: the artifact must be a current-round
    ``PublicationFigure`` accepted by the shared report-eligibility predicate.
    The argument is optional for source compatibility with older callers; an
    omitted collection is conservatively equivalent to no artifact evidence.
    """
    audit = PlanAudit()
    if plan is None or not getattr(plan, "steps", None):
        return audit

    executed = set(executed_method_families(code_history, round_number))
    executed |= set(structured_families or set())
    unavailable = unavailable_method_families(code_history)
    eligible_figures = eligible_current_round_figures(
        publication_figures,
        round_number,
        target_lineage_id=target_lineage_id,
    )
    planning_stage = str(getattr(plan, "stage", "") or "")
    plan_id = str(getattr(plan, "id", "") or "")
    plan_created_at = getattr(plan, "created_at", None)
    plan_revision_of = getattr(plan, "revision_of", None)
    plan_phase = str(getattr(plan, "phase", "") or "")
    audit.executed_families = sorted(executed)
    audit.unavailable_families = sorted(unavailable)
    audit.eligible_figures = [_artifact_label(figure) for figure in eligible_figures]

    for step in plan.steps:
        planned = extract_planned_methods(step)
        planned_figures, missing_figures = _audit_step_figure_requirements(
            step,
            eligible_figures,
            round_number,
            planning_stage,
            plan_id,
            plan_created_at,
            plan_revision_of,
            plan_phase,
            inherited_plan_ids_by_step,
            target_lineage_id,
        )
        step_id = int(getattr(step, "id", 0) or 0)
        title = str(getattr(step, "title", "") or "")[:300]

        if not planned and not planned_figures:
            audit.steps.append(StepAudit(
                step_id=step_id, title=title, status=STATUS_PENDING,
                reason="No machine-checkable method or figure identifier in the step text.",
                verifiable=False,
            ))
            continue

        missing = sorted(planned - executed)
        if not missing and not missing_figures:
            evidence: List[str] = []
            if planned:
                evidence.append("successful execution of: " + ", ".join(sorted(planned)))
            if planned_figures:
                evidence.append(
                    "shared figure contract for current-round report-eligible "
                    "PublicationFigure type(s): " + ", ".join(sorted(planned_figures))
                )
            audit.steps.append(StepAudit(
                step_id=step_id, title=title, status=STATUS_COMPLETED,
                reason="Evidenced by " + "; ".join(evidence),
                planned_families=sorted(planned),
                planned_figure_kinds=sorted(planned_figures),
            ))
            continue

        blocked = [f for f in missing if f in unavailable]
        if blocked and len(blocked) == len(missing):
            audit.steps.append(StepAudit(
                step_id=step_id, title=title, status=STATUS_SKIPPED,
                reason="Required tooling unavailable in this kernel: " + ", ".join(blocked),
                planned_families=sorted(planned), missing_families=missing,
                planned_figure_kinds=sorted(planned_figures),
                missing_figure_kinds=missing_figures,
            ))
            continue

        if _method_requirement_is_conditional(step, planned):
            absent = list(missing)
            absent.extend(f"figure:{kind}" for kind in missing_figures)
            audit.steps.append(StepAudit(
                step_id=step_id, title=title, status=STATUS_SKIPPED,
                reason="Step is conditional in the approved plan; not evidenced: "
                       + ", ".join(absent),
                planned_families=sorted(planned), missing_families=missing,
                planned_figure_kinds=sorted(planned_figures),
                missing_figure_kinds=missing_figures,
            ))
            continue

        absent = list(missing)
        absent.extend(f"figure:{kind}" for kind in missing_figures)
        audit.steps.append(StepAudit(
            step_id=step_id, title=title, status=STATUS_PENDING,
            reason="No current-round eligible evidence for: " + ", ".join(absent),
            planned_families=sorted(planned), missing_families=missing,
            planned_figure_kinds=sorted(planned_figures),
            missing_figure_kinds=missing_figures,
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
        figures = sorted({kind for s in unmet for kind in s.missing_figure_kinds})
        absent = families + [f"figure:{kind}" for kind in figures]
        return (True,
                f"{len(unmet)} approved step(s) show no eligible execution/artifact "
                f"evidence ({', '.join(absent[:6])}).")

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


def _clean_list(
    values: Any,
    limit: int,
    item_limit: int,
    *,
    diagnostic: bool = False,
) -> List[str]:
    """Drop blanks BEFORE applying the limit, so the cap counts real items.

    Slicing first would let a model's empty strings silently consume the task
    budget, which is the one number in this feature that has to mean what it says.
    """
    bounded_diagnostic = None
    if diagnostic:
        from backend.utils.error_parsing import bounded_diagnostic

    out: List[str] = []
    for value in (values or []):
        raw = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False)
        )
        text = (
            bounded_diagnostic(raw, item_limit)
            if diagnostic
            else raw
        ).strip()
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


def _sanitize_evidence_records(values: Any) -> List[Dict[str, str]]:
    """Allowlist bounded execution observations for PI/storage transport.

    Raw ``CodeExecution`` and ``AnalysisStep`` instances remain on the separate
    provenance merge path.  Only the four primitive observation fields cross the
    review boundary, and the terminal portions of output/error are retained so
    the useful result or exception context survives truncation.
    """
    from backend.utils.error_parsing import bounded_diagnostic

    if not isinstance(values, (list, tuple)):
        return []
    records: List[Dict[str, str]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        status = str(value.get("status", "") or "").strip().lower()
        if status not in {"completed", "failed"}:
            continue
        error = str(value.get("error", "") or "")
        records.append({
            "execution_id": str(value.get("execution_id", "") or "")[:100],
            "status": status,
            "stdout_excerpt": str(value.get("stdout_excerpt", "") or "")[-2000:],
            "error": bounded_diagnostic(error, 1000) if error else "",
        })
        if len(records) >= _MAX_LIST:
            break
    return records


def _sanitize_finding_records(values: Any) -> List[Dict[str, Any]]:
    """Allowlist supplementary finding provenance for merge and persistence."""
    if not isinstance(values, (list, tuple)):
        return []
    records: List[Dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        text = str(value.get("text", "") or "").strip()
        modality = str(value.get("modality", "") or "").strip()
        evidence_ids = [
            str(item).strip()[:100]
            for item in (value.get("evidence_execution_ids") or [])
            if str(item).strip()
        ]
        if (not text
                or modality not in {"single_cell", "bulk_rna_seq", "cross_modal"}
                or not evidence_ids):
            continue
        records.append({
            "text": text[:_MAX_TASK_LEN],
            "modality": modality,
            "evidence_execution_ids": evidence_ids[:_MAX_LIST],
        })
        if len(records) >= _MAX_FINDINGS:
            break
    return records


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
                "findings": [], "figures": [], "warnings": ["non-dict result"],
                "evidence_records": []}
    warnings = _clean_list(
        result.get("warnings"), _MAX_LIST, _MAX_GAP_LEN, diagnostic=True,
    )
    if result.get("figures"):
        warnings = _clean_list(
            [
                "Unreviewed supplementary figures were discarded; publication "
                "visuals must use the main figure/review pipeline."
            ] + warnings,
            _MAX_LIST,
            _MAX_GAP_LEN,
            diagnostic=True,
        )
    return {
        "status": str(result.get("status", "completed") or "completed"),
        "summary": str(result.get("summary", "") or "Supplementary analysis completed.")[:4000],
        "findings": _clean_list(result.get("findings"), _MAX_FINDINGS, _MAX_TASK_LEN),
        "finding_records": _sanitize_finding_records(result.get("finding_records")),
        # The automatic supplementary pass has no publication-figure review
        # callback. Never serialize its raw notebook images into report state.
        "figures": [],
        "warnings": warnings,
        "evidence_records": _sanitize_evidence_records(result.get("evidence_records")),
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
