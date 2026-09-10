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
    technical_linkage: Optional[str] = None
    awaiting_pi_decision: bool = False


@dataclass
class PlanAudit:
    """Aggregate technical plan/execution observations for PI review."""
    steps: List[StepAudit] = field(default_factory=list)
    executed_families: List[str] = field(default_factory=list)
    unavailable_families: List[str] = field(default_factory=list)
    eligible_figures: List[str] = field(default_factory=list)
    evidence_mode: str = "legacy_unverifiable"

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
        """Legacy steps for which no structured execution audit is available."""
        return [
            s for s in self.steps
            if not s.verifiable and not s.awaiting_pi_decision
        ]

    @property
    def awaiting_pi_decision(self) -> List[StepAudit]:
        """Current structured steps whose scientific status belongs to the PI."""
        return [s for s in self.steps if s.awaiting_pi_decision]

    @property
    def figure_unmet(self) -> List[StepAudit]:
        """Steps that named a required figure the round did not produce.

        Deliberately not filtered through ``unmet``. Whether an approved step's
        declared artifact exists and links to this round is a technical
        observation, while ``unmet`` carries a completion verdict that
        ``host_execution_linkage`` mode leaves to the PI (every step there is
        ``awaiting_pi_decision`` with ``verifiable=False``). Routing the
        artifact fact through the verdict made this list permanently empty for
        exactly the rounds that rely on it. Steps whose requirement the plan
        waived, or that are text-only, never populate ``missing_figure_kinds``
        upstream, so a step that legitimately produces no figure is not
        reported here. This observation informs PI review; it does not gate,
        retry, or reject anything.
        """
        return [s for s in self.steps if s.missing_figure_kinds]

    @property
    def family_unmet(self) -> List[StepAudit]:
        """Steps where the audit LOOKED for a named method and found none linked.

        ``figure_unmet``'s counterpart, and for the same reason. Whether an
        approved step's named method appears in this round's executions is a
        technical observation about linkage; ``unmet`` carries the completion
        verdict that ``host_execution_linkage`` mode leaves to the PI, so
        routing this fact through it left ``unmet_families`` permanently empty
        for exactly the rounds that run in that mode — while
        ``missing_families`` was populated on every step there. The PI read
        "no execution evidence 0" with no families line for a round whose named
        methods never ran, and the published quality record affirmed
        ``unmet_families: []``.

        Unlike the figure list this cannot simply read every step, because
        ``missing_families`` is NOT always an observation: the legacy and
        Supervisor-unverified branches set it to the whole planned list to mean
        "this cannot be checked", and reporting that as "nothing ran" asserts an
        absence the audit never established. Those are precisely the steps
        ``unverifiable`` collects, so they are excluded here. Informs review;
        gates nothing.
        """
        return [
            s for s in self.steps
            if s.missing_families and (s.verifiable or s.awaiting_pi_decision)
        ]

    def summary(self) -> Dict[str, Any]:
        figure_steps = [s for s in self.steps if s.planned_figure_kinds]
        figure_unmet = self.figure_unmet
        family_unmet = self.family_unmet
        return {
            "n_steps": len(self.steps),
            "n_completed": len(self.completed),
            "n_skipped": len(self.skipped),
            "n_unmet": len(self.unmet),
            "n_unverifiable": len(self.unverifiable),
            "n_awaiting_pi_decision": len(self.awaiting_pi_decision),
            "n_figure_steps": len(figure_steps),
            "n_figure_unmet": len(figure_unmet),
            "n_family_unmet": len(family_unmet),
            "unmet_titles": [s.title for s in self.unmet][:20],
            "unmet_families": sorted({
                f for s in family_unmet for f in s.missing_families
            }),
            "unmet_figure_kinds": sorted({
                kind for s in figure_unmet for kind in s.missing_figure_kinds
            }),
            "executed_families": sorted(self.executed_families),
            "unavailable_families": sorted(self.unavailable_families),
            "eligible_figures": sorted(self.eligible_figures),
            "evidence_mode": self.evidence_mode,
            "step_observations": [
                {
                    "plan_step_id": step.step_id,
                    "title": step.title,
                    "technical_linkage": (
                        step.technical_linkage
                        or (
                            "unverified"
                            if not step.verifiable
                            else {
                                STATUS_COMPLETED: "evidenced",
                                STATUS_PENDING: "not_evidenced",
                                STATUS_SKIPPED: "technically_unavailable",
                            }.get(step.status, "unverified")
                        )
                    ),
                    "verifiable": step.verifiable,
                    "awaiting_pi_decision": step.awaiting_pi_decision,
                    "reason": step.reason[:2000],
                }
                for step in self.steps
            ],
        }


def audit_plan_execution(
    plan: Any,
    code_history: Sequence[Any],
    round_number: int = 0,
    structured_families: Optional[Set[str]] = None,
    publication_figures: Optional[Sequence[Any]] = None,
    inherited_plan_ids_by_step: Optional[Mapping[int, Sequence[str]]] = None,
    target_lineage_id: str = "main",
    target_cohort_digest: Optional[str] = None,
    target_cohort_generation: Optional[int] = None,
    semantic_reconciliation: Optional[Mapping[str, Any]] = None,
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
    plan_receipt_records = [
        execution
        for execution in code_history
        if getattr(execution, "evidence_schema_version", None) == 2
        and int(getattr(execution, "round_number", 0) or 0) == int(round_number)
        and str(getattr(execution, "plan_id", "") or "") == plan_id
        and (getattr(execution, "lineage_id", None) or "main")
        == target_lineage_id
    ]
    receipt_records = [
        execution
        for execution in plan_receipt_records
        if (
            target_cohort_digest is None
            or str(getattr(execution, "cohort_digest", "") or "")
            == str(target_cohort_digest)
        )
        and (
            target_cohort_generation is None
            or int(getattr(execution, "cohort_generation", 0) or 0)
            == int(target_cohort_generation)
        )
    ]
    eligible_receipt_records = [
        execution
        for execution in receipt_records
        if execution_supports_evidence(execution)
        and getattr(execution, "execution_status", "succeeded") == "succeeded"
    ]
    reconciliation_current = bool(
        isinstance(semantic_reconciliation, Mapping)
        and str(semantic_reconciliation.get("plan_id") or "") == plan_id
        and int(semantic_reconciliation.get("round_number") or 0)
        == int(round_number)
        and str(semantic_reconciliation.get("lineage_id") or "main")
        == target_lineage_id
        and (
            target_cohort_digest is None
            or str(semantic_reconciliation.get("cohort_digest") or "")
            == str(target_cohort_digest)
        )
        and (
            target_cohort_generation is None
            or int(semantic_reconciliation.get("cohort_generation") or 0)
            == int(target_cohort_generation)
        )
    )
    semantic_links = {
        int(item.get("plan_step_id")): item
        for item in (
            semantic_reconciliation.get("links", [])
            if reconciliation_current else []
        )
        if isinstance(item, Mapping)
        and str(item.get("plan_step_id", "")).isdigit()
    }
    receipt_mode = bool(plan_receipt_records)
    if receipt_mode:
        audit.evidence_mode = (
            "supervisor_semantic"
            if reconciliation_current
            else "host_execution_linkage"
        )
    else:
        audit.evidence_mode = "legacy_unverifiable"
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

        if receipt_mode and not reconciliation_current:
            linked_execution_ids = [
                str(execution.id)
                for execution in eligible_receipt_records
                if step_id in (getattr(execution, "plan_step_ids", None) or [])
            ][:40]
            if linked_execution_ids:
                reason = (
                    "Host verified current successful execution receipt(s) linked "
                    "to this approved step: "
                    + ", ".join(linked_execution_ids[:8])
                    + ". PI review determines scientific completion."
                )
                if missing_figures:
                    reason += (
                        " Required reviewed artifact(s) remain missing: "
                        + ", ".join(missing_figures)
                        + "."
                    )
                technical_linkage = "execution_linked"
            else:
                reason = (
                    "No current successful execution receipt is linked to this "
                    "approved step. PI review determines whether it is completed, "
                    "pending, or skipped."
                )
                technical_linkage = "not_linked"
            audit.steps.append(StepAudit(
                step_id=step_id,
                title=title,
                status=STATUS_PENDING,
                reason=reason,
                planned_families=sorted(planned),
                missing_families=([] if linked_execution_ids else sorted(planned)),
                planned_figure_kinds=sorted(planned_figures),
                missing_figure_kinds=missing_figures,
                verifiable=False,
                technical_linkage=technical_linkage,
                awaiting_pi_decision=True,
            ))
            continue

        if receipt_mode:
            semantic = semantic_links.get(step_id)
            semantic_status = str(
                (semantic or {}).get("status") or "unverified"
            )
            linked_execution_ids = [
                str(value) for value in ((semantic or {}).get("execution_ids") or [])
            ]
            if semantic_status == "evidenced":
                receipt_reason = (
                    "Confirmed by Supervisor semantic reconciliation"
                    + (
                        "; execution receipt(s): "
                        + ", ".join(linked_execution_ids[:8])
                        if linked_execution_ids else ""
                    )
                )
                if not missing_figures:
                    audit.steps.append(StepAudit(
                        step_id=step_id,
                        title=title,
                        status=STATUS_COMPLETED,
                        reason=receipt_reason,
                        planned_families=sorted(planned),
                        planned_figure_kinds=sorted(planned_figures),
                    ))
                else:
                    audit.steps.append(StepAudit(
                        step_id=step_id,
                        title=title,
                        status=STATUS_PENDING,
                        reason=(
                            receipt_reason
                            + "; required reviewed artifact(s) remain missing: "
                            + ", ".join(missing_figures)
                        ),
                        planned_families=sorted(planned),
                        planned_figure_kinds=sorted(planned_figures),
                        missing_figure_kinds=missing_figures,
                    ))
            elif semantic_status == "partial":
                absent = []
                absent.append("remaining semantic work")
                absent.extend(f"figure:{kind}" for kind in missing_figures)
                audit.steps.append(StepAudit(
                    step_id=step_id,
                    title=title,
                    status=STATUS_PENDING,
                    reason=(
                        str((semantic or {}).get("reason") or "Partially evidenced")
                        + "; unresolved: " + ", ".join(absent)
                    ),
                    planned_families=sorted(planned),
                    missing_families=sorted(planned),
                    planned_figure_kinds=sorted(planned_figures),
                    missing_figure_kinds=missing_figures,
                ))
            else:
                audit.steps.append(StepAudit(
                    step_id=step_id,
                    title=title,
                    status=STATUS_PENDING,
                    reason=str(
                        (semantic or {}).get("reason")
                        or "Supervisor semantic reconciliation did not verify this step."
                    ),
                    planned_families=sorted(planned),
                    missing_families=sorted(planned),
                    planned_figure_kinds=sorted(planned_figures),
                    missing_figure_kinds=missing_figures,
                    verifiable=False,
                ))
            continue

        # Historical sessions have no Host receipt identity. Keep the keyword
        # families above for operator diagnostics only; they cannot grant plan
        # credit or trigger an automatic repair pass.
        audit.steps.append(StepAudit(
            step_id=step_id, title=title, status=STATUS_PENDING,
            reason=(
                "Legacy execution has no Host receipt identity and is therefore "
                "semantically unverifiable. Keyword-family matches are advisory only."
            ),
            planned_families=sorted(planned),
            missing_families=sorted(planned),
            planned_figure_kinds=sorted(planned_figures),
            missing_figure_kinds=missing_figures,
            verifiable=False,
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
    """Decide whether to show post-execution observations to the PI.

    The boolean requests a PI review; it never authorizes supplementary work.
    The PI alone decides whether any observed gap warrants another calculation.
    A missing runtime limits what can be executed but does not hide observations
    from the PI.
    """
    reasons: List[str] = []
    status = str(result_status or "").strip().lower()
    if status in ("error", "timeout", "max_steps_reached", "completed_with_fallback"):
        reasons.append(f"Execution ended as '{status}' rather than a clean completion.")

    if admissions:
        reasons.append(
            f"Agent recorded {len(admissions)} explicit not-run/failure admission(s)."
        )

    unmet = audit.unmet
    if unmet:
        families = sorted({f for s in unmet for f in s.missing_families})
        figures = sorted({kind for s in unmet for kind in s.missing_figure_kinds})
        absent = families + [f"figure:{kind}" for kind in figures]
        reasons.append(
            f"{len(unmet)} approved step(s) show no eligible execution/artifact "
            f"evidence ({', '.join(absent[:6])})."
        )
    else:
        # `unmet` is empty by construction in host_execution_linkage mode, so the
        # same absences have to be read off the steps directly — otherwise the
        # reason names nothing for exactly the rounds that run in that mode.
        linkage_steps = {s.step_id for s in audit.family_unmet} | {
            s.step_id for s in audit.figure_unmet
        }
        if linkage_steps:
            families = sorted({
                f for s in audit.family_unmet for f in s.missing_families
            })
            figures = sorted({
                kind for s in audit.figure_unmet for kind in s.missing_figure_kinds
            })
            absent = families + [f"figure:{kind}" for kind in figures]
            reasons.append(
                f"{len(linkage_steps)} approved step(s) have no linked execution "
                f"or artifact evidence ({', '.join(absent[:6])})."
            )

    if audit.unverifiable:
        reasons.append(
            f"Technical linkage observations leave {len(audit.unverifiable)} "
            "approved step(s) for PI completion assessment."
        )

    # A technically clean linkage record still does not establish scientific
    # sufficiency. The PI owns the plan-completion decision, so any audited plan
    # receives a review rather than a Host-written completion verdict.
    if audit.steps and not reasons:
        reasons.append(
            f"Technical linkage observations are available for {len(audit.steps)} "
            "approved step(s); PI review determines plan completion."
        )

    if not reasons:
        return (False, "No post-execution plan observations were available for PI review.")

    if is_last_direction_of_batch:
        reasons.append("This is the last direction of the active batch.")
    if not has_runtime:
        reasons.append(
            "No live analysis runtime is available; PI advice can be recorded but "
            "supplementary execution is unavailable."
        )

    return (True, " ".join(reasons))


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
    if limit <= 0:
        return []
    # Review fields are list-shaped contracts.  Treat scalar and nested mapping
    # values as malformed instead of iterating a string character-by-character
    # or serializing an arbitrary object into the review record.
    if not isinstance(values, (list, tuple)):
        return []
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


def sanitize_review_plan(
    plan: Any,
    max_tasks: int,
    valid_step_ids: Optional[Set[int]] = None,
    *,
    direction_pass_id: str = "",
    source_round: int = 0,
    direction: str = "",
    lineage_id: str = "main",
) -> Dict[str, Any]:
    """Bound the PI's review output. The task cap is the cost ceiling: an LLM PI
    asked what else could be done answers with 1-15 items every single time, so
    the number of tasks admitted is a hard constant here, not the PI's choice."""
    if not isinstance(plan, dict):
        return {"gaps": [], "requested_tasks": [], "supplementary_tasks": [], "out_of_scope": [],
                "step_decisions": [], "verdict": "no_action",
                "rationale": "Review returned no structured plan.", "followups": []}
    cap = max(0, int(max_tasks))
    requested_tasks = _clean_list(
        plan.get("supplementary_tasks"), _MAX_LIST, _MAX_TASK_LEN,
    )
    tasks = requested_tasks[:cap]
    verdict = str(plan.get("verdict", "") or "").strip().lower()
    if verdict not in ("supplement", "no_action"):
        # Supplementary execution requires an explicit PI authorization. Keep
        # any proposed tasks visible in the stored review, but never infer the
        # authorization from their presence.
        verdict = "no_action"
    allowed_ids = set(valid_step_ids or set())
    step_decisions: List[Dict[str, Any]] = []
    for value in plan.get("step_decisions") or []:
        if not isinstance(value, dict):
            continue
        try:
            step_id = int(value.get("plan_step_id"))
        except (TypeError, ValueError):
            continue
        if valid_step_ids is not None and step_id not in allowed_ids:
            continue
        status = str(value.get("status") or "").strip().lower()
        if status not in {STATUS_COMPLETED, STATUS_PENDING, STATUS_SKIPPED}:
            continue
        step_decisions.append({
            "plan_step_id": step_id,
            "status": status,
            "reason": str(value.get("reason") or "")[:_MAX_GAP_LEN],
        })
    # Follow-up IDs are assigned exactly once at this decision boundary from a
    # server-owned direction pass identity. Persisted records are later read as
    # canonical values and never re-sanitized or re-hashed.
    from backend.utils.direction_payload import sanitize_followup_items
    followups = sanitize_followup_items(
        plan.get("followups"),
        pass_identity=direction_pass_id,
        source_round=source_round,
        direction=direction,
        lineage_id=lineage_id,
    )
    return {
        "gaps": _clean_list(plan.get("gaps"), _MAX_LIST, _MAX_GAP_LEN),
        # Keep the PI's bounded intent visible when the technical budget trims
        # every executable task.  The executor consumes only supplementary_tasks.
        "requested_tasks": requested_tasks,
        "supplementary_tasks": tasks,
        "out_of_scope": _clean_list(plan.get("out_of_scope"), _MAX_LIST, _MAX_GAP_LEN),
        "step_decisions": step_decisions,
        "verdict": verdict,
        "rationale": str(plan.get("rationale", "") or "")[:2000],
        "followups": followups,
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
        if not text:
            continue
        if modality not in {"single_cell", "bulk_rna_seq", "cross_modal"}:
            modality = ""

        def _ids(field_name: str) -> List[str]:
            return [
                str(item).strip()[:100]
                for item in (value.get(field_name) or [])
                if str(item).strip()
            ][:_MAX_LIST]

        def _step_ids(field_name: str) -> List[int]:
            normalized = []
            for item in value.get(field_name) or []:
                try:
                    step_id = int(item)
                except (TypeError, ValueError):
                    continue
                if step_id > 0 and step_id not in normalized:
                    normalized.append(step_id)
                if len(normalized) >= _MAX_LIST:
                    break
            return normalized

        def _nonnegative_int(field_name: str) -> int:
            try:
                return max(0, int(value.get(field_name) or 0))
            except (TypeError, ValueError):
                return 0

        evidence_status = str(
            value.get("evidence_status") or "unverified"
        ).lower()
        if evidence_status not in {"verified", "unverified"}:
            evidence_status = "unverified"
        semantic_status = str(
            value.get("semantic_status") or "unverified"
        ).lower()
        if semantic_status not in {"evidenced", "partial", "unverified"}:
            semantic_status = "unverified"
        record = {
            "id": str(value.get("id") or "")[:100],
            "text": text[:_MAX_TASK_LEN],
            "modality": modality or None,
            "evidence_execution_ids": _ids("evidence_execution_ids"),
            "candidate_evidence_execution_ids": _ids(
                "candidate_evidence_execution_ids"
            ),
            "claimed_evidence_execution_ids": _ids(
                "claimed_evidence_execution_ids"
            ),
            "candidate_plan_step_ids": _step_ids(
                "candidate_plan_step_ids"
            ),
            "confirmed_plan_step_ids": _step_ids(
                "confirmed_plan_step_ids"
            ),
            "evidence_status": evidence_status,
            "semantic_status": semantic_status,
            "round_number": _nonnegative_int("round_number"),
            "lineage_id": str(value.get("lineage_id") or "main")[:100],
            "plan_id": str(value.get("plan_id") or "")[:100] or None,
            "analysis_generation": _nonnegative_int("analysis_generation"),
            "cohort_generation": _nonnegative_int("cohort_generation"),
            "cohort_digest": (
                str(value.get("cohort_digest") or "")[:128] or None
            ),
        }
        records.append(record)
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
        "followups": [],
        "review_result": {},
        "plan_audit": {},
        "supplementary_findings": [],
    }
