"""Invariant helpers for score and mechanism registration."""
from __future__ import annotations

from .executor_dispatch import ToolAction
from .shared import current_execution_identity_error


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


def eligible_execution_ids(code_history, *, round_number: int, successful_executions):
    return {
        execution.id
        for execution in successful_executions(
            code_history, round_number=round_number
        )
    }


def mechanism_execution_id_error(programs, eligible_ids: set[str]) -> str | None:
    invalid_ids = sorted(
        {
            evidence_id
            for program in programs
            for evidence_id in program.evidence_execution_ids
            if evidence_id not in eligible_ids
        }
    )
    if not invalid_ids:
        return None
    return (
        "Mechanism programs cite failed, contradictory, or unknown execution IDs: "
        f"{invalid_ids}. Available IDs: {sorted(eligible_ids)[-20:]}"
    )


async def handle_register_method_decision(context, tool_call) -> ToolAction:
    """Persist an agent-selected bulk method without prescribing one."""
    from ...models.schemas import MethodDecision

    state = context.state
    args = tool_call.arguments
    profile_id = str(args.get("input_profile_id") or "").strip()
    profile = next((
        item for item in state.bulk_input_profiles if item.id == profile_id
    ), None)
    execution_id = str(args.get("execution_id") or "").strip()
    execution = next((
        item for item in context.agent.code_history if item.id == execution_id
    ), None)
    error = current_execution_identity_error(state, execution)
    if profile is None:
        error = "input_profile_id does not identify a current host profile"
    elif error is None and profile.id not in (execution.input_profile_ids or []):
        error = "execution_id is not bound to the selected host input profile"
    rationale = str(args.get("rationale") or "").strip()
    method = str(args.get("method") or "").strip()
    claim_level = str(args.get("claim_level") or "").strip()
    if not method:
        error = error or "method is required"
    elif claim_level not in {"inferential", "descriptive", "not_feasible"}:
        error = error or "claim_level is outside the closed vocabulary"
    elif len(rationale) < 20:
        error = error or "rationale must explain the scientific choice"
    if error:
        context.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"register_method_decision rejected: {error}.",
        })
        return ToolAction.CONTINUE

    decision = MethodDecision(
        input_profile_id=profile.id,
        claim_level=claim_level,
        method=method,
        rationale=rationale[:4000],
        execution_id=execution.id,
        round_number=state.current_round,
    )
    state.method_decisions.append(decision)
    context.messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": (
            f"Method decision recorded: {decision.id}. Claim level: "
            f"{decision.claim_level}; method: {decision.method}."
        ),
    })
    context.analysis_log.append(AnalysisStep(
        step_number=context.step_count,
        step_type="register_method_decision",
        agent_reasoning=context.agent_reasoning,
        code_description=f"Bulk method decision {decision.id}: {decision.method}",
    ))
    return ToolAction.CONTINUE


async def handle_register_publication_table(context, tool_call) -> ToolAction:
    """Register a CSV through host-owned path, digest, and lifecycle checks."""
    import csv
    import hashlib
    from pathlib import Path

    from ...models.schemas import PublicationTable, FigureReviewStatus
    from ...services.cohort_identity import active_cohort_identity

    state = context.state
    args = tool_call.arguments
    workspace = Path(context.agent.sandbox.working_dir).resolve()
    raw_path = str(args.get("csv_path") or "").strip()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(workspace)
    except (OSError, ValueError):
        context.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": "register_publication_table rejected: csv_path must be an existing workspace file.",
        })
        return ToolAction.CONTINUE
    if candidate.is_symlink() or not resolved.is_file() or resolved.suffix.lower() != ".csv":
        context.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": "register_publication_table rejected: only regular, non-symlink CSV files are supported.",
        })
        return ToolAction.CONTINUE

    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = [str(item)[:300] for item in (reader.fieldnames or [])][:200]
            preview = []
            total_rows = 0
            for row in reader:
                total_rows += 1
                if len(preview) < 10:
                    preview.append({
                        str(key)[:300]: str(value)[:2000]
                        for key, value in (row or {}).items()
                    })
    except (OSError, csv.Error, UnicodeError) as exc:
        context.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"register_publication_table rejected: CSV could not be read ({exc}).",
        })
        return ToolAction.CONTINUE

    active_lineage = getattr(state, "restore_branch_id", None) or "main"
    cohort = active_cohort_identity(state)
    source_id = str(args.get("evidence_execution_id") or "").strip()
    source = next((
        item for item in context.agent.code_history
        if str(getattr(item, "id", "") or "") == source_id
    ), None)
    source_error = current_execution_identity_error(state, source)
    source_current = source_error is None
    approved_steps = {
        int(step.id) for step in (getattr(state.current_plan, "steps", None) or [])
    }
    candidate_steps = sorted({
        int(value) for value in (args.get("plan_step_ids") or [])
        if isinstance(value, int)
        and not isinstance(value, bool)
        and value in approved_steps
    })
    if not source_current:
        candidate_steps = []
    table = PublicationTable(
        name=str(args.get("name") or relative.stem)[:300],
        description=str(args.get("description") or "")[:4000],
        csv_path=str(resolved),
        csv_sha256=digest.hexdigest(),
        columns=columns,
        preview_data=preview,
        total_rows=total_rows,
        interpretation=str(args.get("interpretation") or "")[:4000] or None,
        review_status=FigureReviewStatus.APPROVED,
        lineage_id=active_lineage,
        plan_id=str(getattr(state.current_plan, "id", "") or "") or None,
        plan_step_ids=candidate_steps,
        plan_binding_status="candidate" if candidate_steps else "unverified",
        plan_binding_reason=(
            "Candidate producer link awaiting Supervisor semantic reconciliation."
            if candidate_steps else
            (
                f"No current eligible producer receipt was available: {source_error}."
                if source_error else
                "No candidate current-plan step was supplied."
            )
        ),
        round_number=int(state.current_round or 0),
        analysis_generation=int(
            (state.sandbox_state or {}).get("analysis_generation", 0) or 0
        ),
        cohort_generation=int(cohort.get("cohort_generation", 0) or 0),
        cohort_digest=str(cohort.get("cohort_digest") or "") or None,
        source_execution_id=source.id if source_current else None,
    )
    state.publication_tables.append(table)
    context.messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": (
            f"Publication table registered: {table.id} ({relative.as_posix()}, "
            f"{table.total_rows} rows). Plan binding: {table.plan_binding_status}."
        ),
    })
    context.analysis_log.append(AnalysisStep(
        step_number=context.step_count,
        step_type="register_publication_table",
        agent_reasoning=context.agent_reasoning,
        code_description=f"Publication table {table.id}: {table.name}",
    ))
    return ToolAction.CONTINUE


async def handle_register_analysis_plan(context, tool_call) -> ToolAction:
    state = context.state
    messages = context.messages
    analysis_log = context.analysis_log
    step_count = context.step_count
    agent_reasoning = context.agent_reasoning
    args = tool_call.arguments
    from ...models.schemas import ScoringPlan
    from ...services.analysis_plan import (
        canonical_panel, panel_hash, summarize_edits,
        validate_edit_operations,
    )
    from ...services.gene_set_provenance import parse_provenance

    panel = canonical_panel(args.get("panel_genes"))
    if not panel:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": (
                "register_analysis_plan rejected: panel_genes is empty. "
                "Freeze the exact panel you intend to score."
            ),
        })
        return ToolAction.CONTINUE

    _prov, prov_error = parse_provenance(args.get("panel_source"))
    if prov_error:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"register_analysis_plan rejected: {prov_error}",
        })
        return ToolAction.CONTINUE

    edits, edit_error = validate_edit_operations(
        args.get("edit_operations")
    )
    if edit_error:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"register_analysis_plan rejected: {edit_error}",
        })
        return ToolAction.CONTINUE

    plan = ScoringPlan(
        score_name=args.get("score_name", ""),
        method=args.get("method", ""),
        hypothesis=args.get("hypothesis", ""),
        falsifier=args.get("falsifier", ""),
        panel_genes=panel,
        panel_hash=panel_hash(panel),
        panel_source=args.get("panel_source", ""),
        baseline_genes=canonical_panel(args.get("baseline_genes")),
        baseline_source=args.get("baseline_source", ""),
        edit_operations=edits,
        testing_family=[str(t) for t in (args.get("testing_family") or [])],
        registered_at_step=step_count,
        round_number=state.current_round,
    )
    state.scoring_plans.append(plan)

    plan_lines = [
        f"Plan frozen. plan_id={plan.id}. "
        f"Panel: {len(panel)} gene(s), hash {plan.panel_hash}. "
        f"Cite this plan_id when you call register_score."
    ]
    edit_summary = summarize_edits(edits)
    if edit_summary:
        plan_lines.append(edit_summary + ".")
    if not plan.falsifier:
        plan_lines.append(
            "No falsifier was stated. A score with no result that "
            "would count against it cannot support a conclusion "
            "either way."
        )
    if not plan.testing_family:
        plan_lines.append(
            "No testing_family was declared. If this panel goes into a "
            "shared enrichment call, its q-value depends on the other "
            "sets in that call and cannot be interpreted alone."
        )
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": " ".join(plan_lines),
    })

    analysis_log.append(AnalysisStep(
        step_number=step_count, step_type="register_analysis_plan",
        agent_reasoning=agent_reasoning,
        code_description=(
            f"Plan {plan.id} for {plan.score_name}: "
            f"{len(panel)} genes via {plan.method}"
        ),
    ))

    return ToolAction.CONTINUE


async def handle_register_score(context, tool_call) -> ToolAction:
    self = context.agent
    state = context.state
    messages = context.messages
    analysis_log = context.analysis_log
    step_count = context.step_count
    agent_reasoning = context.agent_reasoning
    args = tool_call.arguments
    from ...models.schemas import ScoreDefinition
    from ...services.analysis_plan import (
        describe_drift, diff_panels, panel_hash,
    )
    from ...services.gene_set_provenance import (
        assess_panel_power, parse_provenance, power_caveat,
    )

    gene_list = args.get("gene_list") or []
    provenance, prov_error = parse_provenance(
        args.get("gene_list_source")
    )
    review_notes = []

    def record_review_observation(*, gate: str, rule_id: str, detail: str) -> None:
        record = _evidence_gate_record(
            gate=gate,
            rule_id=rule_id,
            detail=detail,
            subject=str(args.get("score_name", "") or ""),
            severity="advise",
        )
        record["round_number"] = state.current_round
        key = (record["gate"], record["rule_id"], record["subject"], record["round_number"])
        if not any(
            (
                existing.get("gate"),
                existing.get("rule_id"),
                existing.get("subject"),
                existing.get("round_number"),
            ) == key
            for existing in context.evidence_gate_findings
            if isinstance(existing, dict)
        ):
            context.evidence_gate_findings.append(record)

    if prov_error:
        logger.warning(
            "[Bioinformatics] register_score provenance unresolved; "
            "recording a PI review observation"
        )
        provenance, _ = parse_provenance(
            "bespoke:valid provenance was not supplied; "
            "this panel is unattributed and not reproducible"
        )
        review_notes.append(
            "Recorded without valid provenance; the PI must decide how much "
            "weight to give this unattributed panel."
        )
        record_review_observation(
            gate="register-score-provenance",
            rule_id="cite-the-resource-version",
            detail=str(prov_error),
        )

    # Link to a frozen panel when one exists; missing linkage remains visible.
    plan_id = str(args.get("plan_id", "") or "").strip()
    plan = next(
        (pl for pl in state.scoring_plans if pl.id == plan_id),
        None,
    )
    drift = {}
    # A set's size is a property of the set, not of the excerpt
    # reported alongside it. `gene_list` is documented as an
    # excerpt, so sizing and hashing the panel from it mis-states
    # every panel above that cap: n_genes, match_rate, the
    # underpowered flag and the drift hash all come out wrong, and
    # the power observation then fires -- or fails to fire -- on a number
    # that describes the excerpt. The frozen plan carries the full
    # canonicalised membership; use it when one is cited, and keep
    # the excerpt for what it is for, which is detecting drift
    # away from that record.
    panel_members = list(gene_list)
    if gene_list:
        if not plan_id or plan is None:
            known = ", ".join(
                pl.id for pl in state.scoring_plans[-5:]
            )
            detail = (
                "this score uses a gene panel, but no frozen plan_id was "
                "supplied. The record cannot establish that the panel was "
                "pre-specified before its result was visible."
                if not plan_id else
                f"no frozen plan with plan_id={plan_id!r}. "
                f"Recent plan_ids: {known or 'none'}."
            )
            logger.warning(
                "[Bioinformatics] register_score has no frozen plan; "
                "recording a PI review observation"
            )
            review_notes.append(
                "Recorded without a frozen plan; the panel was not pre-specified, "
                "so the PI must decide how this affects interpretation."
            )
            record_review_observation(
                gate="register-score-plan",
                rule_id="prespecify-the-family-before-results",
                detail=detail,
            )
            plan_id = ""
        else:
            drift = diff_panels(plan.panel_genes, gene_list)
            if plan.panel_genes:
                panel_members = list(plan.panel_genes)

    n_matched = args.get("n_genes_matched")
    try:
        n_matched = int(n_matched) if n_matched is not None else None
    except (TypeError, ValueError):
        n_matched = None
    power = assess_panel_power(len(panel_members), n_matched)

    context.has_register_score = True
    score_def = ScoreDefinition(
        score_name=args.get("score_name", ""),
        method=args.get("method", ""),
        description=args.get("description", ""),
        parameters=args.get("parameters", {}),
        gene_list=gene_list,
        gene_list_source=args.get("gene_list_source", ""),
        background_source=str(args.get("background_source", "") or ""),
        provenance=provenance.to_dict(),
        n_genes=len(panel_members),
        n_genes_matched=n_matched,
        power=power,
        plan_id=plan_id,
        panel_hash=panel_hash(panel_members),
        plan_drift=drift,
        round_number=state.current_round,
    )
    # Deduplicate: update existing score with same name, or append new
    existing_idx = next(
        (i for i, sd in enumerate(state.score_definitions) if sd.score_name == score_def.score_name),
        None
    )
    if existing_idx is not None:
        state.score_definitions[existing_idx] = score_def
        await self._notify("status", f"Updated score: {score_def.score_name} ({score_def.method})")
    else:
        state.score_definitions.append(score_def)
        await self._notify("status", f"Registered score: {score_def.score_name} ({score_def.method})")

    action_word = "updated" if existing_idx is not None else "registered"
    result_lines = [
        f"Score '{score_def.score_name}' {action_word}. "
        f"Method: {score_def.method}. "
        f"Provenance: {provenance.citation()}. "
        f"This will appear in the report methodology."
    ]
    if provenance.is_bespoke:
        result_lines.append(
            "This panel is recorded as hand-assembled. If a versioned "
            "database set covers the same concept, prefer it: bespoke "
            "panels are reported as such and carry less weight."
        )
    if n_matched is None and gene_list:
        result_lines.append(
            "n_genes_matched was not supplied, so power was assessed on "
            "the nominal panel size. Report the number of panel genes "
            "actually present in the matrix for an accurate assessment."
        )
    if review_notes:
        # The observations must reach the report, not just the log.
        score_def.plan_drift = dict(score_def.plan_drift or {})
        score_def.plan_drift["pi_review_observations"] = list(review_notes)
    drift_text = describe_drift(drift) if drift else ""
    if drift_text:
        result_lines.append(
            f"NOTE: the scored panel departs from plan {plan_id}: "
            f"{drift_text}. This is recorded on the score and will "
            f"appear in the report. If the change was deliberate, say "
            f"why in the findings; if not, re-score the frozen panel. "
            + _method_rule_text("prespecify-the-family-before-results")
        )
    result_lines.extend(review_notes)
    caveat = power_caveat(power)
    if caveat:
        result_lines.append(caveat)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": " ".join(result_lines),
    })

    analysis_log.append(AnalysisStep(
        step_number=step_count, step_type="register_score",
        agent_reasoning=agent_reasoning,
        code_description=f"Score: {score_def.score_name} via {score_def.method}",
    ))

    return ToolAction.CONTINUE


async def handle_register_mechanisms(context, tool_call) -> ToolAction:
    self = context.agent
    state = context.state
    messages = context.messages
    analysis_log = context.analysis_log
    step_count = context.step_count
    agent_reasoning = context.agent_reasoning
    args = tool_call.arguments
    progs, err = validate_mechanism_program_payload(
        args,
        round_number=state.current_round,
        phase=str(getattr(state.analysis_phase, "value", state.analysis_phase)),
        stage=context.planning_stage,
        # The provenance of every registered score, so a claim
        # cannot be graded above what its instruments support.
        score_provenance={
            sd.score_name: (sd.provenance or {})
            for sd in state.score_definitions
            if getattr(sd, "score_name", "")
        },
    )
    if err:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"[REJECTED] {err} Please correct and call register_mechanism_programs again.",
        })
        analysis_log.append(AnalysisStep(
            step_number=step_count, step_type="register_mechanism_programs",
            agent_reasoning=agent_reasoning,
            findings=[], findings_summary=f"REJECTED: {err}",
        ))
        return ToolAction.CONTINUE
    eligible_ids = eligible_execution_ids(
        self.code_history,
        round_number=state.current_round,
        successful_executions=successful_evidence_executions,
    )
    err = mechanism_execution_id_error(progs, eligible_ids)
    if err:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"[REJECTED] {err}",
        })
        analysis_log.append(AnalysisStep(
            step_number=step_count,
            step_type="register_mechanism_programs",
            agent_reasoning=agent_reasoning,
            findings=[],
            findings_summary=f"REJECTED: {err}",
        ))
        return ToolAction.CONTINUE
    context.mechanism_programs_registered = True
    context.mechanism_programs.extend(progs)
    # Handler is the single writer to AnalysisState (see plan decision 1).
    state.data_driven_mechanism_programs.extend(progs)
    if progs:
        await self._notify("status", f"Registered {len(progs)} data-driven mechanism program(s)")
        content = (
            f"Registered {len(progs)} data-driven mechanism program(s). They will be "
            "reviewed by the PI, mapped against literature, and may appear as CANDIDATE "
            "mechanisms (not confirmed) in the report. You may now call report_findings."
        )
    else:
        await self._notify("status", "No data-driven mechanism program registered (reason recorded)")
        content = "Recorded that no data-driven mechanism program was registered. You may now call report_findings."
    messages.append({
        "role": "tool", "tool_call_id": tool_call.id, "content": content,
    })
    analysis_log.append(AnalysisStep(
        step_number=step_count, step_type="register_mechanism_programs",
        agent_reasoning=agent_reasoning,
        code_description=f"{len(progs)} mechanism program(s)",
    ))

    return ToolAction.CONTINUE
