"""Invariant helpers for score and mechanism registration."""
from __future__ import annotations

from .executor_dispatch import ToolAction


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
    gate_note = ""
    if prov_error:
        # Reject rather than record an unattributable panel: an
        # unsourced gene list is indistinguishable downstream from
        # a versioned one, which is how two runs of the same
        # analysis ended up with different panels and no record.
        if context.score_gate_reject_count < context.max_score_gate_rejections:
            context.score_gate_reject_count += 1
            logger.warning(
                f"[Bioinformatics] register_score rejected: provenance "
                f"({context.score_gate_reject_count}/{context.max_score_gate_rejections})"
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": (
                    f"register_score rejected: {prov_error} "
                    f"Re-call register_score with a valid gene_list_source. "
                    + _method_rule_text("cite-the-resource-version")
                ),
            })
            return ToolAction.CONTINUE
        # Ceiling reached: record the score so the methodology
        # section is not simply empty, but mark what is missing.
        logger.warning(
            "[Bioinformatics] register_score provenance gate: max "
            "rejections reached, recording score as unattributed"
        )
        provenance, _ = parse_provenance(
            "bespoke:provenance not supplied after repeated requests; "
            "this panel is unattributed and not reproducible"
        )
        gate_note = (
            "Recorded WITHOUT valid provenance after "
            f"{context.max_score_gate_rejections} attempts; treat this panel "
            "as unattributed."
        )
        context.record_forced_score_gate(_evidence_gate_record(
            gate="register-score-provenance",
            rule_id="cite-the-resource-version",
            detail=str(prov_error),
            subject=str(args.get("score_name", "") or ""),
            severity="block",
            rejection_count=context.score_gate_reject_count,
            rejection_limit=context.max_score_gate_rejections,
            forced_release=True,
        ))

    # A panel score must cite the plan that froze the panel.
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
    # the power gate then fires -- or fails to fire -- on a number
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
                "this score uses a gene panel, so it must cite the "
                "plan_id returned by register_analysis_plan. Call "
                "register_analysis_plan FIRST with this panel, then "
                "re-call register_score with the plan_id it returns."
                if not plan_id else
                f"no frozen plan with plan_id={plan_id!r}. "
                f"Recent plan_ids: {known or 'none'}."
            )
            if context.score_gate_reject_count < context.max_score_gate_rejections:
                context.score_gate_reject_count += 1
                logger.warning(
                    f"[Bioinformatics] register_score rejected: plan "
                    f"({context.score_gate_reject_count}/{context.max_score_gate_rejections})"
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"register_score rejected: {detail}",
                })
                return ToolAction.CONTINUE
            logger.warning(
                "[Bioinformatics] register_score plan gate: max rejections "
                "reached, recording score without a frozen plan"
            )
            gate_note = (gate_note + " " if gate_note else "") + (
                f"Recorded WITHOUT a frozen plan after "
                f"{context.max_score_gate_rejections} attempts; the panel was not "
                f"pre-specified, so it cannot be distinguished from one "
                f"chosen after seeing the result."
            )
            context.record_forced_score_gate(_evidence_gate_record(
                gate="register-score-plan",
                rule_id="prespecify-the-family-before-results",
                detail=detail,
                subject=str(args.get("score_name", "") or ""),
                severity="block",
                rejection_count=context.score_gate_reject_count,
                rejection_limit=context.max_score_gate_rejections,
                forced_release=True,
            ))
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
    if gate_note:
        # The escape must reach the report, not just the log.
        score_def.plan_drift = dict(score_def.plan_drift or {})
        score_def.plan_drift["gate_escape"] = gate_note.strip()
    drift_text = describe_drift(drift) if drift else ""
    if drift_text:
        result_lines.append(
            f"NOTE: the scored panel departs from plan {plan_id}: "
            f"{drift_text}. This is recorded on the score and will "
            f"appear in the report. If the change was deliberate, say "
            f"why in the findings; if not, re-score the frozen panel. "
            + _method_rule_text("prespecify-the-family-before-results")
        )
    if gate_note:
        result_lines.append(gate_note.strip())
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
