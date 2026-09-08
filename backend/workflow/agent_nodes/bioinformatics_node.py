"""Bioinformatics workflow-node phases behind AgentNodesMixin."""
from __future__ import annotations


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )

async def start_bioinformatics_analysis(self, state):
    self._current_graph_state = state
    analysis_state = state["analysis_state"]

    skip = await self._require_core_agent(state, "bioinformatics", "bioinformatics_analysis")
    if skip is not None:
        return skip

    cached = cache_hit(analysis_state, AgentType.BIOINFORMATICS)
    if cached is not None:
        logger.info(f"[{self.session_id}] bioinformatics cache hit, skipping re-execution")
        await self._send_message("result", cached, AgentType.BIOINFORMATICS)
        # If the cached result was from stage2, the normal path would have
        # transitioned planning_stage to "full". Carry that forward here so
        # downstream routing (_after_bioinformatics) sees the correct stage.
        result_state = {
            **state,
            "analysis_state": analysis_state,
            "bioinformatics_result": cached,
            "failed_node": None,
        }
        planning_stage = state.get("planning_stage", "full")
        if planning_stage == "stage2":
            result_state["planning_stage"] = "full"
            result_state["planning_context"] = None
        return result_state

    logger.info(f"[{self.session_id}] Starting bioinformatics analysis")
    seq_files = analysis_state.get_files_by_section("sequencing")
    logger.info(f"[{self.session_id}] Sequencing files: {len(seq_files)}")

    await self._send_message("status", "Starting bioinformatics analysis...", AgentType.BIOINFORMATICS)

    # Check if data is available
    if not analysis_state.has_sequencing_data():
        waiting_result = {"findings": [], "status": "waiting_data"}
        await self._send_message(
            "result",
            {"message": "Please upload sequencing data first", "findings": []},
            AgentType.BIOINFORMATICS
        )
        cache_store(analysis_state, AgentType.BIOINFORMATICS, waiting_result)
        return {
            **state,
            "analysis_state": analysis_state,
            "bioinformatics_result": waiting_result
        }

    return {"_continue": analysis_state}

async def prepare_bioinformatics_instruction(self, state, analysis_state):
    # Planning stage support: filter tasks and adjust instructions based on stage
    planning_stage = state.get("planning_stage", "full")

    # Detect resume: a non-completed checkpoint for this round+stage means a
    # prior run already started the autonomous loop (e.g. it failed and the
    # user clicked Retry on the LLM-failure gate). On resume the agent restores
    # the conversation from the checkpoint and ignores the instruction we build
    # here, so the PI planning prelude — goal definition, task refinement, and
    # the heavy pre-execution snapshot — is pure waste and re-triggers PI LLM
    # calls. Skip it and go straight back into the autonomous loop. The
    # planning_stage argument prevents a stale stage1 checkpoint (same round)
    # from being treated as resumable once stage2 begins.
    resuming = await self.bioinformatics_agent.has_resumable_checkpoint(
        analysis_state, planning_stage
    )
    if resuming:
        logger.info(
            f"[{self.session_id}] Resumable checkpoint detected — "
            f"skipping PI planning prelude and resuming autonomous analysis"
        )

    # Incorporate human guidance from workflow state (set by _initialize_round)
    human_guidance = ""
    if state.get("human_guidance"):
        human_guidance = f"\n\nHuman researcher guidance: {state['human_guidance']}"

    # Step 1: PI provides analysis goals
    if not resuming:
        await self._send_message("status", "PI is defining analysis goals...", AgentType.PI)

    # Get PI instructions from workflow state
    pi_instructions = state.get("pi_instructions", {})
    bioinf_tasks = pi_instructions.get("bioinformatics_tasks", [])
    merge_strategy = pi_instructions.get("merge_strategy", "")
    cluster_comparison_strategy = pi_instructions.get("cluster_comparison_strategy", "")

    # Stage1 task filtering: only QC + clustering + annotation tasks
    if planning_stage == "stage1":
        stage1_tasks = [t for t in bioinf_tasks if _has_stage1_keyword(_get_task_text(t))]
        if not stage1_tasks:
            stage1_tasks = [
                "Execute scRNA-seq QC, clustering, and cell type annotation"
            ]
        bioinf_tasks = stage1_tasks
    elif planning_stage == "stage2":
        # The user-reviewed stage2 plan lives in current_plan; pi_instructions
        # bioinformatics_tasks were cleared by _reset_pi_instructions_for_stage2.
        # Recover the executable tasks from the approved plan so the agent runs
        # what the user reviewed instead of falling back to default goals.
        plan = analysis_state.current_plan
        if plan and getattr(plan, "steps", None):
            bioinf_tasks = [
                (f"{s.title}: {s.method}" if s.method else s.title)
                for s in plan.steps if getattr(s, "title", "")
            ]

    # Dynamic PI planning: refine bioinformatics tasks based on completed results
    # (skipped on resume — the refined instruction would be discarded anyway)
    overview = pi_instructions.get("overview", "")
    if overview and not resuming:
        completed = self._collect_current_round_results(state)
        if completed:
            await self._send_message("status", "PI is refining bioinformatics tasks...", AgentType.PI)
            bioinf_tasks = await self.pi_agent.plan_agent_tasks(
                agent_name="bioinformatics",
                original_tasks=bioinf_tasks,
                completed_results=completed,
                overall_strategy=overview,
                research_objective=analysis_state.initial_instruction,
                state=analysis_state,
            )

    # Build previous findings and code history context for round 2+
    previous_bio_findings = self._build_previous_findings_context(analysis_state, "bioinformatics")
    code_history_context = ""
    if analysis_state.current_round > 1:
        code_history_context = self._build_code_history_summary(analysis_state)

    # Include current round pathology findings if available
    pathology_context = ""
    pathology_result = state.get("pathology_result")
    if pathology_result and pathology_result.get("findings"):
        path_findings = pathology_result["findings"][:5]
        pathology_context = (
            "\n\nCurrent round pathology findings:\n"
            + "\n".join(f"- {f}" for f in path_findings)
        )

    # Include current round biochemistry findings if available
    biochemistry_context = ""
    biochemistry_result = state.get("biochemistry_result")
    if biochemistry_result and biochemistry_result.get("findings"):
        biochem_findings = biochemistry_result["findings"][:5]
        biochemistry_context = (
            "\n\nCurrent round biochemistry findings:\n"
            + "\n".join(f"- {f}" for f in biochem_findings)
        )

    # Build comprehensive instruction for bioinformatics agent
    # Phase-aware instruction for bioinformatics agent
    # Detect data types for dynamic instruction (single pass)
    has_scrna = analysis_state.has_scrnaseq()
    has_bulk = analysis_state.has_bulk_rnaseq()
    if has_bulk:
        from backend.services.bulk_input_profile import ensure_bulk_input_profiles

        ensure_bulk_input_profiles(analysis_state)
    phase_instruction = build_phase_instruction(
        planning_stage=planning_stage,
        analysis_phase=analysis_state.analysis_phase.value,
        has_bulk=has_bulk, has_scrna=has_scrna,
        direction=analysis_state.current_round_direction or "",
    )
    design_facts = sequencing_design_instruction(analysis_state)

    # Build merge strategy context from PI instructions
    merge_context = ""
    if merge_strategy:
        merge_context += f"\n\nPI's cluster merge strategy:\n{merge_strategy}"
    if cluster_comparison_strategy:
        merge_context += f"\n\nPI's cluster comparison strategy:\n{cluster_comparison_strategy}"

    if has_scrna and has_bulk:
        data_desc = "single-cell RNA-seq and bulk RNA-seq"
    elif has_bulk:
        data_desc = "bulk RNA-seq"
    else:
        data_desc = "single-cell RNA-seq"

    if bioinf_tasks:
        goals_text = "\n".join(f"- {task}" for task in _normalize_tasks(bioinf_tasks))
    elif has_bulk and not has_scrna:
        goals_text = (
            "- Execute the PI-approved bulk objectives. Treat the Host input profile "
            "as facts; PI owns the question and claim level, while Bioinformatics "
            "selects and records the method and implementation details"
        )
    elif has_bulk and has_scrna:
        goals_text = (
            "- Execute the PI-approved single-cell and bulk objectives. PI decides "
            "which cross-modal questions and claim levels are justified; "
            "Bioinformatics selects and records compatible methods and details"
        )
    else:
        goals_text = "- Perform standard scRNA-seq analysis pipeline"

    instruction = f"""Analyze {data_desc} data for project: {analysis_state.name}

Research context: {analysis_state.initial_instruction}
{phase_instruction}

Measured sequencing-design facts for PI-owned scientific judgment:
{design_facts or "No additional measured sequencing-design facts."}

PI's analysis goals for this round:
{goals_text}
{merge_context}
{human_guidance}
{pathology_context}
{biochemistry_context}
{previous_bio_findings}
{code_history_context}

This is Round {analysis_state.current_round}. The Jupyter kernel retains all variables from previous rounds.
Focus on PI's specific goals above — do NOT re-run completed steps (QC, clustering, cell annotation, etc.).
Call report_findings ONLY after completing ALL assigned tasks above."""

    # On resume the agent restores its own conversation from the checkpoint,
    # so showing this freshly-built (and discarded) instruction would mislead.
    if not resuming:
        await self._send_message("prompt", instruction, AgentType.PI)

    # Step 2: Bioinformatics agent runs autonomously with figure review support
    await self._send_message("status", "Bioinformatics agent is running autonomous analysis...", AgentType.BIOINFORMATICS)

    # Reset supervisor state for this round
    self.ensure_agent_ready("supervisor")
    if self.supervisor_agent:
        self.supervisor_agent.reset_for_round(planning_stage=planning_stage)

    # Set current state references for _pi_callback plan deviation detection
    self._current_analysis_state = analysis_state
    self._current_graph_state = state

    # Open the round for artifact-staleness purposes. Everything that writes
    # a figure or table into the workspace runs below this line, so this is
    # the boundary `unchanged_artifact_stems` needs in order to blacklist
    # only what this round produced. It cannot be derived from
    # analysis_log: the agent accumulates its steps locally and they are
    # merged and stamped with round_number further down this same method,
    # i.e. never while the round is still running. Idempotent per round, so
    # stage 2 keeps stage 1's stamp — a stage-2 normalization does
    # invalidate a figure stage 1 drew.
    record_round_start(analysis_state)

    max_steps = self.config.bioinfo_max_steps
    return {
        "state": state, "analysis_state": analysis_state,
        "planning_stage": planning_stage, "instruction": instruction,
        "max_steps": max_steps,
    }

async def execute_bioinformatics_analysis(self, ctx):
    analysis_state = ctx["analysis_state"]
    planning_stage = ctx["planning_stage"]
    instruction = ctx["instruction"]
    max_steps = ctx["max_steps"]
    # No retry loop: a failed runtime save no longer raises for a person to
    # answer (spec 5.1). RuntimeUnavailable still propagates -- there is nothing
    # to retry when nothing was ever saved, and the route offers the re-run.
    result = await self.bioinformatics_agent.run_autonomous_analysis(
        analysis_state,
        instruction,
        max_steps=max_steps,
        figure_review_callback=self._figure_review_callback,
        supervisor_callback=self._supervisor_callback,
        planning_stage=planning_stage,  # Pass stage context to agent
    )

    return result

async def collect_bioinformatics_results(self, ctx, result):
    analysis_state = ctx["analysis_state"]
    planning_stage = ctx["planning_stage"]
    # Collect supervisor stats
    if self.supervisor_agent:
        supervision_stats = self.supervisor_agent.get_summary_stats()
        await self._send_message("supervisor_summary", json.dumps(supervision_stats), AgentType.SUPERVISOR)
        logger.info(
            f"[{self.session_id}] Supervisor: {supervision_stats['total_steps_monitored']} steps, "
            f"{supervision_stats['steps_with_technical_issues']} issues, "
            f"checkpoints: {supervision_stats['checkpoints_reached']}"
        )

    # Check result status and log accordingly
    result_status = result.get("status", "unknown")
    if result_status == "error":
        logger.error(f"[{self.session_id}] Bioinformatics analysis failed: {result.get('error', 'Unknown error')}")
        await self._send_message("error", f"Bioinformatics analysis encountered an error: {result.get('error', 'Unknown error')}", AgentType.BIOINFORMATICS)
    elif result_status in ("timeout", "max_steps_reached"):
        logger.warning(f"[{self.session_id}] Bioinformatics analysis reached step limit after {result.get('steps_executed', 0)} steps")
        await self._send_message("status", "Bioinformatics analysis reached step limit, proceeding with partial results", AgentType.BIOINFORMATICS)

    # Collect results
    all_findings = result.get("findings", [])
    all_figures = result.get("figures", [])

    existing_code_ids = {c.id for c in analysis_state.code_history}
    for code_exec in result.get("code_history", []):
        # Execution identity, not source text, defines evidence provenance.
        # The same code may fail first and succeed after runtime state changes;
        # collapsing by code hash would discard the successful evidence ID.
        if code_exec.id not in existing_code_ids:
            analysis_state.code_history.append(code_exec)
            existing_code_ids.add(code_exec.id)

    # Store analysis log (dedup by step id to handle node retries)
    existing_step_ids = {s.id for s in analysis_state.analysis_log}
    for step in result.get("analysis_log", []):
        if isinstance(step, dict):
            step = AnalysisStep(**step)
        if step.id not in existing_step_ids:
            step.round_number = max(analysis_state.current_round, 1)
            analysis_state.analysis_log.append(step)
            existing_step_ids.add(step.id)
    # Cap log size to prevent unbounded memory growth
    _MAX_ANALYSIS_LOG = 2000
    if len(analysis_state.analysis_log) > _MAX_ANALYSIS_LOG:
        analysis_state.analysis_log = analysis_state.analysis_log[-_MAX_ANALYSIS_LOG:]

    # Dedup publication_figures by lineage + name to handle node retries
    # without deleting the source copy retained for snapshot branches.
    seen_fig_names = {}
    for i, fig in enumerate(analysis_state.publication_figures):
        seen_fig_names[
            ((getattr(fig, "lineage_id", None) or "main"), fig.name)
        ] = i  # later index wins (keeps latest within one lineage)
    if len(seen_fig_names) < len(analysis_state.publication_figures):
        analysis_state.publication_figures = [
            analysis_state.publication_figures[i]
            for i in sorted(seen_fig_names.values())
        ]

    # Step 2a: platform-owned exploration visuals. QC, annotation, cluster,
    # batch/sample and marker baselines must not depend on the model remembering
    # to draw them. Stable names make both renderers idempotent and let a node
    # retry replace rather than duplicate an artifact. Missing optional fields
    # suppress only the panel that needs them.
    # Both stages of the exploration round, not just the first. Stage 1 and
    # Stage 2 re-enter this node with the same round number, and every
    # successful Stage-2 cell that mutates the expression representation calls
    # invalidate_current_artifacts for round 1 — which would set is_current=False
    # on a panel rendered at the end of Stage 1 and drop it from every surface
    # before `diagnostic` is ever consulted (that check is downstream of
    # is_current in is_report_eligible). Observed in session 86a9255a, where a
    # Stage-2 normalization cell invalidated the round's only round-1 figure.
    # Rendering again after Stage 2 also means every baseline describes the
    # representation the round actually ended on.
    if (
        planning_stage in ("stage1", "stage2", "full")
        and int(analysis_state.current_round or 0) <= 1
    ):
        try:
            await ensure_exploration_diagnostic_figure(
                self.bioinformatics_agent, analysis_state
            )
        except Exception as qc_diag_err:
            logger.warning(
                f"[{self.session_id}] QC diagnostic rendering skipped: {qc_diag_err!r}"
            )
        try:
            await ensure_stage1_baseline_figures(
                self.bioinformatics_agent, analysis_state
            )
        except Exception as baseline_err:
            logger.warning(
                f"[{self.session_id}] Stage 1 baseline rendering skipped: {baseline_err!r}"
            )
    elif should_render_innovation_baselines(analysis_state.current_round):
        # The same guarantee for rounds 2+. Every panel an innovation round
        # shows is otherwise the executor's own choice, so a round could
        # execute a dozen method families and register almost nothing, and
        # nothing downstream can turn a computed result into an artifact once
        # the round has settled. Rendered before the plan audit below, so the
        # PI reviews the round as it will actually be reported. Fail-soft: a
        # missing panel must never fail a completed analysis.
        try:
            await ensure_innovation_baseline_figures(
                self.bioinformatics_agent, analysis_state
            )
        except Exception as innovation_err:
            logger.warning(
                f"[{self.session_id}] Innovation baseline rendering skipped: "
                f"{innovation_err!r}"
            )

    # Adoption first. It binds workspace renders the executor produced with a
    # bare savefig to unmet plan requirements, and the plan audit below reads
    # `state.publication_figures` and nothing else — so run after it, those
    # records exist but no requirement ever consults them. That was harmless
    # only while `figure_unmet` was permanently empty; it now reaches the PI's
    # review prompt, where a round whose figures were adopted would otherwise be
    # reported as having produced none. The post-adoption `audit_figure_contract`
    # below is unaffected: it still re-reads after every workflow-owned pass.
    await self._adopt_workspace_figures(analysis_state, planning_stage)

    # Step 2b: audit the approved plan against what actually ran, then — only
    # on hard structural signals — let the PI commission a bounded repair pass.
    # Both run inside this node so the round still produces exactly one
    # RoundResult for one direction: max_rounds, the direction queue and the
    # integration bookkeeping are untouched. Neither can fail the round.
    reconciliation = await self._reconcile_plan_evidence(analysis_state, result)
    audit_summary = self._audit_round_plan_execution(
        analysis_state, result, reconciliation
    )
    review_meta, review_merge = await self._run_post_execution_review(
        analysis_state, result, audit_summary, planning_stage
    )
    if review_merge:
        all_findings, all_figures = self._merge_supplementary_into_round(
            analysis_state, result, all_findings, all_figures, review_merge
        )
        # Re-audit so the persisted plan reflects the repair, not the state
        # that triggered it.
        reconciliation = await self._reconcile_plan_evidence(
            analysis_state, result
        )
        audit_summary = self._audit_round_plan_execution(
            analysis_state, result, reconciliation
        )
        review_meta["plan_audit"] = audit_summary

    # Re-evaluate after deterministic baselines and any supplementary work.
    # The executor's own audit was taken before these workflow-owned repair
    # paths ran, so carrying it forward would mark a figure missing even
    # after the platform generated it successfully.
    figure_contract = audit_figure_contract(
        analysis_state,
        plan=analysis_state.current_plan,
        planning_stage=planning_stage,
    )
    result["figure_contract"] = figure_contract.summary()

    ctx.update(
        all_findings=all_findings, all_figures=all_figures,
        audit_summary=audit_summary, review_meta=review_meta,
        figure_contract=figure_contract,
    )

async def finalize_bioinformatics_analysis(self, ctx, result):
    state = ctx["state"]
    analysis_state = ctx["analysis_state"]
    planning_stage = ctx["planning_stage"]
    all_findings = ctx["all_findings"]
    all_figures = ctx["all_figures"]
    audit_summary = ctx["audit_summary"]
    review_meta = ctx["review_meta"]
    figure_contract = ctx["figure_contract"]
    # Step 3: PI summarizes results
    await self._send_message("status", "PI is reviewing analysis results...", AgentType.PI)

    summary = await self.pi_agent.summarize_bioinformatics_analysis(
        analysis_state,
        [{"findings": all_findings, "figures": all_figures}]
    )

    effective_status = result.get("status", "completed")

    final_result = {
        "findings": all_findings,
        "finding_records": list(result.get("finding_records") or []),
        "figures": all_figures,
        "summary": summary,
        "steps_executed": result.get("steps_executed", 0),
        "status": effective_status,
        "mechanism_programs": result.get("mechanism_programs", []),
        # Evidence binding must survive this dict. It is read by three
        # consumers that all fail silently without it: the completeness gate in
        # derive_round_completion_status (which withholds direction-coverage
        # eligibility from a round whose findings are not bound to successful
        # executions), the evidence note handed to the PI at synthesis, and the
        # report's hedging. Dropping the keys made every round look verified,
        # because each of those consumers tests `is False` and an absent key
        # reads as None.
        "evidence_verified": result.get("evidence_verified"),
        "evidence_gate_degraded": bool(
            result.get("evidence_gate_degraded")
            or any(
                item.get("forced_release")
                for item in result.get("evidence_gate_findings") or []
                if isinstance(item, dict)
            )
        ),
        "evidence_execution_ids": list(result.get("evidence_execution_ids") or []),
        "evidence_gate_findings": list(result.get("evidence_gate_findings") or []),
        # The executing agent's own view of what is left undone. It is present
        # in every real report_findings call and used to be dropped here, so
        # the PI synthesizing the round never saw it.
        "agent_suggestions": self._collect_agent_suggestions(result),
        "plan_audit": audit_summary,
        "post_execution_review": review_meta,
        "figure_contract": figure_contract.summary(),
    }

    _saved_fig_count = count_saved_figures(result.get("analysis_log", []))
    logger.info(f"[{self.session_id}] Bioinformatics analysis completed: {len(all_findings)} findings, {len(all_figures)} inline figures, {_saved_fig_count} saved figures, {result.get('steps_executed', 0)} steps")

    await self._send_message(
        "result",
        {
            "findings": all_findings,
            "finding_records": final_result["finding_records"],
            "summary": summary,
            "steps_executed": result.get("steps_executed", 0),
            "evidence_verified": final_result["evidence_verified"],
            "evidence_gate_degraded": final_result["evidence_gate_degraded"],
            "evidence_execution_ids": final_result["evidence_execution_ids"],
            "evidence_gate_findings": final_result["evidence_gate_findings"],
        },
        AgentType.BIOINFORMATICS
    )

    analysis_state.add_message(
        AgentType.BIOINFORMATICS,
        MessageRole.ASSISTANT,
        json.dumps({
            "findings": all_findings,
            "finding_records": final_result["finding_records"],
            "summary": summary,
        }, ensure_ascii=False)
    )

    # Mid-execution plan revision is signalled on the raw agent `result`, not
    # on `final_result` (rebuilt above with only a fixed set of keys, so it
    # never carries plan_revision_needed). A "paused_for_revision" return means
    # the agent stopped partway through the current plan — NOT a stage
    # completion — so the stage must not advance here.
    plan_revision_requested = bool(result.get("plan_revision_needed"))
    settlement_status = (
        "forced_stop" if result.get("forced_stop") else effective_status
    )
    # The counter is keyed by round+stage because that is the scope the
    # cycle re-enters: it bypasses check_convergence, so `current_round`
    # does not move and a fresh key would never be minted.
    revision_key = f"{analysis_state.current_round}:{planning_stage}"
    prior_attempts = int(
        (analysis_state.execution_revision_attempts or {}).get(revision_key, 0)
    )
    action = decide_bioinfo_post_execution(
        planning_stage,
        plan_revision_requested,
        result_status=settlement_status,
        incomplete_attempts=prior_attempts,
    )

    gate_observations = final_result.setdefault("evidence_gate_findings", [])
    status_was_unclean = str(settlement_status or "").strip().lower() not in (
        "completed",
        "partial",
    )
    if action == "incomplete":
        analysis_state.execution_revision_attempts = {
            **(analysis_state.execution_revision_attempts or {}),
            revision_key: prior_attempts + 1,
        }
    elif status_was_unclean:
        # Released at the ceiling. Recorded as an observation, not as a
        # forced release: `forced_release` is read one frame up as
        # `evidence_gate_degraded`, which withholds direction-coverage
        # eligibility from the round. The round already carries its unclean
        # status, which post-execution review reads directly, so the PI
        # learns this without that side effect.
        gate_observations.append(_evidence_gate_record(
            gate="execution-revision-ceiling",
            rule_id="scai-native/execution-revision-ceiling",
            subject=revision_key,
            detail=(
                f"Execution ended as '{settlement_status}' after "
                f"{prior_attempts} plan revision(s) in this stage. The "
                "revision budget is spent; the round settles as degraded "
                "rather than re-planning the same failure again."
            ),
            severity="advise",
            rejection_count=prior_attempts,
            rejection_limit=MAX_EXECUTION_REVISION_ATTEMPTS,
        ))

    if not figure_contract.passed:
        # No longer a gate. Measured here because this audit runs after the
        # workflow-owned repair paths (deterministic baselines, the
        # supplementary pass) and therefore sees more than the executor's
        # own advisory did.
        gate_observations.append(_evidence_gate_record(
            gate="required-figure-observation",
            rule_id="scai-native/required-figure-observation",
            subject="required-publication-figures",
            detail=(
                "Required result figures were not complete when the round "
                "settled. They are unrendered, stale, or held back by "
                "review/layout gates, so the PI must decide whether to "
                "repair them, accept the result with a limitation, or drop "
                "the deliverable: "
                + json.dumps(
                    figure_contract.summary().get("missing", []),
                    ensure_ascii=False,
                )
            ),
            severity="advise",
        ))

    # One-shot: the recompute prompt has now been delivered and acted on.
    analysis_state.runtime_recompute_pending = False
    # Likewise the pair restore (B2-2): the executor has read the checkpoint
    # under the exact-revision rule, so from here a checkpoint ahead of the
    # committed generation is an ordinary live kernel again.
    analysis_state.runtime_pair_restored_revision = None

    result_state = {
        **state,
        "analysis_state": analysis_state,
        "bioinformatics_result": final_result
    }

    if action in ("revision", "incomplete"):
        # Stay in the current stage and keep current_plan intact: after the
        # revision is approved the agent resumes this same plan from its
        # checkpoint (e.g. stage1's remaining annotation steps). Do NOT cache
        # the partial result and do NOT advance the stage.
        result_state["plan_revision_needed"] = True
        # P0-8: invalidate cache so post-revision re-execution actually runs
        # instead of returning the stale (pre-revision) cached result.
        cache_invalidate(analysis_state, AgentType.BIOINFORMATICS)
        if action == "revision":
            revision_context = result.get("plan_revision_context", {})
        else:
            revision_context = {
                "reason": "Bioinformatics execution did not satisfy the completion contract.",
                "result_status": effective_status,
                "figure_contract": figure_contract.summary(),
            }
        result_state["plan_revision_context"] = revision_context
        logger.info(
            f"[{self.session_id}] Execution revision triggered in stage "
            f"'{planning_stage}' ({action}); staying in stage, bioinfo cache invalidated"
        )
        # This revision signal exists only on the node output. LangGraph has
        # not installed that output yet, so the workflow-level pointer still
        # names the input; move it forward for everything that reads it.
        self._current_graph_state = result_state
    elif action == "stage1_complete":
        # Stage transition: stage1 genuinely completed → set up for stage2.
        # Don't cache the stage1 result — stage2 re-enters this node and must
        # run fresh. Biochem/pathology/literature cache survives so stage2
        # reuses them.
        result_state["planning_stage"] = "stage2"
        result_state["plan_approved"] = False
        result_state["pi_instructions"] = _reset_pi_instructions_for_stage2(
            state.get("pi_instructions", {})
        )
        result_state["planning_context"] = {
            "stage1_summary": final_result.get("summary", ""),
            "stage1_findings": final_result.get("findings", []),
            "stage1_figures": final_result.get("figures", []),
            "stage1_steps_executed": final_result.get("steps_executed", 0),
        }
        analysis_state.current_plan = None
        cache_invalidate(analysis_state, AgentType.BIOINFORMATICS)
    else:  # "complete"
        cache_store(analysis_state, AgentType.BIOINFORMATICS, final_result)
        if planning_stage == "stage2":
            result_state["planning_stage"] = "full"
            result_state["planning_context"] = None  # Clear stage1 context after stage2 completes

    return result_state


async def run_bioinformatics_analysis(self, state):
    started = await start_bioinformatics_analysis(self, state)
    if "_continue" not in started:
        return started
    ctx = await prepare_bioinformatics_instruction(
        self, state, started["_continue"]
    )
    result = await execute_bioinformatics_analysis(self, ctx)
    await collect_bioinformatics_results(self, ctx, result)
    return await finalize_bioinformatics_analysis(self, ctx, result)
