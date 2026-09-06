"""Bounded autonomous tool-turn cycle.

The facade globals are bound at call time so historical monkeypatch targets
continue to affect a resumed executor without creating an import cycle.
"""
from __future__ import annotations

from ...constants import NO_OUTPUT_TOKEN_CAP
from .executor_auxiliary import LoopExit, TurnEffects, TurnStart, finalize_run


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


async def start_turn(context) -> TurnStart:
    if context.agent._pause_event is not None and not context.agent._pause_event.is_set():
        logger.info(
            f"[Bioinformatics] Pause requested at step {context.step_count}, "
            f"waiting for resume"
        )
        try:
            await context.agent._notify("status", "Analysis paused, awaiting resume...")
        except Exception:
            pass
        await context.agent._pause_event.wait()
        logger.info(f"[Bioinformatics] Resumed at step {context.step_count}")
        try:
            await context.agent._notify("status", "Analysis resumed")
        except Exception:
            pass
    session_id = getattr(context.agent, "session_id", None) or getattr(context.analysis_state, "id", None)
    if session_id:
        from backend.services.snapshot_v2.safe_point import (
            get_snapshot_safe_point_manager,
        )
        sp = get_snapshot_safe_point_manager()
        if sp.has_pending_request(session_id):
            gen = sp.get_pending_generation(session_id)
            if gen is not None:
                await sp.acknowledge_safe_point_request(
                    session_id, gen, role="workflow",
                )
        if sp.is_paused(session_id):
            logger.info(
                f"[Bioinformatics] Restore safe-point active at step "
                f"{context.step_count}; freezing until release"
            )
            await sp.wait_for(session_id)
            if sp.is_superseded(session_id):
                from backend.workflow.helpers import (
                    WorkflowSupersededError,
                )
                logger.info(
                    f"[Bioinformatics] Run superseded by a snapshot "
                    f"restore at step {context.step_count}; unwinding"
                )
                raise WorkflowSupersededError(
                    "snapshot restore replaced this run's checkpoint"
                )
    context.step_count += 1
    context.force_break = False  # Reset per-iteration
    if context.checkpoint_manager.should_checkpoint(context.step_count):
        try:
            await context.save_checkpoint(
                f'step_{context.step_count}',
                save_sandbox=True,
            )
        except Exception as e:
            # A checkpoint save that failed leaves the previous committed
            # generation; the run continues and coherence reports it stale.
            logger.warning(f"[Bioinformatics] Checkpoint save failed at step {context.step_count}: {e}")
    context.agent._set_llm_context()
    MAX_MESSAGES = 80
    TRIM_TARGET = 50  # Large trim gap avoids frequent cache-breaking trims
    if len(context.messages) > MAX_MESSAGES:
        system_msg = context.messages[0]
        content_val = context.messages[1].get("content", "") if len(context.messages) > 1 else ""
        is_dynamic = False
        if isinstance(content_val, str):
            is_dynamic = content_val.startswith(_DYNAMIC_CONTEXT_PREFIX)
        elif isinstance(content_val, list):
            is_dynamic = any(p.get("text", "").startswith(_DYNAMIC_CONTEXT_PREFIX) for p in content_val if isinstance(p, dict))
        dynamic_msg = context.messages[1] if len(context.messages) > 1 and is_dynamic else None
        anchor_count = 2 if dynamic_msg else 1
        keep_tail = TRIM_TARGET - anchor_count - 1  # -1 for the summary message
        cut = len(context.messages) - keep_tail
        while cut < len(context.messages) and context.messages[cut].get("role") == "tool":
            cut -= 1
        if cut < anchor_count:
            cut = anchor_count
        dropped = context.messages[anchor_count:cut]
        summary = context.agent._summarize_dropped_messages(dropped)
        summary_msg = {
            "role": "user",
            "content": f"[Previous analysis summary]\n{summary}"
        }
        if dynamic_msg:
            context.messages = [system_msg, dynamic_msg, summary_msg] + context.messages[cut:]
        else:
            context.messages = [system_msg, summary_msg] + context.messages[cut:]
        logger.info(f"Trimmed message history: dropped {len(dropped)} context.messages, {len(context.messages)} remaining")
    context.messages, _compacted = context.agent._compact_context(
        context.messages, context.agent.COMPACT_THRESHOLD_TOKENS
    )
    if _compacted:
        logger.info(
            f"[Bioinformatics] Context compacted: "
            f"estimated {context.agent._estimate_context_tokens(context.messages)} tokens"
        )
    if context.annotation_retry_required and context.annotation_retry_attempts <= 3:
        nudge = (
            f"[System] annotate_clusters has not completed (attempt "
            f"{context.annotation_retry_attempts}). Your next action MUST be to either "
            f"(a) run execute_code that prepares the sandbox (e.g., assign "
            f"adata.obs['leiden'] from your actual clustering column, ensure "
            f"sc.tl.rank_genes_groups has populated adata.uns), or (b) call "
            f"annotate_clusters again with the correct cluster_key/rank_key "
            f"arguments. Do NOT call report_findings, consult_pi about other "
            f"topics, or generate_publication_figure until annotation completes."
        )
        nudge_key = f"__annot_retry_nudge_{context.annotation_retry_attempts}"
        if nudge_key not in context.agent._recent_sup_feedback:
            context.messages.append({"role": "user", "content": nudge})
            context.agent._recent_sup_feedback.add(nudge_key)
    if context.annotation_retry_required and context.annotation_retry_attempts > 3:
        logger.warning(
            f"[Bioinformatics] annotate_clusters retries exhausted "
            f"({context.annotation_retry_attempts}); halting loop for human review."
        )
        await context.agent._notify(
            "status",
            "Cell type annotation could not be completed automatically. "
            "Please review the session and resume manually.",
        )
        return TurnStart(exit=LoopExit.TERMINAL)
    effective_tool_choice = context.agent._compute_tool_choice(
        context.text_response_count, context.max_text_responses
    )
    if context.empty_response_count > 0 and effective_tool_choice == "auto":
        effective_tool_choice = "required"
        logger.info(
            f"[Bioinformatics] Escalating tool_choice to 'required' "
            f"after {context.empty_response_count} empty response(s)"
        )
    try:
        response = await context.agent._budgeted_text_call(
            context.agent.llm_client.chat_with_tools,
            messages=context.messages,
            max_tokens=NO_OUTPUT_TOKEN_CAP,
            anchor_count=context.agent._context_anchor_count(context.messages),
            summarize=context.agent._summarize_dropped_messages,
            model=context.agent.model,
            tools=TOOLS,
            tool_choice=effective_tool_choice,
            temperature=context.agent.config.temperature,
            prefer_stream=True,
        )
    except UpstreamExhaustedError:
        logger.error(f"[Bioinformatics] Upstream exhausted, propagating to @with_retry")
        raise
    except Exception as e:
        context.llm_error = str(e)
        await context.agent._notify("error", f"LLM call failed: {context.llm_error}")
        return TurnStart(exit=LoopExit.TERMINAL)
    agent_reasoning = response.content[:2000] if response.content else None
    if context.agent._is_empty_response(response):
        context.empty_response_count += 1
        logger.warning(
            f"[Bioinformatics] Empty/truncated response "
            f"({context.empty_response_count}/{context.max_empty_responses})"
        )
        context.messages, _reactive_compacted = context.agent._compact_context(
            context.messages, context.agent.COMPACT_TARGET_TOKENS
        )
        if _reactive_compacted:
            est = context.agent._estimate_context_tokens(context.messages)
            logger.info(
                f"[Bioinformatics] Reactive context compaction after empty "
                f"response: estimated {est} tokens"
            )
            await context.agent._notify(
                "status",
                f"Context compacted due to large outputs (≈{est:,} tokens), retrying…"
            )
        if context.empty_response_count >= context.max_empty_responses:
            try:
                await context.save_checkpoint(
                    f'step_{context.step_count}_empty_exhausted',
                )
            except Exception as e:
                logger.warning(f"[Bioinformatics] Pre-empty-exhaustion checkpoint save failed: {e}")
            raise UpstreamExhaustedError(
                f"[Bioinformatics] {context.max_empty_responses} consecutive empty responses",
                agent_name="bioinformatics",
            )
        if not _reactive_compacted:
            context.messages.append({
                "role": "user",
                "content": (
                    f"[System] Your previous reply was empty — no text and no "
                    f"tool call (attempt {context.empty_response_count}). An empty turn "
                    f"is not a valid step. You MUST respond now with exactly one "
                    f"concrete action: call execute_code to run the next analysis "
                    f"step, call another appropriate tool, or call report_findings "
                    f"if every assigned task is already complete. Do not return an "
                    f"empty message."
                ),
            })
        await asyncio.sleep(context.agent._empty_retry_backoff(context.empty_response_count))
        return TurnStart()
    context.empty_response_count = 0
    if not response.has_tool_calls:
        if response.content:
            await context.agent._notify("response", response.content)
            context.messages.append(response.raw_assistant_message or {
                "role": "assistant", "content": response.content
            })
        context.analysis_log.append(AnalysisStep(
            step_number=context.step_count, step_type="text_response",
            agent_reasoning=agent_reasoning,
            text_content=(response.content or "")[:2000],
        ))
        agent_reasoning = None
        context.text_response_count += 1
        if context.text_response_count >= context.max_text_responses:
            logger.error(
                f"[Bioinformatics] {context.max_text_responses} consecutive text-only "
                f"responses; escalating to LLM-failure HITL via UpstreamExhaustedError"
            )
            try:
                await context.save_checkpoint(
                    f'step_{context.step_count}_text_only_exhausted',
                )
            except Exception as e:
                logger.warning(f"[Bioinformatics] Pre-escalation checkpoint save failed: {e}")
            context.agent._raise_text_only_exhausted(context.text_response_count)
        context.messages.append({
            "role": "user",
            "content": "Please continue the analysis. If you have completed your analysis, "
                       "call the report_findings tool with your key findings. "
                       "Do NOT respond with text only - you must use a tool."
        })
        return TurnStart()
    return TurnStart(response=response)


async def process_response(context, response, agent_reasoning) -> TurnEffects:
    self = context.agent
    state = context.analysis_state
    messages = context.messages
    analysis_log = context.analysis_log
    step_count = context.step_count
    planning_stage = context.planning_stage
    review_callback = context.review_callback
    all_figures = context.figures
    all_findings = context.findings
    mechanism_programs = context.mechanism_programs
    annotation_completed = context.annotation_completed
    annotation_retry_required = context.annotation_retry_required
    annotation_retry_attempts = context.annotation_retry_attempts
    consecutive_errors = context.consecutive_errors
    total_errors = context.total_errors
    figures_rendered = context.figures_rendered
    evidence_gate_degraded = context.evidence_gate_degraded
    evidence_gate_findings = context.evidence_gate_findings
    reported_evidence_ids = context.reported_evidence_ids
    accepted_finding_records = context.accepted_finding_records
    mechanism_programs_registered = context.mechanism_programs_registered
    completed = context.completed
    _plan_revision_needed = context.plan_revision_needed
    _plan_revision_context = context.plan_revision_context
    _save_executor_checkpoint = context.save_checkpoint
    _supervisor_callback = context.supervisor_callback
    # Track what kind of work this iteration contains for step budget decisions
    has_substantive_work = False   # execute_code/consult_pi/search_literature/annotate_clusters/report_findings
    has_figure_work = False        # generate_publication_figure actually rendered
    has_register_score = False     # register_score
    _plan_revision_needed = False  # Set when PI indicates plan needs revision
    _plan_revision_context = {}
    # Add assistant message ONCE before processing tool calls.
    # Using raw_assistant_message preserves provider-specific fields such as
    # Gemini's thought_signature, which must be echoed back in subsequent requests.
    if response.raw_assistant_message:
        messages.append(response.raw_assistant_message)
    else:
        messages.append({
            "role": "assistant",
            "content": response.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}
                }
                for tc in response.tool_calls
            ]
        })
    _step_figures: List[str] = []  # Figures generated in this iteration (for supervisor)
    tool_runtime = ToolRuntime(
        agent=self,
        state=state,
        messages=messages,
        analysis_log=analysis_log,
        step_count=step_count,
        agent_reasoning=agent_reasoning,
        planning_stage=planning_stage,
        review_callback=review_callback,
        step_figures=_step_figures,
        all_figures=all_figures,
        all_findings=all_findings,
        mechanism_programs=mechanism_programs,
        annotation_completed=annotation_completed,
        annotation_retry_required=annotation_retry_required,
        annotation_retry_attempts=annotation_retry_attempts,
        consecutive_errors=consecutive_errors,
        total_errors=total_errors,
        figures_rendered=figures_rendered,
        evidence_gate_degraded=evidence_gate_degraded,
        evidence_gate_findings=evidence_gate_findings,
        reported_evidence_ids=reported_evidence_ids,
        accepted_finding_records=accepted_finding_records,
        mechanism_programs_registered=mechanism_programs_registered,
        completed=completed,
        plan_revision_needed=_plan_revision_needed,
        plan_revision_context=_plan_revision_context,
        save_checkpoint=_save_executor_checkpoint,
        supervisor_callback=_supervisor_callback,
    )
    dispatch_context = ToolDispatchContext(
        messages=messages,
        known_tool_names=KNOWN_TOOL_NAMES,
        logger=logger,
        handlers={
            "execute_code": functools.partial(handle_execute_code, tool_runtime),
            "report_findings": functools.partial(
                handle_report_findings, tool_runtime
            ),
            "consult_pi": functools.partial(handle_consult_pi, tool_runtime),
            "annotate_clusters": functools.partial(
                handle_annotate_clusters, tool_runtime
            ),
            "search_literature": functools.partial(
                handle_search_literature, tool_runtime
            ),
            "lookup_documentation": functools.partial(
                handle_lookup_documentation, tool_runtime
            ),
            "generate_publication_figure": functools.partial(
                handle_publication_figure, tool_runtime
            ),
            "fetch_gene_set": functools.partial(
                handle_fetch_gene_set, tool_runtime
            ),
            "register_analysis_plan": functools.partial(
                handle_register_analysis_plan, tool_runtime
            ),
            "register_method_decision": functools.partial(
                handle_register_method_decision, tool_runtime
            ),
            "register_publication_table": functools.partial(
                handle_register_publication_table, tool_runtime
            ),
            "register_score": functools.partial(
                handle_register_score, tool_runtime
            ),
            "register_mechanism_programs": functools.partial(
                handle_register_mechanisms, tool_runtime
            ),
            "validate_species_genes": functools.partial(
                handle_validate_species, tool_runtime
            ),
        },
        propagate_exceptions=(UpstreamExhaustedError,),
        error_formatter=lambda exc: compact_traceback(str(exc), max_chars=3000),
    )
    for tool_call in response.tool_calls:
        tool_name = tool_call.name
        args = tool_call.arguments
        dispatch_result = await dispatch_tool(dispatch_context, tool_call)
        if dispatch_result.status is not DispatchStatus.LEGACY_HANDLER:
            if tool_name not in KNOWN_TOOL_NAMES:
                analysis_log.append(AnalysisStep(
                    step_number=step_count, step_type="unknown_tool",
                    agent_reasoning=agent_reasoning,
                    text_content=dispatch_result.content[:2000],
                ))
            elif dispatch_result.status is DispatchStatus.ERROR:
                analysis_log.append(AnalysisStep(
                    step_number=step_count,
                    step_type="tool_error",
                    agent_reasoning=agent_reasoning,
                    text_content=dispatch_result.content[:2000],
                ))
            if dispatch_result.value is ToolAction.BREAK:
                break
            continue
    _plan_revision_needed = tool_runtime.plan_revision_needed
    _plan_revision_context = tool_runtime.plan_revision_context
    has_substantive_work = (
        has_substantive_work or tool_runtime.has_substantive_work
    )
    has_figure_work = has_figure_work or tool_runtime.has_figure_work
    figures_rendered = tool_runtime.figures_rendered
    has_register_score = has_register_score or tool_runtime.has_register_score
    mechanism_programs_registered = tool_runtime.mechanism_programs_registered
    annotation_completed = tool_runtime.annotation_completed
    annotation_retry_required = tool_runtime.annotation_retry_required
    annotation_retry_attempts = tool_runtime.annotation_retry_attempts
    consecutive_errors = tool_runtime.consecutive_errors
    total_errors = tool_runtime.total_errors
    evidence_gate_degraded = (
        context.evidence_gate_degraded
        or evidence_gate_degraded
        or tool_runtime.evidence_gate_degraded
    )
    evidence_gate_findings = tool_runtime.evidence_gate_findings
    reported_evidence_ids = tool_runtime.reported_evidence_ids
    accepted_finding_records = tool_runtime.accepted_finding_records
    completed = tool_runtime.completed
    # Clear reasoning after all tool calls in this iteration are processed
    agent_reasoning = None
    # Reset text-only counter when a substantive tool call was made
    if has_substantive_work and response.has_tool_calls:
        text_response_count = 0
    context.plan_revision_needed = _plan_revision_needed
    context.plan_revision_context = _plan_revision_context
    context.annotation_completed = annotation_completed
    context.annotation_retry_required = annotation_retry_required
    context.annotation_retry_attempts = annotation_retry_attempts
    context.consecutive_errors = consecutive_errors
    context.total_errors = total_errors
    context.figures_rendered = figures_rendered
    context.evidence_gate_degraded = evidence_gate_degraded
    context.evidence_gate_findings = evidence_gate_findings
    context.reported_evidence_ids = reported_evidence_ids
    context.accepted_finding_records = accepted_finding_records
    context.mechanism_programs_registered = mechanism_programs_registered
    context.completed = completed
    return TurnEffects(
        response=response,
        step_figures=_step_figures,
        has_substantive_work=has_substantive_work,
        has_figure_work=has_figure_work,
        has_register_score=has_register_score,
    )


async def finish_turn(context, effects) -> LoopExit:
    self = context.agent
    messages = context.messages
    analysis_log = context.analysis_log
    _supervisor_callback = context.supervisor_callback
    _step_figures = effects.step_figures
    response = effects.response
    has_substantive_work = effects.has_substantive_work
    has_figure_work = effects.has_figure_work
    has_register_score = effects.has_register_score
    agent_reasoning = None
    text_response_count = context.text_response_count
    force_break = context.force_break
    all_findings = context.findings
    evidence_gate_degraded = context.evidence_gate_degraded
    evidence_gate_findings = context.evidence_gate_findings
    completed = context.completed
    _plan_revision_needed = context.plan_revision_needed
    _plan_revision_context = context.plan_revision_context
    _save_executor_checkpoint = context.save_checkpoint
    step_count = context.step_count
    figure_iteration_count = context.figure_iteration_count
    register_score_iteration_count = context.register_score_iteration_count
    figures_rendered = context.figures_rendered
    MAX_FREE_FIGURE_ITERATIONS = context.max_free_figure_iterations
    MAX_FREE_REGISTER_SCORE_ITERATIONS = context.max_free_register_score_iterations
    # === SUPERVISOR monitoring (every step) ===
    if _supervisor_callback and analysis_log:
        try:
            sup_result = await _supervisor_callback(
                analysis_log[-1],   # Latest step
                _step_figures       # Figures from this iteration (may be empty)
            )
            if sup_result.feedback:
                # De-duplicate: skip if the same feedback (first 120 chars)
                # was already injected recently to avoid wasting tokens on
                # repeated identical warnings.
                sup_key = sup_result.feedback[:120]
                if sup_key not in self._recent_sup_feedback:
                    messages.append({
                        "role": "user",
                        "content": f"[Supervisor] {sup_result.feedback}"
                    })
                    self._recent_sup_feedback.add(sup_key)
                else:
                    logger.debug(f"[Bioinformatics] Skipping duplicate supervisor feedback: {sup_key[:60]}...")
            if sup_result.force_skip:
                current_err_type = extract_error_type(analysis_log[-1].code_error) if analysis_log[-1].code_error else None
                if self._force_skip_error_type != current_err_type:
                    self._force_skip_counter = 0
                    self._force_skip_error_type = current_err_type
                self._force_skip_counter += 1
                if self._force_skip_counter <= 2:
                    messages.append({
                        "role": "user",
                        "content": (
                            "[Supervisor] MANDATORY: You must skip the current step immediately. "
                            "Do NOT retry this code or attempt to fix it. "
                            "Move on to the next analysis step in your plan. "
                            "This instruction is not optional — you will be forced to "
                            "report findings if you continue on this path."
                        )
                    })
                else:
                    # 3rd force-skip on same error type: the agent has ignored
                    # two mandatory-skip instructions. Force a hard exit from
                    # the tool-call loop by injecting report_findings and breaking.
                    logger.warning(
                        f"[Bioinformatics] Force-skip ignored after {self._force_skip_counter} "
                        f"attempts for {current_err_type}; forcing report_findings"
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "[System] You have repeatedly failed to skip the problematic step "
                            "despite mandatory instructions. You MUST call report_findings now "
                            "with whatever results you have so far. Do NOT attempt any more code execution."
                        )
                    })
                    self._force_skip_counter = 0
                    self._force_skip_error_type = None
                    force_break = True
        except Exception as e:
            logger.warning(f"[Bioinformatics] Supervisor callback error: {e}")
    # === END SUPERVISOR ===
    # Force-break: supervisor force-skip was ignored 3 times; the loop
    # must exit now. Mark completed so the return path produces a valid
    # status rather than falling through to max_steps_reached with empty findings.
    if force_break:
        logger.info("[Bioinformatics] Force-break triggered by supervisor force-skip")
        if not all_findings:
            all_findings.append(
                "[Supervisor] Analysis was forcibly stopped: the agent was stuck "
                "on a repeated error despite mandatory-skip instructions and "
                "PI guidance. Results below are partial."
            )
        # A forced stop is not a clean completion.  Reuse the existing
        # degraded-evidence channel so the workflow cannot advance a
        # stage or report the round as fully verified.
        evidence_gate_degraded = True
        evidence_gate_findings = [
            _evidence_gate_record(
                gate="supervisor-force-stop",
                rule_id="scai-native/supervisor-force-stop",
                detail="Analysis was forcibly stopped after repeated execution failures.",
                severity="block",
                forced_release=True,
            )
        ]
        completed = True
        context.force_break = True
        context.evidence_gate_degraded = evidence_gate_degraded
        context.evidence_gate_findings = evidence_gate_findings
        context.completed = True
        return LoopExit.TERMINAL
    # Check if plan revision was triggered during this step
    if _plan_revision_needed:
        # The consult_pi plan-revision path breaks the per-tool_call loop
        # early, so sibling tool_calls in this assistant message may lack
        # results. Backfill placeholders so the saved checkpoint stays
        # resumable (orphaned tool_calls cause an upstream HTTP 400).
        append_missing_tool_results(
            messages,
            [tc.id for tc in response.tool_calls],
            note="[System: plan revision was triggered; this tool call was not executed.]",
        )
        # Plan revision is a semantic execution boundary. Persist the
        # exact transcript and Jupyter state now instead of relying on
        # the five-step periodic checkpoint, which could replay work.
        try:
            await _save_executor_checkpoint(
                f'step_{step_count}_plan_revision',
                save_sandbox=True,
            )
        except Exception as e:
            logger.warning(
                "[Bioinformatics] Plan-revision checkpoint save failed at "
                "step %s: %s",
                step_count,
                e,
            )
        logger.info("[Bioinformatics] Plan revision needed, pausing execution")
        # Result dict will be populated after the loop exits
        return LoopExit.PLAN_REVISED
    # Metadata-only iterations (no substantive work) don't count toward step budget.
    # When both figure and register_score are present, the figure allowance
    # takes precedence; beyond the free allowance it consumes a normal step.
    if not has_substantive_work and response.has_tool_calls:
        if has_figure_work:
            figure_iteration_count += 1
            if figure_iteration_count <= MAX_FREE_FIGURE_ITERATIONS:
                step_count -= 1
                logger.debug(
                    f"[Bioinformatics] Successful figure-only step excluded "
                    f"({figure_iteration_count}/{MAX_FREE_FIGURE_ITERATIONS} free, "
                    f"{figures_rendered} rendered), step_count={step_count}"
                )
            else:
                logger.debug(
                    "[Bioinformatics] Free figure allowance exhausted; "
                    "this successful figure consumes a normal analysis step"
                )
        elif has_register_score:
            register_score_iteration_count += 1
            if register_score_iteration_count <= MAX_FREE_REGISTER_SCORE_ITERATIONS:
                step_count -= 1
                logger.debug(
                    f"[Bioinformatics] register_score-only step excluded "
                    f"({register_score_iteration_count}/"
                    f"{MAX_FREE_REGISTER_SCORE_ITERATIONS} free), "
                    f"step_count={step_count}"
                )
            else:
                logger.debug(
                    "[Bioinformatics] Free register_score allowance exhausted; "
                    "this successful registration consumes a normal analysis step"
                )
    context.text_response_count = text_response_count
    context.force_break = force_break
    context.evidence_gate_degraded = evidence_gate_degraded
    context.evidence_gate_findings = evidence_gate_findings
    context.completed = completed
    context.plan_revision_needed = _plan_revision_needed
    context.plan_revision_context = _plan_revision_context
    context.step_count = step_count
    context.figure_iteration_count = figure_iteration_count
    context.register_score_iteration_count = register_score_iteration_count
    return LoopExit.CONTINUE


async def prepare_loop_context(
    self,
    state: AnalysisState,
    initial_instruction: str,
    max_steps: int = 200,
    figure_review_callback: Optional[Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]] = None,
    supervisor_callback: Optional[Callable] = None,
    resume_from_checkpoint: bool = True,
    planning_stage: PlanStage = "full",  # "stage1", "stage2", or "full"
) -> Dict[str, Any]:
    """
    Run analysis with autonomous technical steps, PI consultation, and step-level checkpointing.

    Args:
        state: Current analysis state
        initial_instruction: The instruction to execute
        max_steps: Maximum number of steps (default 200 to accommodate full analysis + figure review loops)
        figure_review_callback: Optional callback for PI to review generated figures
        resume_from_checkpoint: Whether to attempt resuming from checkpoint (default True)

    Returns:
        Dict with analysis results
    """
    self.refresh_runtime_config_if_needed()
    per_attempt_max_steps = max_steps
    # Use provided callback or fall back to instance callback
    review_callback = figure_review_callback or self.figure_review_callback
    _supervisor_callback = supervisor_callback or self.supervisor_callback

    # Initialize checkpoint manager for step-level recovery
    if hasattr(self.sandbox, 'state_dir') and self.sandbox.state_dir:
        state_dir = str(self.sandbox.state_dir)
    else:
        # Use sandbox working directory as base for states
        state_dir = str(Path(self.sandbox.working_dir) / "states")
        Path(state_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_mgr = CheckpointManager(session_id=state.id, state_dir=state_dir)

    checkpoint, checkpoint_valid = await executor_checkpoint.load_checkpoint(
        self,
        checkpoint_mgr,
        state,
        resume_from_checkpoint=resume_from_checkpoint,
        planning_stage=planning_stage,
        bindings=sys.modules[__name__],
    )


    logger.info(f"[Bioinformatics] Starting autonomous analysis with max_steps={max_steps}")
    await executor_runtime.prepare_sandbox(
        self,
        state,
        bindings=sys.modules[__name__],
    )


    # Build initial prompt — split into static prefix (cacheable) and dynamic context
    self._current_analysis_state = state
    static_system_prompt = self._static_system_prompt
    dynamic_context = self._build_dynamic_context(state)
    dynamic_ctx_msg = {"role": "user", "content": f"{_DYNAMIC_CONTEXT_PREFIX}\n{dynamic_context}"}

    restored = executor_restore.restore_executor_state(
        self,
        state,
        checkpoint=checkpoint,
        checkpoint_valid=checkpoint_valid,
        static_system_prompt=static_system_prompt,
        dynamic_context_message=dynamic_ctx_msg,
        initial_instruction=initial_instruction,
        max_steps=max_steps,
        bindings=sys.modules[__name__],
    )
    messages = restored.messages
    all_findings = restored.findings
    all_figures = restored.figures
    step_count = restored.step_count
    max_steps = restored.max_steps
    analysis_log = restored.analysis_log
    restored_messages_rebuilt = restored.messages_rebuilt


    def _active_plan_id() -> Optional[str]:
        return executor_support.active_plan_id(state)

    # A PI-approved replacement plan starts a new bounded analysis attempt.
    # Keep absolute step numbers monotonic for evidence identity, but reset
    # the per-attempt counter and grant the configured number of fresh steps.
    # The next checkpoint carries the new plan id and attempt offset, so an
    # ordinary retry or reconnect resumes the same budget instead of resetting.
    max_steps, attempt_start_step, attempt_reset = _resume_budget_after_plan_revision(
        checkpoint,
        checkpoint_valid=checkpoint_valid,
        step_count=step_count,
        current_max_steps=max_steps,
        per_attempt_max_steps=per_attempt_max_steps,
        active_plan_id=_active_plan_id(),
    )
    if attempt_reset:
        logger.info(
            "[Bioinformatics] PI-approved plan changed; resetting the "
            "analysis-attempt counter and granting %s fresh steps "
            "(absolute cap=%s)",
            per_attempt_max_steps,
            max_steps,
        )

    async def _save_executor_checkpoint(
        step_name: str,
        *,
        save_sandbox: bool = False,
        budget_exhausted: bool = False,
    ) -> None:
        """Persist the executor transcript independently of planning history."""
        checkpoint_data = executor_checkpoint.build_checkpoint_payload(
            step_name=step_name,
            state=state,
            planning_stage=planning_stage,
            step_number=step_count,
            total_steps=max_steps,
            attempt_start_step=attempt_start_step,
            active_plan_id=_active_plan_id(),
            messages=messages,
            findings=all_findings,
            figures=all_figures,
            code_history=[
                ce.model_dump(mode="json") for ce in self.code_history
            ],
            analysis_log=[
                step.model_dump(mode="json") for step in analysis_log
            ],
            analysis_context=self.analysis_context,
            budget_exhausted=budget_exhausted,
        )
        await checkpoint_mgr.save(checkpoint_data)
        if save_sandbox:
            try:
                await self.sandbox.save_session()
            finally:
                # The record is published either way. A failed save keeps the
                # previous committed generation and does not stop the run.
                self.sandbox.sync_runtime_persistence_state(state)

    # A resumed executor must keep its original tool transcript, but an
    # approved replacement plan has to be made visible before the next LLM
    # call. A plan ID in the checkpoint prevents duplicate injection on
    # ordinary retries. Legacy/rebuilt checkpoints receive the plan once.
    if checkpoint and checkpoint_valid and state.current_plan:
        checkpoint_plan_id = checkpoint.get('active_plan_id')
        if restored_messages_rebuilt:
            checkpoint_plan_id = None
        if inject_revised_plan_context(
            messages,
            state.current_plan,
            checkpoint_plan_id,
        ):
            try:
                await _save_executor_checkpoint(
                    f"step_{step_count}_plan_{state.current_plan.id}",
                )
                logger.info(
                    "[Bioinformatics] Injected approved plan %s into resumed "
                    "executor checkpoint (previous plan=%s)",
                    state.current_plan.id,
                    checkpoint_plan_id,
                )
            except Exception as e:
                logger.warning(
                    "[Bioinformatics] Failed to persist resumed plan context: %s",
                    e,
                )

    annotation_completed = False
    # Restore annotation_completed from analysis_log so the PI observation is
    # accurate after resume. Failed annotation
    # attempts also write an annotate_clusters step (with summary prefixed
    # "[ANNOTATION FAILED]"), so exclude those.
    if analysis_log and any(
        s.step_type == "annotate_clusters"
        and not (s.annotation_result_summary or "").startswith("[ANNOTATION FAILED]")
        for s in analysis_log
    ):
        annotation_completed = True
    mechanism_programs: List[DataDrivenMechanismProgram] = []
    mechanism_programs_registered = bool(analysis_log and any(
        s.step_type == "register_mechanism_programs"
        and not (s.findings_summary or "").startswith("REJECTED")
        for s in analysis_log
    ))
    evidence_gate_findings: List[Dict[str, Any]] = []
    loop_context = executor_context.ExecutorLoopContext(
        agent=self,
        analysis_state=state,
        messages=messages,
        findings=all_findings,
        figures=all_figures,
        analysis_log=analysis_log,
        mechanism_programs=mechanism_programs,
        checkpoint_manager=checkpoint_mgr,
        save_checkpoint=_save_executor_checkpoint,
        review_callback=review_callback,
        supervisor_callback=_supervisor_callback,
        planning_stage=planning_stage,
        max_steps=max_steps,
        attempt_start_step=attempt_start_step,
        step_count=step_count,
        annotation_completed=annotation_completed,
        mechanism_programs_registered=mechanism_programs_registered,
        evidence_gate_findings=evidence_gate_findings,
    )

    return loop_context


async def run_turn_loop(context) -> Dict[str, Any]:
    while (
        context.step_count < context.max_steps
        and not context.completed
    ):
        turn_start = await start_turn(context)
        if turn_start.exit is LoopExit.TERMINAL:
            break
        if turn_start.response is None:
            continue
        response = turn_start.response
        agent_reasoning = response.content[:2000] if response.content else None

        effects = await process_response(context, response, agent_reasoning)
        loop_exit = await finish_turn(context, effects)
        if loop_exit is not LoopExit.CONTINUE:
            break
    return await finalize_run(context)
