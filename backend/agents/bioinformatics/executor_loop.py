"""Bounded autonomous tool-turn cycle.

The facade globals are bound at call time so historical monkeypatch targets
continue to affect a resumed executor without creating an import cycle.
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid

from ...constants import NO_OUTPUT_TOKEN_CAP
from ...utils.llm_client_parts.retry import retry_replay_safe_llm_call
from ...utils.llm_client_parts.prompt_cache import (
    PROMPT_CACHE_EPHEMERAL_TAIL_KEY,
    PromptCachePlan,
)
from .executor_auxiliary import LoopExit, TurnEffects, TurnStart, finalize_run


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


def _supervisor_feedback_key(feedback: str) -> str:
    """Deduplicate the complete verdict without discarding changed guidance."""
    return "supervisor:" + hashlib.sha256(
        feedback.encode("utf-8", errors="replace")
    ).hexdigest()


async def _wait_for_pause_gate(context) -> None:
    # Workflow-owned post-execution work may invoke a nested autonomous run.
    # Its tail lock is the pause boundary, so the nested run must finish rather
    # than park while pause() is waiting for that same tail lock.
    if getattr(context.agent, "_pause_gate_deferred", False):
        return
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


async def start_turn(context) -> TurnStart:
    await _wait_for_pause_gate(context)
    context.step_count += 1
    context.force_break = False  # Reset per-iteration
    # Two independent reasons to checkpoint here. The periodic one is the
    # five-step counter; the other is §5.1's runtime cadence, which had no
    # trigger of its own -- `step_count` restarts at 0 on every executor
    # invocation, so `step % 5` almost never came round and the floor governed a
    # branch that never ran. Asking every turn is what makes the floor mean
    # something; `runtime_save_due` returns False until it has elapsed, so this
    # does not save more often than §5.1 says.
    runtime_due = False
    try:
        from backend.services.save_cadence import runtime_save_due

        sandbox = getattr(context.agent, "sandbox", None)
        # `live_dirty` is set by the execution boundary after a cell that could
        # have changed the namespace, and cleared by a commit. Without it the
        # floor's "nothing has committed yet, so save" rule fires on turn one of
        # every run and dumps a kernel that has executed nothing.
        record = getattr(
            getattr(sandbox, "notebook_state", None), "runtime_persistence", None,
        )
        dirty = isinstance(record, dict) and record.get("status") == "live_dirty"
        runtime_due = (
            sandbox is not None
            and dirty
            and runtime_save_due(sandbox, "step_boundary")
        )
    except Exception:
        # The cadence is an optimisation over the round boundary, never a
        # precondition for continuing the analysis.
        logger.debug("[Bioinformatics] runtime cadence check failed", exc_info=True)
    if context.checkpoint_manager.should_checkpoint(context.step_count) or runtime_due:
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
    request_messages = [dict(message) for message in context.messages]
    prompt_cache_plan = None
    figure_followup = (
        getattr(context, "executor_run_kind", "ordinary") == "figure_followup"
    )
    request_tools = FIGURE_FOLLOWUP_TOOLS if figure_followup else TOOLS
    state = getattr(context.agent, "_current_analysis_state", None)
    if state is not None:
        input_anchor = context.agent._build_authoritative_input_anchor(state)
        if input_anchor:
            # Request-only: every turn receives the freshest registry without
            # duplicating the reminder in history or checkpoint transcripts.
            request_messages.append({
                "role": "user",
                "content": input_anchor,
                PROMPT_CACHE_EPHEMERAL_TAIL_KEY: True,
            })
            prompt_cache_plan = PromptCachePlan()
    anchor_count = context.agent._context_anchor_count(request_messages)
    if figure_followup and state is not None:
        from backend.services.figure_followup import figure_followup_source_prompt

        # Request-only protected anchor: survives history trimming, checkpoint
        # resume, and shared context compaction without duplicating transcripts.
        request_messages.insert(anchor_count, {
            "role": "user",
            "content": figure_followup_source_prompt(
                state, context.figure_followup_scope or {},
            ),
            PROMPT_CACHE_EPHEMERAL_TAIL_KEY: True,
        })
        anchor_count += 1
        prompt_cache_plan = PromptCachePlan()

    async def dispatch_tool_request(**kwargs):
        # No tool has run yet. Replay only this request, keeping context
        # budgeting, turn counters, checkpointing and tool execution outside.
        client = context.agent.llm_client

        async def request():
            await _wait_for_pause_gate(context)
            return await client.chat_with_tools(**kwargs)

        async def notify_retry(attempt, maximum, delay, error):
            await context.agent._notify(
                "status",
                f"LLM response incomplete; retrying request "
                f"({attempt}/{maximum}) in {delay:.0f}s...",
            )

        return await retry_replay_safe_llm_call(
            request,
            agent_name="bioinformatics",
            model_name=str(kwargs.get("model") or ""),
            llm_server=str(getattr(client, "base_url", "") or ""),
            on_retry=notify_retry,
        )

    try:
        response = await context.agent._budgeted_text_call(
            dispatch_tool_request,
            messages=request_messages,
            max_tokens=NO_OUTPUT_TOKEN_CAP,
            anchor_count=anchor_count,
            summarize=context.agent._summarize_dropped_messages,
            model=context.agent.model,
            tools=request_tools,
            tool_choice=effective_tool_choice,
            temperature=context.agent.config.temperature,
            prefer_stream=True,
            **(
                {"prompt_cache_plan": prompt_cache_plan}
                if prompt_cache_plan is not None
                else {}
            ),
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
            retry_instruction = (
                "call generate_publication_figure for a scoped render or "
                "complete_figure_followup when every assigned figure task has "
                "been attempted. Do not request new analysis."
                if figure_followup
                else (
                    "call execute_code to run the next analysis step, call another "
                    "appropriate tool, or call report_findings if every assigned "
                    "task is already complete."
                )
            )
            context.messages.append({
                "role": "user",
                "content": (
                    f"[System] Your previous reply was empty — no text and no "
                    f"tool call (attempt {context.empty_response_count}). An empty turn "
                    f"is not a valid step. You MUST respond now with exactly one "
                    f"concrete action: {retry_instruction} Do not return an empty message."
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
            "content": (
                "Continue only with the scoped publication-figure tasks. Call "
                "generate_publication_figure for a task or "
                "complete_figure_followup when each task was attempted. Do not "
                "perform analysis or request new measurements."
                if figure_followup else
                "Please continue the analysis. If you have completed your analysis, "
                "call the report_findings tool with your key findings. "
                "Do NOT respond with text only - you must use a tool."
            )
        })
        return TurnStart()
    return TurnStart(response=response)


async def process_response(context, response, agent_reasoning) -> TurnEffects:
    self = context.agent
    state = context.analysis_state
    messages = context.messages
    figure_followup = (
        getattr(context, "executor_run_kind", "ordinary") == "figure_followup"
    )
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
        executor_run_kind=getattr(context, "executor_run_kind", "ordinary"),
        executor_run_scope_id=getattr(context, "executor_run_scope_id", None),
        executor_settlement_scope_id=getattr(
            context, "executor_settlement_scope_id", None
        ),
        supplementary_request_id=getattr(
            context, "supplementary_request_id", None
        ),
        attempt_execution_ids=getattr(context, "attempt_execution_ids", None)
        if getattr(context, "attempt_execution_ids", None) is not None
        else [],
        pending_dispatch=getattr(context, "pending_dispatch", None),
        resolved_adata_source_context=dict(
            getattr(context, "resolved_adata_source_context", None) or {}
        ),
        figure_followup_scope=getattr(context, "figure_followup_scope", None),
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
            "list_adata_checkpoints": functools.partial(
                handle_list_adata_checkpoints, tool_runtime
            ),
            "generate_publication_figure": functools.partial(
                handle_publication_figure, tool_runtime
            ),
            "complete_figure_followup": functools.partial(
                handle_complete_figure_followup, tool_runtime
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
        allowed_tool_names=(
            FIGURE_FOLLOWUP_TOOL_NAMES if figure_followup else None
        ),
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
    context.pending_dispatch = tool_runtime.pending_dispatch
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
                # The same warning can acquire new redirect or PI guidance at
                # the end. Hash the complete verdict so a shared prefix cannot
                # suppress materially new instructions.
                sup_key = _supervisor_feedback_key(sup_result.feedback)
                if sup_key not in self._recent_sup_feedback:
                    messages.append({
                        "role": "user",
                        "content": f"[Supervisor] {sup_result.feedback}"
                    })
                    self._recent_sup_feedback.add(sup_key)
                else:
                    logger.debug(
                        "[Bioinformatics] Skipping exact duplicate supervisor "
                        "feedback: %s...", sup_key[:24],
                    )
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
                    # two mandatory-skip instructions. Exit the tool-call loop
                    # with a degraded terminal settlement: evidence_gate_degraded
                    # is set below and LoopExit.TERMINAL is returned. No
                    # report_findings call is made.
                    logger.warning(
                        f"[Bioinformatics] Force-skip ignored after {self._force_skip_counter} "
                        f"attempts for {current_err_type}; forcing a degraded terminal "
                        "settlement (report_findings was not invoked)"
                    )
                    # This [System] message is transcript-only: force_break
                    # returns LoopExit.TERMINAL, and run_turn_loop breaks before
                    # the next start_turn() LLM call, so the model never
                    # receives this message. The text below is unchanged and is
                    # not a live instruction.
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
    # Force-break: supervisor force-skip was ignored 3 times. Enter a degraded
    # terminal settlement with no report_findings call. Mark completed so the
    # return path produces a valid status rather than falling through to
    # max_steps_reached with empty findings.
    if force_break:
        logger.info("[Bioinformatics] Force-break triggered by supervisor force-skip")
        # A forced stop is not a clean completion. Reuse the existing
        # degraded-evidence channel so the workflow cannot advance a stage or
        # report the round as fully verified. The termination state is carried
        # by evidence_gate_degraded, the structured supervisor-force-stop record,
        # and the forced_stop result field; do not add it to scientific findings.
        evidence_gate_degraded = True
        evidence_gate_findings = list(evidence_gate_findings or [])
        forced_stop_record = _evidence_gate_record(
            gate="supervisor-force-stop",
            rule_id="scai-native/supervisor-force-stop",
            detail="Analysis was forcibly stopped after repeated execution failures.",
            severity="block",
            forced_release=True,
        )
        forced_stop_key = (
            forced_stop_record.get("gate"),
            forced_stop_record.get("rule_id"),
            forced_stop_record.get("detail"),
        )
        existing_gate_keys = {
            (record.get("gate"), record.get("rule_id"), record.get("detail"))
            for record in evidence_gate_findings
            if isinstance(record, dict)
        }
        if forced_stop_key not in existing_gate_keys:
            evidence_gate_findings.append(forced_stop_record)
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
    figure_followup: bool = False,
    adata_source: Any = None,
    run_scope_id: Optional[str] = None,
    settlement_scope_id: Optional[str] = None,
    run_kind: Optional[str] = None,
    supplementary_request_id: Optional[str] = None,
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

    logger.info(f"[Bioinformatics] Starting autonomous analysis with max_steps={max_steps}")

    # Resolve the durable owner and Host-owned scopes before touching the
    # runtime.  The scoped transcript identity below depends on these values;
    # allocating them after checkpoint lookup both crashed the first run and
    # allowed a transcript to be selected without its owner being known.
    if figure_followup:
        if run_kind not in (None, "figure_followup"):
            raise ValueError("figure_followup contradicts executor run_kind")
        _run_kind = "figure_followup"
    else:
        _run_kind = str(run_kind or "ordinary")
    if _run_kind not in {"ordinary", "supplementary", "figure_followup"}:
        raise ValueError(f"Unsupported executor run_kind: {_run_kind}")
    _source_owner_plan = None
    _request_record = {}
    _owner_state_changed = False
    if _run_kind == "supplementary":
        _request_map = dict(
            (getattr(state, "sandbox_state", None) or {}).get(
                "supplementary_requests_v1"
            )
            or {}
        )
        _request_id = str(supplementary_request_id or "").strip()
        _request_record = dict(_request_map.get(_request_id) or {})
        if (
            not _request_id
            or not _request_record
            or _request_record.get("compacted")
            or _request_record.get("status") in {"settled", "failed"}
        ):
            from backend.services.adata_checkpoints.errors import (
                AdataSourcePreparationError,
            )

            raise AdataSourcePreparationError(
                "Supplementary executor has no active durable request owner",
                reason="supplementary_request_unavailable",
                context={"supplementary_request_id": _request_id or None},
            )
        _durable_run_scope = str(
            _request_record.get("executor_run_scope_id")
            or _request_record.get("request_scope_id")
            or ""
        )
        _durable_settlement_scope = str(
            _request_record.get("executor_settlement_scope_id")
            or _request_record.get("request_scope_id")
            or ""
        )
        if run_scope_id and _durable_run_scope and str(run_scope_id) != _durable_run_scope:
            raise ValueError("supplementary run scope does not match its durable request")
        if (
            settlement_scope_id
            and _durable_settlement_scope
            and str(settlement_scope_id) != _durable_settlement_scope
        ):
            raise ValueError(
                "supplementary settlement scope does not match its durable request"
            )
        _resolved_run_scope = str(
            _durable_run_scope
            or f"supp_run_{uuid.uuid4().hex}"
        )
        _resolved_settlement_scope = str(
            _durable_settlement_scope
            or _resolved_run_scope
        )
        if _request_record.get("executor_run_scope_id") != _resolved_run_scope:
            _request_record["executor_run_scope_id"] = _resolved_run_scope
            _owner_state_changed = True
        if (
            _request_record.get("executor_settlement_scope_id")
            != _resolved_settlement_scope
        ):
            _request_record["executor_settlement_scope_id"] = (
                _resolved_settlement_scope
            )
            _owner_state_changed = True
        _request_map[_request_id] = _request_record
        _scope_state = dict(getattr(state, "sandbox_state", None) or {})
        _scope_state["supplementary_requests_v1"] = _request_map
        state.sandbox_state = _scope_state
    elif _run_kind == "figure_followup":
        _source_owner_plan = getattr(state, "current_plan", None)
        _resolved_run_scope = str(run_scope_id or "").strip()
        if not _resolved_run_scope:
            raise ValueError("Figure follow-up requires a persisted run scope")
        # This identity scopes only a disposable executor checkpoint. It is not
        # an analytical settlement owner and is never added to the main ledger.
        _resolved_settlement_scope = _resolved_run_scope
    else:
        _source_owner_plan = getattr(state, "current_plan", None)
        _owner_scope_key = (
            f"ordinary:{int(getattr(state, 'current_round', 0) or 0)}:"
            f"{planning_stage}:{'figure' if figure_followup else 'run'}"
        )
        _owner_scopes = dict(
            (getattr(state, "sandbox_state", None) or {}).get(
                "executor_owner_scopes_v1"
            )
            or {}
        )
        _stored_scopes = dict(_owner_scopes.get(_owner_scope_key) or {})
        _resolved_run_scope = str(
            (
                run_scope_id
                if _run_kind == "figure_followup"
                else None
            )
            or getattr(_source_owner_plan, "bioinformatics_run_scope_id", None)
            or _stored_scopes.get("run_scope_id")
            or f"run_scope_{uuid.uuid4().hex}"
        )
        _resolved_settlement_scope = str(
            getattr(
                _source_owner_plan,
                "bioinformatics_settlement_scope_id",
                None,
            )
            or _stored_scopes.get("settlement_scope_id")
            or f"settle_scope_{uuid.uuid4().hex}"
        )
        if _source_owner_plan is not None:
            if not getattr(_source_owner_plan, "bioinformatics_run_scope_id", None):
                _source_owner_plan.bioinformatics_run_scope_id = _resolved_run_scope
                _owner_state_changed = True
            if not getattr(
                _source_owner_plan,
                "bioinformatics_settlement_scope_id",
                None,
            ):
                _source_owner_plan.bioinformatics_settlement_scope_id = (
                    _resolved_settlement_scope
                )
                _owner_state_changed = True
        else:
            _new_scope_record = {
                "run_scope_id": _resolved_run_scope,
                "settlement_scope_id": _resolved_settlement_scope,
            }
            if _stored_scopes != _new_scope_record:
                _owner_scopes[_owner_scope_key] = _new_scope_record
                _owner_state_changed = True
            _scope_state = dict(getattr(state, "sandbox_state", None) or {})
            _scope_state["executor_owner_scopes_v1"] = _owner_scopes
            state.sandbox_state = _scope_state
    if (
        _run_kind != "figure_followup"
        and _resolved_settlement_scope not in state.bioinformatics_settlement_attempt_ids
    ):
        state.bioinformatics_settlement_attempt_ids[_resolved_settlement_scope] = []
        _owner_state_changed = True
    if _owner_state_changed:
        await state.increment_version()
        from backend.services.session_store import async_save_session

        await async_save_session(str(state.id), state, force=True)

    if _run_kind != "figure_followup":
        await executor_runtime.prepare_sandbox(
            self,
            state,
            bindings=sys.modules[__name__],
        )

    # Apply or verify the selected AnnData source after the runtime is
    # restored/started and before the executor restores its transcript.  A
    # failure is operation-local and propagates as AdataSourcePreparationError.
    _source_selection = None
    _inherited_preparation_id = None
    if _run_kind == "supplementary":
        _source_selection = _request_record.get("adata_source")
        _inherited_preparation_id = _request_record.get(
            "adata_source_preparation_id"
        )
    else:
        _source_selection = getattr(_source_owner_plan, "adata_source", None)
    from backend.models.schemas import AdataSourceSelection

    if _source_selection is None:
        _source_selection = AdataSourceSelection()
    elif isinstance(_source_selection, dict):
        _source_selection = AdataSourceSelection.model_validate(_source_selection)
    _preparation = None
    _figure_scope = None
    if _run_kind == "figure_followup":
        from backend.services.figure_followup import get_figure_followup_scope

        _figure_scope = get_figure_followup_scope(state, _resolved_run_scope)
        if _figure_scope is None:
            raise ValueError("Figure follow-up has no persisted capability scope")
        _prep_id = str(_figure_scope.get("source_preparation_id") or "")
        _preparation = next(
            (
                record
                for record in (getattr(state, "adata_source_preparations", None) or [])
                if str(getattr(record, "preparation_id", "") or "") == _prep_id
            ),
            None,
        ) if _prep_id else None
    if (
        _run_kind != "figure_followup"
        and _source_selection is not None
        and getattr(_source_selection, "mode", "current") != "current"
    ):
        from backend.services.adata_checkpoints import (
            ensure_adata_source_prepared,
        )

        _preparation = await ensure_adata_source_prepared(
            self.sandbox,
            state,
            _source_selection,
            plan=_source_owner_plan,
            owner_kind=(
                "supplementary" if _run_kind == "supplementary" else "plan"
            ),
            owner_id=(
                supplementary_request_id
                or getattr(_source_owner_plan, "id", None)
            ),
            request_scope_id=_resolved_settlement_scope,
            inherited_preparation_id=_inherited_preparation_id,
        )

    _source_preparation_ids = (
        [str(getattr(_preparation, "preparation_id"))]
        if getattr(_preparation, "preparation_id", None)
        else []
    )
    from backend.services.adata_checkpoints.selection import selection_signature

    _resolved_source_context = {
        "mode": getattr(_source_selection, "mode", "current"),
        "preparation_id": getattr(_preparation, "preparation_id", None),
        "active_source_preparation_id": getattr(
            _preparation, "preparation_id", None
        ),
        "source_preparation_ids": _source_preparation_ids,
        "scope_id": getattr(_preparation, "scope_id", None),
        "active_source_scope_id": getattr(_preparation, "scope_id", None),
        "source_selection_signature": getattr(
            _preparation, "selection_signature", None
        ) or selection_signature(_source_selection),
        "binding_id": getattr(_preparation, "binding_id", None),
        "binding_name": getattr(_preparation, "binding_name", None),
        "binding_identity_status": getattr(
            _preparation, "binding_identity_status", "unknown"
        ),
        "adata_lineage_id": getattr(_preparation, "adata_lineage_id", None),
        "loaded_checkpoint_id": getattr(
            _preparation, "loaded_checkpoint_id", None
        ),
        "ancestor_checkpoint_id": getattr(
            _preparation, "ancestor_checkpoint_id", None
        ),
    }
    if _run_kind != "figure_followup":
        _sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
        _sandbox_state["resolved_adata_source_context_v1"] = dict(
            _resolved_source_context
        )
        state.sandbox_state = _sandbox_state
        await state.increment_version()
        from backend.services.session_store import async_save_session

        await async_save_session(str(state.id), state, force=True)

    # Build the scoped executor checkpoint identity only after runtime and
    # source resolution, so a transcript can never cross run/source/request.
    if hasattr(self.sandbox, 'state_dir') and self.sandbox.state_dir:
        state_dir = str(self.sandbox.state_dir)
    else:
        state_dir = str(Path(self.sandbox.working_dir) / "states")
        Path(state_dir).mkdir(parents=True, exist_ok=True)
    if _preparation is None and _run_kind == "supplementary":
        _prep_id = _request_record.get("adata_source_preparation_id")
    elif _preparation is None:
        _prep_id = getattr(_source_owner_plan, "adata_source_preparation_id", None)
    else:
        _prep_id = getattr(_preparation, "preparation_id", None)
    if _preparation is None:
        for _record in getattr(state, "adata_source_preparations", None) or []:
            if str(getattr(_record, "preparation_id", "")) == str(_prep_id):
                _preparation = _record
                break
    _checkpoint_identity = executor_checkpoint.build_executor_checkpoint_identity(
        state,
        run_kind=_run_kind,
        planning_stage=planning_stage,
        run_scope_id=_resolved_run_scope,
        settlement_scope_id=_resolved_settlement_scope,
        source_selection=_source_selection,
        preparation=_preparation,
        supplementary_request_id=supplementary_request_id,
        figure_followup=figure_followup,
    )
    _scope_digest = _checkpoint_identity.scope_digest
    checkpoint_kwargs = {
        "prefix": executor_checkpoint.executor_checkpoint_prefix(
            run_kind=_run_kind,
            scope_digest=_scope_digest,
            figure_followup=figure_followup,
        )
    }
    checkpoint_mgr = CheckpointManager(
        session_id=state.id, state_dir=state_dir, **checkpoint_kwargs
    )
    if getattr(checkpoint_mgr, "checkpoint_path", None) is not None:
        self.sandbox.active_executor_checkpoint_name = (
            checkpoint_mgr.checkpoint_path.name
        )
    _legacy_allowed = bool(
        getattr(_source_selection, "mode", "current") == "current"
        and not getattr(state, "adata_source_preparations", None)
        and not getattr(state, "adata_live_bindings", None)
        and not getattr(state, "adata_checkpoints", None)
    )
    legacy_checkpoint_mgr = (
        CheckpointManager(
            session_id=state.id,
            state_dir=state_dir,
            prefix=executor_checkpoint.legacy_checkpoint_prefix(),
        )
        if _legacy_allowed
        else None
    )

    checkpoint, checkpoint_valid = await executor_checkpoint.load_checkpoint(
        self,
        checkpoint_mgr,
        state,
        resume_from_checkpoint=resume_from_checkpoint,
        planning_stage=planning_stage,
        bindings=sys.modules[__name__],
        expected_identity=_checkpoint_identity.checkpoint_fields(),
        legacy_checkpoint_manager=legacy_checkpoint_mgr,
        allow_legacy_current=_legacy_allowed,
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
    if figure_followup and not checkpoint_valid:
        # Plotting existing results must still be able to cite their execution
        # receipts, which the ordinary fresh-run path clears from the agent.
        self.code_history = list(state.code_history)


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
        extra_messages: list | None = None,
        completed: bool = False,
    ) -> None:
        """Persist the executor transcript independently of planning history.

        ``extra_messages`` are appended to the transcript this checkpoint
        records without touching the live list. One caller uses it: the
        annotation decision snapshot, which checkpoints the loop while a tool
        call is still outstanding and has to pair that call with a result or
        the restore rebuilds the transcript from nothing.
        """
        checkpoint_data = executor_checkpoint.build_checkpoint_payload(
            step_name=step_name,
            state=state,
            planning_stage=planning_stage,
            step_number=loop_context.step_count,
            total_steps=loop_context.max_steps,
            attempt_start_step=loop_context.attempt_start_step,
            active_plan_id=_active_plan_id(),
            messages=[*loop_context.messages, *(extra_messages or [])],
            findings=loop_context.findings,
            figures=loop_context.figures,
            code_history=[
                ce.model_dump(mode="json") for ce in self.code_history
            ],
            analysis_log=[
                step.model_dump(mode="json") for step in loop_context.analysis_log
            ],
            analysis_context=self.analysis_context,
            budget_exhausted=budget_exhausted,
            runtime_revision=getattr(self.sandbox, "runtime_revision", None),
            checkpoint_scope_digest=_scope_digest,
            checkpoint_run_kind=_run_kind,
            checkpoint_run_scope_id=_resolved_run_scope,
            checkpoint_settlement_scope_id=_resolved_settlement_scope,
            checkpoint_source_scope_id=getattr(_preparation, "scope_id", None),
            checkpoint_preparation_id=getattr(_preparation, "preparation_id", None),
            checkpoint_selection_signature=getattr(
                _preparation, "selection_signature", None
            ),
            checkpoint_token_digest=(
                hashlib.sha256(
                    str(getattr(_preparation, "application_token", "") or "").encode()
                ).hexdigest()
                if _preparation is not None else None
            ),
            checkpoint_request_id=supplementary_request_id,
            checkpoint_figure_followup=figure_followup,
            attempt_execution_ids=list(loop_context.attempt_execution_ids),
            pending_dispatch=(
                dict(loop_context.pending_dispatch)
                if loop_context.pending_dispatch
                else None
            ),
            checkpoint_identity=_checkpoint_identity.checkpoint_fields(),
        )
        checkpoint_data["completed"] = bool(completed)
        await checkpoint_mgr.save(checkpoint_data)
        if save_sandbox:
            from backend.services.save_cadence import runtime_save_due

            # Spec 5.1: a step boundary persists only once the wall-clock floor
            # has passed. A save is minutes of work, and this boundary can fire
            # several times a minute; the round boundary is what bounds the
            # exposure. Skipping is not a failure and writes no record.
            if not runtime_save_due(self.sandbox, "step_boundary"):
                logger.debug(
                    "[Bioinformatics] Step-boundary save skipped: inside the "
                    "runtime save floor",
                )
                return
            try:
                await self.sandbox.save_session()
            finally:
                # The record is published either way. A failed save keeps the
                # previous committed generation and does not stop the run.
                self.sandbox.sync_runtime_persistence_state(state)

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
        executor_run_kind=_run_kind,
        executor_run_scope_id=_resolved_run_scope,
        executor_settlement_scope_id=_resolved_settlement_scope,
        supplementary_request_id=supplementary_request_id,
        figure_followup_scope=_figure_scope,
        attempt_execution_ids=list(
            (checkpoint or {}).get("attempt_execution_ids") or []
        ) if checkpoint_valid else [],
        pending_dispatch=(
            dict((checkpoint or {}).get("pending_dispatch") or {})
        ) if checkpoint_valid else {},
        resolved_adata_source_context=dict(_resolved_source_context),
        adata_source=_source_selection,
    )

    if loop_context.pending_dispatch.get("status") == "pending_dispatch":
        # The process ended after the pre-dispatch checkpoint and before it
        # could publish an objective completion.  Preserve the preallocated ID
        # as an uncertain reached-kernel attempt; never credit it as evidence.
        loop_context.pending_dispatch["status"] = "dispatch_unknown"
        await _save_executor_checkpoint(
            f"step_{loop_context.step_count}_dispatch_unknown"
        )

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
            source_context=dict(
                (getattr(state, "sandbox_state", None) or {}).get(
                    "resolved_adata_source_context_v1"
                )
                or {}
            ),
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

    return loop_context


async def run_turn_loop(context) -> Dict[str, Any]:
    turn_lock = getattr(context.agent, "_pause_turn_lock", None)
    if turn_lock is None:
        turn_lock = asyncio.Lock()
        context.agent._pause_turn_lock = turn_lock
    while (
        context.step_count < context.max_steps
        and not context.completed
    ):
        # Wait outside the turn lock so pause() can acquire it and persist the
        # quiescent runtime. Recheck after acquisition to close the race where
        # pause is requested between this wait and the lock hand-off.
        await _wait_for_pause_gate(context)
        async with turn_lock:
            pause_event = context.agent._pause_event
            if pause_event is not None and not pause_event.is_set():
                continue
            context.agent._pause_turn_active = True
            try:
                turn_start = await start_turn(context)
                if turn_start.exit is LoopExit.TERMINAL:
                    return await finalize_run(context)
                if turn_start.response is None:
                    continue
                response = turn_start.response
                agent_reasoning = response.content[:2000] if response.content else None

                effects = await process_response(context, response, agent_reasoning)
                loop_exit = await finish_turn(context, effects)
                if loop_exit is not LoopExit.CONTINUE:
                    return await finalize_run(context)
                if context.completed or context.step_count >= context.max_steps:
                    return await finalize_run(context)
            finally:
                context.agent._pause_turn_active = False
    return await finalize_run(context)
