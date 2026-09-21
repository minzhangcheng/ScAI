"""Bioinformatics-PI planning dialogue phases."""
from __future__ import annotations

import uuid
from ..helpers import WorkflowTerminatedError


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


async def prepare_planning_dialogue(self, state):
    analysis_state = state["analysis_state"]

    # Planning calls use explicit, stateless prompts. Clear legacy BaseAgent
    # history here so in-memory sessions created before context isolation do
    # not retain multi-round planning/annotation transcripts indefinitely.
    reset_bioinfo_history = getattr(self.bioinformatics_agent, "reset_history", None)
    if callable(reset_bioinfo_history):
        reset_bioinfo_history()

    for agent_name, node in [("pi", "planning_dialogue"), ("bioinformatics", "planning_dialogue")]:
        skip = await self._require_core_agent(state, agent_name, node)
        if skip is not None:
            return skip

    pi_instructions = state.get("pi_instructions", {})
    bioinf_tasks = pi_instructions.get("bioinformatics_tasks", [])
    overview = pi_instructions.get("overview", "")
    stage = state.get("planning_stage", "full")  # "full", "stage1", "stage2"
    existing_plan = analysis_state.current_plan  # may have a partial plan from revision
    if existing_plan and existing_plan.stage and existing_plan.stage != stage and stage != "full":
        existing_plan = None

    phase = analysis_state.analysis_phase.value
    combined_previous = self._build_planning_context(state)
    execution_revision_feedback = ""
    if state.get("plan_revision_needed"):
        revision_context = state.get("plan_revision_context") or {}
        revision_text = json.dumps(
            revision_context,
            ensure_ascii=False,
            default=str,
        )[:4000]
        results_text = json.dumps(
            combined_previous or {},
            ensure_ascii=False,
            default=str,
        )[:4000]
        execution_revision_feedback = (
            "Execution-triggered revision context:\n"
            f"{revision_text or '(no explicit trigger details)'}\n\n"
            "Bounded summary of results available at the pause:\n"
            f"{results_text or '(no result summary available)'}"
        )
    dialogue_history = []
    finalization_reason: Optional[str] = None

    round_num = 1
    return {
        "_continue": True, "analysis_state": analysis_state, "stage": stage,
        "phase": phase, "bioinf_tasks": bioinf_tasks, "overview": overview,
        "combined_previous": combined_previous,
        "execution_revision_feedback": execution_revision_feedback,
        "dialogue_history": dialogue_history, "existing_plan": existing_plan,
        "round_num": round_num, "finalization_reason": finalization_reason,
    }

async def run_planning_dialogue_rounds(self, ctx, state):
    from ...agents.bioinformatics.planning import PlanGenerationFailure

    analysis_state = ctx["analysis_state"]
    stage = ctx["stage"]
    phase = ctx["phase"]
    bioinf_tasks = ctx["bioinf_tasks"]
    overview = ctx["overview"]
    combined_previous = ctx["combined_previous"]
    execution_revision_feedback = ctx["execution_revision_feedback"]
    dialogue_history = ctx["dialogue_history"]
    existing_plan = ctx["existing_plan"]
    round_num = ctx["round_num"]
    finalization_reason = ctx["finalization_reason"]
    while round_num <= MAX_PLANNING_ROUNDS:
        await self._send_message(
            "status",
            f"Planning dialogue: round {round_num}/{MAX_PLANNING_ROUNDS}",
            AgentType.PI,
        )

        # Keep the exact input revision so convergence is based on the two
        # adjacent plans, not on volatile AnalysisState references.
        previous_plan = existing_plan.model_copy(deep=True) if existing_plan else None

        # --- Bioinformatics agent call (with LLM failure intervention) ---
        from ..helpers import reset_unattended_gate_budget

        reset_unattended_gate_budget(self, "planning_dialogue")
        while True:
            try:
                if round_num == 1 and not existing_plan:
                    plan = await self.bioinformatics_agent.generate_analysis_plan(
                        state=analysis_state,
                        pi_tasks=bioinf_tasks,
                        overview=overview,
                        phase=phase,
                        planning_stage=stage,
                        previous_results=combined_previous,
                    )
                elif existing_plan and existing_plan.status == "rejected":
                    plan = await self.bioinformatics_agent.generate_analysis_plan(
                        state=analysis_state,
                        pi_tasks=bioinf_tasks,
                        overview=overview,
                        phase=phase,
                        planning_stage=stage,
                        previous_results=combined_previous,
                        existing_plan=existing_plan,
                    )
                else:
                    # Collect PI feedback from all dialogue rounds, not just the last one
                    all_pi_feedback = truncate_for_llm("\n".join(
                        f"Round {d['round']}: {d['pi_feedback']}"
                        for d in dialogue_history
                        if d.get("pi_feedback")
                    ), max_chars=12_000)
                    # P2-6: read user_plan_feedback WITHOUT clearing it.
                    # If the revise call below fails, the feedback must
                    # survive so the user doesn't have to re-type it.
                    user_feedback = truncate_for_llm(
                        state.get("user_plan_feedback") or "",
                        max_chars=4_000,
                    )
                    pi_feedback = all_pi_feedback or (dialogue_history[-1]["pi_feedback"] if dialogue_history else "")
                    feedback_parts = [part for part in (
                        execution_revision_feedback,
                        pi_feedback,
                        f"User feedback: {user_feedback}" if user_feedback else "",
                    ) if part]
                    combined_feedback = "\n\n".join(feedback_parts)
                    # Feedback accumulates every round; the plan is re-derived
                    # every round. Without the tasks and the step-level asks
                    # travelling alongside it, the caution compounds while the
                    # content it was written about decays out of the plan.
                    all_pi_modifications = [
                        {"round": d.get("round"), "modifications": d.get("pi_modifications") or []}
                        for d in dialogue_history
                        if d.get("pi_modifications")
                    ]
                    plan = await self.bioinformatics_agent.revise_plan_incrementally(
                        state=analysis_state,
                        current_plan=existing_plan,
                        pi_feedback=combined_feedback,
                        pi_tasks=bioinf_tasks,
                        overview=overview,
                        pi_modifications=all_pi_modifications,
                    )
                    # Clear only after a successful revise call.
                    if user_feedback:
                        state["user_plan_feedback"] = None
                populate_plan_scope(plan, analysis_state)
                break  # success, exit inner retry loop
            except (UpstreamExhaustedError, PlanGenerationFailure) as e:
                error = e if isinstance(e, UpstreamExhaustedError) else UpstreamExhaustedError(
                    str(e), agent_name="bioinformatics", model=self.bioinformatics_agent.model,
                    failure_kind="parse_error",
                )
                action = await self._wait_for_llm_intervention(
                    error, "planning_dialogue", analysis_state
                )
                if action == LlmFailureAction.RETRY:
                    continue  # retry same bioinfo call
                if action == LlmFailureAction.TERMINATE:
                    raise WorkflowTerminatedError("User terminated planning after LLM failure") from e
                # SKIP
                await self._send_message(
                    "error",
                    "Planning dialogue: bioinformatics agent skipped due to LLM failure.",
                    AgentType.PI,
                )
                if existing_plan:
                    # Don't reinstate a PI-rejected plan as pi_reviewing
                    if existing_plan.status != "rejected":
                        existing_plan.status = "pi_reviewing"
                    analysis_state.current_plan = existing_plan
                    preserved_history = analysis_state.plan_dialogue_history or dialogue_history
                    analysis_state.plan_dialogue_history = preserved_history
                    state["planning_result"] = {
                        "plan": existing_plan.model_dump(mode="json"),
                        "dialogue": preserved_history,
                    }
                else:
                    state["should_continue"] = False
                    analysis_state.status = TaskStatus.FAILED
                    await self._send_state_sync(analysis_state, "planning_aborted")
                    await self._send_message(
                        "error",
                        "Planning aborted: bioinformatics LLM unavailable. Workflow terminating.",
                        AgentType.PI,
                    )
                    state["failed_node"] = "planning_dialogue"
                    state["planning_result"] = {
                        "error": str(e),
                        "findings": [],
                        "status": "failed",
                    }
                return state
            # Bioinfo plan generation: no retry on generic Exception.
            # Unlike PI review below (which retries), bioinfo failures
            # are typically deterministic — retrying won't help.
            except WorkflowTerminatedError:
                raise
            except Exception as e:
                logger.exception(
                    f"[{self.session_id}] Bioinformatics plan generation failed "
                    f"in planning dialogue round {round_num}"
                )
                await self._send_message(
                    "error",
                    f"Bioinformatics plan generation failed: {str(e)}",
                    AgentType.PI,
                )
                state["failed_node"] = "planning_dialogue"
                state["planning_result"] = {
                    "error": str(e),
                    "findings": [],
                    "status": "failed",
                }
                state["should_continue"] = False
                analysis_state.status = TaskStatus.FAILED
                await self._send_state_sync(analysis_state, "planning_aborted")
                return state

        # --- PI review call (with LLM failure intervention) ---
        from ..helpers import reset_unattended_gate_budget

        reset_unattended_gate_budget(self, "planning_dialogue")
        pi_review_attempt = 0
        while True:
            try:
                review = await self.pi_agent.review_bioinfo_plan(
                    state=analysis_state,
                    plan=plan,
                    dialogue_round=round_num,
                )
                break  # success, exit inner retry loop
            except UpstreamExhaustedError as e:
                action = await self._wait_for_llm_intervention(
                    e, "planning_dialogue", analysis_state
                )
                if action == LlmFailureAction.RETRY:
                    continue  # retry same PI review call
                if action == LlmFailureAction.TERMINATE:
                    raise WorkflowTerminatedError("User terminated PI plan review") from e
                # A plan cannot be accepted when the PI review did not complete.
                await self._send_message(
                    "error",
                    "Planning dialogue: PI review did not complete; the plan was not approved.",
                    AgentType.PI,
                )
                state["failed_node"] = "planning_dialogue"
                state["planning_result"] = {
                    "error": str(e),
                    "findings": [],
                    "status": "failed",
                }
                state["should_continue"] = False
                analysis_state.status = TaskStatus.FAILED
                await self._send_state_sync(analysis_state, "planning_aborted")
                return state
            except WorkflowTerminatedError:
                raise
            except Exception as e:
                pi_review_attempt += 1
                if pi_review_attempt <= MAX_NODE_RETRIES:
                    delay = _retry_backoff_delay(pi_review_attempt)
                    logger.warning(
                        f"[{self.session_id}] PI review failed in round {round_num} "
                        f"(attempt {pi_review_attempt}/{MAX_NODE_RETRIES + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await self._send_message(
                        "status",
                        f"PI review failed, retrying ({pi_review_attempt}/{MAX_NODE_RETRIES + 1})...",
                        AgentType.PI,
                    )
                    await asyncio.sleep(delay)
                    continue
                # All retries exhausted — return failed (matches @with_retry)
                logger.exception(
                    f"[{self.session_id}] PI review failed after "
                    f"{MAX_NODE_RETRIES + 1} attempts in round {round_num}"
                )
                await self._send_message(
                    "error",
                    f"PI review failed after {MAX_NODE_RETRIES + 1} attempts: {str(e)}",
                    AgentType.PI,
                )
                state["failed_node"] = "planning_dialogue"
                state["planning_result"] = {
                    "error": str(e),
                    "findings": [],
                    "status": "failed",
                }
                state["should_continue"] = False
                analysis_state.status = TaskStatus.FAILED
                await self._send_state_sync(analysis_state, "planning_aborted")
                return state

        # --- Process PI review decision ---
        _review_source = review.get("adata_source")
        if _review_source is not None:
            try:
                if (
                    _review_source.model_dump(mode="json")
                    != plan.adata_source.model_dump(mode="json")
                ):
                    # B1: a source switch owns a fresh application and a
                    # fresh transcript run scope.  The inherited settlement
                    # scope stays because earlier reached-kernel attempts must
                    # remain in boundary identity.
                    plan.adata_source = _review_source.model_copy(deep=True)
                    plan.adata_source_preparation_id = None
                    plan.bioinformatics_run_scope_id = (
                        f"run_scope_{uuid.uuid4().hex}"
                    )
            except AttributeError:
                pass
        dialogue_history.append({
            "round": round_num,
            "plan": plan.model_dump(mode="json"),
            "pi_decision": review["decision"],
            "pi_feedback": review["feedback"],
            "pi_modifications": review.get("modifications", []),
            "materially_unchanged": plans_materially_equivalent(previous_plan, plan),
        })

        if review["decision"] == "approved":
            plan.status = "pi_reviewing"
            analysis_state.current_plan = plan
            analysis_state.plan_dialogue_history = dialogue_history
            state["planning_result"] = {"plan": dialogue_history[-1]["plan"], "dialogue": dialogue_history}
            await self._send_message("status", "Plan approved by PI", AgentType.PI)
            break
        elif review["decision"] == "reject":
            plan.status = "rejected"
            existing_plan = plan
            await self._send_message(
                "status",
                f"Plan rejected by PI, redesigning... ({review['feedback'][:200]})",
                AgentType.PI,
            )
        else:  # revise
            existing_plan = plan
            existing_plan.status = "draft"
            await self._send_message(
                "status",
                f"Plan needs revision (round {round_num}): {review['feedback'][:200]}",
                AgentType.PI,
            )

        if (
            review["decision"] != "approved"
            and dialogue_history[-1]["materially_unchanged"]
        ):
            finalization_reason = "no_material_plan_change"
            logger.warning(
                "[%s] Planning dialogue converged without approval at round %s; "
                "routing to unilateral PI finalization",
                self.session_id,
                round_num,
            )
            break

        round_num += 1

    ctx.update(
        dialogue_history=dialogue_history, existing_plan=existing_plan,
        round_num=round_num, finalization_reason=finalization_reason,
    )
    return ctx

async def finalize_planning_dialogue(self, ctx, state):
    analysis_state = ctx["analysis_state"]
    stage = ctx["stage"]
    dialogue_history = ctx["dialogue_history"]
    existing_plan = ctx["existing_plan"]
    round_num = ctx["round_num"]
    finalization_reason = ctx["finalization_reason"]
    # No agreement at the hard limit, or an unchanged revision proves the
    # dialogue has converged without approval. In both cases PI must actually
    # revise and commit the final plan; never relabel a draft as approved.
    if finalization_reason or round_num > MAX_PLANNING_ROUNDS:
        if finalization_reason is None:
            finalization_reason = "max_rounds"
            logger.warning(
                f"[{self.session_id}] Planning dialogue reached max rounds "
                f"({MAX_PLANNING_ROUNDS})"
            )

        seed_plan = (
            existing_plan
            if existing_plan is not None and existing_plan.steps
            else self._select_fallback_plan_from_history(dialogue_history)
        )
        if seed_plan is not None:
            analysis_state.current_plan = seed_plan
        await self._send_message(
            "status",
            "Planning dialogue converged without approval; PI is finalizing "
            "the plan unilaterally...",
            AgentType.PI,
        )
        try:
            from ...agents.bioinformatics.planning import PlanGenerationFailure

            while True:
                try:
                    final_plan = await self.pi_agent.generate_final_plan_from_dialogue(
                        analysis_state,
                        dialogue_history,
                        current_plan=seed_plan,
                    )
                    break
                except (UpstreamExhaustedError, PlanGenerationFailure) as exc:
                    error = exc if isinstance(exc, UpstreamExhaustedError) else UpstreamExhaustedError(
                        str(exc), agent_name="pi", model=self.pi_agent.model,
                        failure_kind="parse_error",
                    )
                    action = await self._wait_for_llm_intervention(
                        error, "planning_dialogue", analysis_state,
                    )
                    if action == LlmFailureAction.RETRY:
                        continue
                    if action == LlmFailureAction.TERMINATE:
                        raise WorkflowTerminatedError("User terminated PI plan finalization") from exc
                    # Skipping an unavailable finalizer cannot approve its draft.
                    state["plan_approved"] = False
                    raise exc
            from ...services.plan_scope import validate_innovation_plan_scope

            validate_innovation_plan_scope(analysis_state, final_plan)
            populate_plan_scope(final_plan, analysis_state)
            analysis_state.current_plan = final_plan
            analysis_state.plan_dialogue_history = dialogue_history
            state["plan_approved"] = True
            state["planning_result"] = {
                "plan": final_plan.model_dump(mode="json"),
                "dialogue": dialogue_history,
                "finalization_reason": finalization_reason,
            }
            await self._send_message(
                "status",
                "PI has finalized the plan unilaterally.",
                AgentType.PI,
            )
        except WorkflowTerminatedError:
            raise
        except Exception as e:
            logger.exception(
                f"[{self.session_id}] PI failed to generate final plan "
                f"from dialogue: {e}"
            )
            await self._send_message(
                "error",
                f"PI failed to finalize plan after dialogue convergence: {str(e)}",
                AgentType.PI,
            )
            state["failed_node"] = "planning_dialogue"
            state["planning_result"] = {
                "error": str(e),
                "findings": [],
                "status": "failed",
            }
            state["should_continue"] = False
            analysis_state.status = TaskStatus.FAILED
            await self._send_state_sync(analysis_state, "planning_aborted")

    await self._report_direction_coverage_gaps(analysis_state)
    return state


async def run_planning_dialogue(self, state):
    ctx = await prepare_planning_dialogue(self, state)
    if "_continue" not in ctx:
        return ctx
    ctx = await run_planning_dialogue_rounds(self, ctx, state)
    if "_continue" not in ctx:
        return ctx
    return await finalize_planning_dialogue(self, ctx, state)
