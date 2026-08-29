"""Plan approval, review gate, and dialogue presentation helpers."""
from __future__ import annotations


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )

async def _report_direction_coverage_gaps(self, analysis_state) -> None:
    """Surface direction requirements the settled plan does not address.

    Direction coverage elsewhere answers "did a round run for this
    direction". It cannot tell the user that the plan about to execute
    dropped the questions the direction was selected to resolve — which is
    what a long revision dialogue tends to produce. Reported at the plan
    review gate so the user can add them back before execution starts.
    """
    plan = analysis_state.current_plan
    if plan is None or not getattr(plan, "steps", None):
        return
    try:
        gaps = uncovered_direction_requirements(analysis_state, plan)
    except Exception:
        logger.warning(
            "[%s] Direction coverage check failed", self.session_id, exc_info=True
        )
        return
    if not gaps:
        return
    logger.warning(
        "[%s] Settled plan does not address %d direction requirement(s)",
        self.session_id, len(gaps),
    )
    preview = "\n".join(f"- {item}" for item in gaps[:8])
    await self._send_message(
        "status",
        "Plan review: the following questions this direction was selected to "
        f"resolve are not addressed by the plan ({len(gaps)} total):\n{preview}",
        AgentType.PI,
    )

async def _approve_plan(
    self, state: WorkflowState, plan, analysis_state, action: str
) -> None:
    """Mark plan as approved and notify frontend."""
    populate_plan_scope(plan, analysis_state)
    plan.status = "approved"
    plan.approved_at = datetime.now()
    # Preserve reviewed figures for completed steps carried unchanged into a
    # revision. The lineage lives beside runtime state; artifact.plan_id keeps
    # naming the plan that actually rendered and reviewed the bytes.
    record_plan_figure_inheritance(analysis_state, plan)
    analysis_state.current_plan = plan
    self._set_status(analysis_state, TaskStatus.RUNNING)
    await self._send_state_sync(analysis_state, "plan_approved")
    await self._send_message("plan_approved", {
        "plan_id": plan.id,
        "action": action,
    }, AgentType.PI)
    state["plan_approved"] = True
    if state.get("plan_revision_needed"):
        state["plan_revision_resume"] = True
        state["plan_revision_needed"] = False

async def _user_review_gate(self, state: WorkflowState) -> WorkflowState:
    """Present the analysis plan to the user for review and approval.

    Blocks until the user responds (approve/skip/modify).
    For modification requests, loops back to planning_dialogue.
    """
    await ack_restore_safe_point(self)
    analysis_state = state["analysis_state"]
    plan = analysis_state.current_plan

    if not plan or not plan.steps:
        logger.error(
            f"[{self.session_id}] No plan (or empty plan with {len(plan.steps) if plan else 0} "
            f"steps) to review; planning aborted, routing to final report"
        )
        await self._send_message(
            "error",
            "Planning failed: no valid plan is available.",
            AgentType.PI,
        )
        state["failed_node"] = "planning_dialogue"
        state["planning_result"] = {
            "error": "No valid analysis plan is available",
            "findings": [],
            "status": "failed",
        }
        state["should_continue"] = False
        analysis_state.status = TaskStatus.FAILED
        await self._send_state_sync(analysis_state, "planning_aborted")
        return state

    # Wipe any action/event left behind by a previous gate invocation in the
    # same workflow (e.g. stage1's approve still sitting in the slot when
    # stage2's gate is entered). Without this, the `prev_action` capture
    # below would consume the stale value and skip waiting for the user.
    self._plan_review_action = None
    self._plan_review_event.clear()

    # Main review loop — continues until user approves or skips.
    # Generate dialogue HTML once — it doesn't change during the review loop.
    dialogue_html = self._build_plan_dialogue_html(
        analysis_state.plan_dialogue_history
    )
    while True:
        # Set plan status and notify frontend
        plan.status = "user_reviewing"

        # Send plan to frontend
        wait_seconds = resolve_plan_review_wait(analysis_state)
        await self._send_message("plan_review", {
            "plan": plan.model_dump(mode="json"),
            "dialogue_html": dialogue_html,
            "phase": analysis_state.analysis_phase.value,
            "round": analysis_state.current_round,
            "stage": state.get("planning_stage", "full"),
            "wait_seconds": wait_seconds,
        }, AgentType.PI)

        # Set waiting status and sync
        self._set_status(analysis_state, TaskStatus.WAITING_PLAN_REVIEW)
        await self._send_state_sync(analysis_state, "plan_review")

        # Block and wait for user (with 1-hour timeout to prevent permanent hang)
        # Save and reset action before clearing event to close race window
        # where signal_plan_review arrives between clarify continue and event.clear()
        #
        # TOCTOU note: The clear() → None → wait() sequence (lines below) is safe
        # in asyncio because lines 589-591 are synchronous (no `await`), so no
        # other coroutine can interleave between event.clear() and event.wait().
        # The only yield point is the `await` in wait_for() on the line after.
        prev_action = self._plan_review_action
        self._plan_review_event.clear()
        self._plan_review_action = None
        if not prev_action:
            wait_seconds = resolve_plan_review_wait(analysis_state)
            signaled = await wait_with_skip_time(
                self._plan_review_event,
                wait_seconds,
                self.session_id,
                "plan_review",
                safe_point_hook=getattr(self, "await_restore_safe_point", None),
            )
            if not signaled:
                # Auto-skip on timeout regardless of auto_approve_plans,
                # since the user chose a finite wait via round_wait_seconds
                # or plan_review_wait_seconds — same semantics as round-wait.
                self._plan_review_action = {"action": "skip", "user_input": ""}
            action = self._plan_review_action or {}
        else:
            logger.info(f"[{self.session_id}] Plan review action already received, skipping wait")
            action = prev_action

        # Consume: clear the slot + event so the next loop iteration (clarify
        # path) or the next gate invocation starts from a clean state.
        self._plan_review_action = None
        self._plan_review_event.clear()

        action_type = action.get("action", "skip")
        user_input = action.get("user_input", "")

        if action_type == "approve":
            await self._approve_plan(state, plan, analysis_state, "approved")
            break

        elif action_type == "skip":
            await self._approve_plan(state, plan, analysis_state, "skipped")
            break

        elif action_type == "modify" and user_input:
            # User wants modifications → PI judges intent
            pi_judgment = await self.pi_agent.process_user_plan_feedback(
                user_input=user_input,
                current_plan=plan,
            )

            if pi_judgment["action"] == "proceed_to_execution":
                await self._approve_plan(state, plan, analysis_state, "approved_after_clarification")
                break

            elif pi_judgment["action"] == "clarify_with_user":
                # PI needs to clarify — notify user and wait again
                await self._send_message("status",
                    pi_judgment.get("clarification_question", "Could you clarify?"),
                    AgentType.PI)
                continue  # Re-show the review gate

            else:  # revise_with_bioinfo
                # PI will incorporate user feedback into plan revision
                guidance = pi_judgment.get("guidance_for_bioinfo", user_input)
                state["user_plan_feedback"] = guidance
                state["plan_approved"] = False
                if state.get("plan_revision_needed"):
                    state["plan_revision_resume"] = True
                    state["plan_revision_needed"] = False
                break

    return state

def _build_plan_dialogue_html(dialogue_history: List[Dict]) -> str:
    """Build an HTML document from the PI↔Bioinfo planning dialogue.

    Includes step-level diff between consecutive rounds to highlight changes.
    """
    html_parts = [
        "<html><head><meta charset='utf-8'><title>Analysis Plan Dialogue</title>",
        "<style>body{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px;}",
        ".round{border:1px solid #ccc;margin:10px 0;padding:15px;border-radius:8px;}",
        ".approved{background:#e8f5e9}.revise{background:#fff3e0}.reject{background:#ffebee}",
        ".step{margin:5px 0;padding:8px;border-radius:4px;}",
        ".step-added{background:#e8f5e9;border-left:4px solid #4caf50}",
        ".step-removed{background:#ffebee;border-left:4px solid #f44336;text-decoration:line-through}",
        ".step-modified{background:#fff3e0;border-left:4px solid #ff9800}",
        ".step-unchanged{background:#f5f5f5}",
        "</style></head><body><h1>Analysis Plan Dialogue</h1>",
    ]

    prev_steps = None
    for entry in dialogue_history:
        round_num = entry["round"]
        decision = _html_escape(entry.get("pi_decision", ""))
        css_class = {"approved": "approved", "revise": "revise", "reject": "reject"}.get(decision, "")

        html_parts.append(f"<div class='round {css_class}'>")
        html_parts.append(f"<h2>Round {round_num} — PI Decision: {decision.upper()}</h2>")
        html_parts.append(f"<p><strong>PI Feedback:</strong> {_html_escape(entry.get('pi_feedback', ''))}</p>")

        plan = entry.get("plan", {})
        curr_steps = plan.get("steps", [])
        step_diffs = _diff_plan_steps(prev_steps, curr_steps) if prev_steps else {}

        if prev_steps is not None and curr_steps:
            has_changes = any(d != 'unchanged' for d in step_diffs.values())
            if has_changes:
                html_parts.append("<h3>Changes from Previous Round:</h3>")
                changes = []
                for sid, status in step_diffs.items():
                    if status == 'added':
                        step_data = next((s for s in curr_steps if s.get('id') == sid), {})
                        changes.append(f"<span style='color:#4caf50'>+ Added Step {sid}: {_html_escape(step_data.get('title',''))}</span>")
                    elif status == 'removed':
                        step_data = next((s for s in prev_steps if s.get('id') == sid), {})
                        changes.append(f"<span style='color:#f44336'>- Removed Step {sid}: {_html_escape(step_data.get('title',''))}</span>")
                    elif status == 'modified':
                        changes.append(f"<span style='color:#ff9800'>~ Modified Step {sid}</span>")
                html_parts.append("<br>".join(changes))

        if curr_steps:
            html_parts.append("<h3>Plan Steps:</h3>")
            for step in curr_steps:
                sid = step.get('id', '')
                diff_status = step_diffs.get(sid, 'unchanged')
                css = f"step-{diff_status}"
                label = _DIFF_LABELS.get(diff_status, '')
                html_parts.append(
                    f"<div class='step {css}'><strong>Step {_html_escape(str(sid))}: {_html_escape(step.get('title', ''))}</strong> {label}<br>"
                    f"Method: {_html_escape(step.get('method', ''))}<br>"
                    f"Rationale: {_html_escape(step.get('rationale', ''))}</div>"
                )

        html_parts.append("</div>")
        prev_steps = curr_steps

    html_parts.append("</body></html>")
    return "\n".join(html_parts)
