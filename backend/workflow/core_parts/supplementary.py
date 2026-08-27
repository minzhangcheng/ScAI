"""SingleCellAnalysisWorkflow supplementary method owners."""
from __future__ import annotations


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


def signal_supplementary_analysis(self, request_text: str, target: str):
    """Queue a supplementary analysis request at the current HITL point.

    Multiple consecutive requests are preserved in arrival order (FIFO) and
    processed one-by-one by the consumer loop in annotation/synthesis nodes.
    """
    self._supplementary_queue.append((request_text, target))
    if target == "annotation":
        self._annotation_event.set()
    elif target in ("direction", "round"):
        self._continue_event.set()

def _merge_supplementary_findings(self, agent_key: str, new_findings: List[str]):
    """Merge supplementary findings into the current graph state for an agent.

    `_initialize_round` seeds every agent-result channel with an explicit
    `None`, and `_current_graph_state` is the bioinformatics node's INPUT dict
    (the round's real result is returned in a new dict), so this key normally
    holds `None` rather than being absent — which means `.get(key, {})` returns
    `None` and the default never applies. The old form guarded only the read of
    `findings` and then assigned into that `None`, raising
    `TypeError: 'NoneType' object does not support item assignment` for every
    supplementary target — annotation, direction and round alike — with the
    caller's try/except turning it into a generic "supplementary analysis
    failed" after the analysis had already been paid for.
    """
    if not self._current_graph_state or not new_findings:
        return
    existing = self._current_graph_state.get(agent_key)
    if not isinstance(existing, dict):
        existing = {}
    existing_findings = existing.get("findings") or []
    existing["findings"] = list(existing_findings) + [
        f"[Supplementary] {f}" for f in new_findings
    ]
    self._current_graph_state[agent_key] = existing

def _absorb_supplementary_executions(self, state: AnalysisState, supp_result: Dict[str, Any]):
    """Record a user-requested supplementary pass's executions on the session.

    Without this the pass runs, spends kernel time, and leaves nothing citable:
    its `code_history` and `analysis_log` were dropped, so no finding it
    produced could be traced to an execution — unlike the round path and unlike
    the automatic review pass, which both merge them. Dedup is by execution
    identity, matching those two.
    """
    try:
        existing_code_ids = {c.id for c in state.code_history}
        for execution in supp_result.get("code_history") or []:
            if getattr(execution, "id", None) is None or execution.id in existing_code_ids:
                continue
            state.code_history.append(execution)
            existing_code_ids.add(execution.id)

        existing_step_ids = {s.id for s in state.analysis_log}
        for step in supp_result.get("analysis_log") or []:
            if isinstance(step, dict):
                step = AnalysisStep(**step)
            if step.id in existing_step_ids:
                continue
            step.round_number = max(int(getattr(state, "current_round", 1) or 1), 1)
            state.analysis_log.append(step)
            existing_step_ids.add(step.id)
    except Exception as e:
        logger.warning(
            f"[{self.session_id}] could not absorb supplementary executions: {e}"
        )

async def _run_supplementary_analysis(
    self, state: AnalysisState, request_text: str, context: str
):
    """Execute user-requested supplementary analysis at a HITL point."""
    await self._send_message(
        "status",
        f"PI is planning supplementary analysis: {request_text[:200]}...",
        AgentType.PI,
    )

    # PI agent plans supplementary tasks
    supp_plan = await self.pi_agent.plan_supplementary_analysis(
        state, request_text, context
    ) or {}

    # Execute bioinformatics tasks if planned
    bio_tasks = supp_plan.get("bioinformatics_tasks", [])
    if bio_tasks and self.bioinformatics_agent:
        await self._send_message("status", "Running supplementary bioinformatics analysis...", AgentType.BIOINFORMATICS)
        try:
            combined_instruction = "Supplementary analysis tasks:\n" + "\n".join(f"- {t}" for t in bio_tasks)
            supp_result = await self.bioinformatics_agent.run_autonomous_analysis(
                state,
                combined_instruction,
                max_steps=30,
                figure_review_callback=self._figure_review_callback,
            )
            if supp_result:
                self._merge_supplementary_findings(
                    "bioinformatics_result", supp_result.get("findings", [])
                )
                self._absorb_supplementary_executions(state, supp_result)
        except Exception as e:
            logger.error(f"[{self.session_id}] Supplementary bioinformatics failed: {e}")
            await self._send_message("error", f"Supplementary bioinformatics analysis failed: {e}", AgentType.BIOINFORMATICS)

    # Execute literature tasks if planned
    lit_tasks = supp_plan.get("literature_topics", [])
    if lit_tasks and self.literature_agent:
        await self._send_message("status", "Running supplementary literature research...", AgentType.LITERATURE)
        try:
            for topic in lit_tasks[:3]:
                lit_result = await self.literature_agent.process(
                    state, topic, topic=topic, use_deep_research=False
                )
                if lit_result:
                    report_text = lit_result.get("report", "") or lit_result.get("synthesis", "")
                    if report_text:
                        self._merge_supplementary_findings(
                            "literature_result", [report_text[:500]]
                        )
        except Exception as e:
            logger.error(f"[{self.session_id}] Supplementary literature failed: {e}")
            await self._send_message("error", f"Supplementary literature research failed: {e}", AgentType.LITERATURE)

    # Generate new detailed report segment
    seg_index = sum(1 for s in state.detailed_report_segments if "supplementary" in s.phase)
    await self._build_and_save_detailed_segment(
        state,
        f"supplementary_{context}_{seg_index}",
        f"Supplementary Analysis: {request_text[:60]}",
        graph_state=self._current_graph_state,
    )

    await self._send_message(
        "status",
        "Supplementary analysis complete. Updating reports...",
        AgentType.PI,
    )

    # Save state
    if self.state_callback:
        try:
            await self.state_callback(state)
        except Exception as e:
            logger.exception(f"[{self.session_id}] Failed to save state after supplementary analysis")
