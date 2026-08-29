"""Synthesis workflow facade retaining graph-facing mixin methods."""
import sys
from .synthesis_parts import phases as synthesis_phases
from .synthesis_parts import support as _support
globals().update(
    {name: value for name, value in vars(_support).items() if not name.startswith("__")}
)

class SynthesisMixin:
    """Mixin providing synthesis, user-input, and convergence methods. Composed into SingleCellAnalysisWorkflow."""

    _LIT_AUDIT_CHAR_CAP = 32000  # ~8K tokens

    def _build_literature_audit_context(self, state) -> str:
        """Bounded literature context for the cross-modal audit. Priority:
        (1) persisted mechanism_signals, (2) reference titles/abstracts,
        (3) report body excerpt. Hard-capped.

        IMPORTANT (E2): references and abstracts are read from the persisted
        `analysis_state.literature_reports` objects, NOT from
        `state["literature_result"]`. The combined `literature_result` dict built by
        `_literature_research` holds only topics/findings/reports/search_modes — it has
        NO "references" key (verified). The per-topic references are persisted on each
        LiteratureReport, which also survive snapshot restore / session reload.
        """
        analysis_state = state["analysis_state"]
        parts = []

        # (1) mechanism_signals from persisted LiteratureReports (last 8)
        for rep in analysis_state.literature_reports[-8:]:
            if rep.mechanism_signals:
                for sig in rep.mechanism_signals:
                    parts.append(
                        f"- Mechanism: {sig.mechanism} "
                        f"(PMIDs: {', '.join(sig.supporting_pmids) or 'N/A'}; "
                        f"type: {sig.evidence_type}); relevance: {sig.relevance_to_current_study}; "
                        f"links: {', '.join(sig.observed_data_links) or 'N/A'}; "
                        f"limitations: {', '.join(sig.limitations) or 'N/A'}"
                    )

        # (2) reference titles/abstracts from persisted LiteratureReports.
        # Abstracts are NOT guaranteed (PubMed mode may drop them; Deep Research has none),
        # so degrade gracefully. Bound to the most recent reports to approximate the
        # current round; the hard char cap below is the real bound.
        ref_count = 0
        for rep in analysis_state.literature_reports[-4:]:
            for ref in (rep.references or []):
                if not isinstance(ref, dict):
                    continue
                title = ref.get("title", "")
                abstract = ref.get("abstract", "")  # may be absent — degrade gracefully
                pmid = ref.get("pmid", "")
                line = f"- {title}"
                if pmid:
                    line += f" (PMID: {pmid})"
                if abstract:
                    line += f": {abstract[:400]}"
                parts.append(line)
                ref_count += 1
                if ref_count >= 30:
                    break
            if ref_count >= 30:
                break

        # (3) current round's report body excerpts (this key DOES exist on literature_result)
        lit_result = state.get("literature_result") or {}
        for report_body in (lit_result.get("reports") or [])[:3]:
            if report_body:
                parts.append(f"Report excerpt: {report_body[:2000]}")

        text = "\n".join(parts)
        return text[:self._LIT_AUDIT_CHAR_CAP]

    def _build_biochemistry_raw_context(self, state) -> str:
        """Raw biochemistry measurements + manual indicators for the current round."""
        analysis_state = state["analysis_state"]
        parts = []
        bio_result = state.get("biochemistry_result") or {}
        findings = bio_result.get("findings") or []
        if findings:
            parts.append("Findings: " + "; ".join(str(f) for f in findings))
        if analysis_state.biochemistry_indicators:
            for ind in analysis_state.biochemistry_indicators[:50]:
                parts.append(
                    f"- {ind.indicator}: {ind.value}{ind.unit or ''} "
                    f"(ref: {ind.reference_range or 'N/A'}; {ind.interpretation or ''})"
                )
        return ("\n".join(parts))[:16000]

    def _exploration_round_succeeded(self, state) -> bool:
        """P1-12: whitelist — exploration succeeded when bioinformatics settled
        cleanly, i.e. reported status='completed' or 'partial'. All other
        statuses (error, failed, timeout, max_steps_reached,
        completed_with_fallback, waiting_data, waiting_input, skipped, no_data,
        unavailable, empty, missing) mean the exploration was incomplete and we
        must NOT advance EXPLORATION -> INNOVATION; the user retains the option
        to retry stage 1 in the next round.

        'partial' is a disclosure of degraded or operationally incomplete
        evidence, not a host scientific verdict. Statistical and figure checks
        are PI review observations and do not set this status. Operational paths
        such as a supervisor force-stop may still return partial. Measured on
        session 3d675631, treating a scientifically limited result as a host
        failure produced an unrecoverable exploration_failed dialog despite 10
        findings and 22 saved figures; that scientific veto must not return.

        The degradation still reaches the report — evidence_gate_degraded,
        evidence_verified and evidence_notes ride on the round result and are
        read by Report A generation and the report's hedging.

        P1-12's actual concern was an exploration report generated from empty
        findings, so 'partial' is accepted only when the round produced some.

        Also honors @with_retry's failed_node signal (HITL Skip path).
        """
        if state.get("failed_node") == "bioinformatics_analysis":
            return False
        bio = state.get("bioinformatics_result") or {}
        status = bio.get("status")
        if status == "completed":
            return True
        if status == "partial":
            return bool(bio.get("findings"))
        return False

    def _skipped_direction_evidence(self, raw_candidates, reason):
        return build_fallback_direction_evidence(raw_candidates, status="skipped", reason=reason)

    def _note_candidate_generation_failure(self, report):
        """Carry a direction-proposal tooling failure onto the report.

        An empty candidate list has two very different causes -- the PI proposed
        nothing, or its reply could not be read -- and the report must not present
        the second as the first.
        """
        if report is None:
            return report
        notes = [getattr(self.pi_agent, "last_candidate_generation_error", None)]
        warnings = list(getattr(report, "data_quality_warnings", []) or [])
        changed = False
        for note in notes:
            if note and note not in warnings:
                warnings.append(note)
                changed = True
        if changed:
            report.data_quality_warnings = warnings
        return report

    async def _generate_evidence_gated_exploration_report(self, graph_state, analysis_state, round_result):
        raw_candidates = await self.pi_agent.generate_raw_direction_candidates(analysis_state, round_result)
        if not raw_candidates:
            # No candidates to verify. The literature gate is skipped outright.
            # score_directions_with_evidence is still the Report A summariser (it
            # produces cluster_overview, quality warnings and the round narrative),
            # so it is still called -- but with an empty candidate list, and every
            # candidate-shaped field is cleared afterwards so a scoring step cannot
            # introduce directions nobody proposed.
            logger.info(
                f"[{self.session_id}] No direction candidates; skipping literature "
                "verification, and no candidate may survive direction scoring"
            )
            report = await self.pi_agent.score_directions_with_evidence(
                analysis_state, round_result, [], []
            )
            report.research_direction_candidates = []
            report.raw_direction_candidates = []
            report.direction_literature_evidence = []
            report.direction_dependency_graph = {}
            return self._note_candidate_generation_failure(report)
        literature_result = (graph_state or {}).get("literature_result") or {}
        if literature_result.get("status") in {"skipped", "disabled"}:
            reason = "Direction-level literature verification was skipped because the Literature Agent was skipped or disabled earlier in the workflow."
            direction_evidence = self._skipped_direction_evidence(raw_candidates, reason)
            report = await self.pi_agent.score_directions_with_evidence(analysis_state, round_result, raw_candidates, direction_evidence)
            self.pi_agent._sanitize_direction_dependency_graph(report)
            return self._note_candidate_generation_failure(report)

        try:
            direction_evidence = await self.literature_agent.evaluate_direction_novelty_candidates(raw_candidates, state=analysis_state)
        except Exception as lit_err:
            logger.exception(f"[{self.session_id}] Direction literature evidence gate failed")
            direction_evidence = build_fallback_direction_evidence(
                raw_candidates,
                status="failed",
                reason=f"Direction-level literature verification failed: {lit_err}",
                limitations=["Literature verification failed; PI scoring must be conservative."],
            )
        report = await self.pi_agent.score_directions_with_evidence(analysis_state, round_result, raw_candidates, direction_evidence)
        self.pi_agent._sanitize_direction_dependency_graph(report)
        return self._note_candidate_generation_failure(report)

    async def _pi_synthesis(self, state: WorkflowState) -> WorkflowState:
        synthesis_phases.bind_facade(sys.modules[__name__])
        return await synthesis_phases.run_pi_synthesis(self, state)

    async def _wait_for_user_input(self, state: WorkflowState) -> WorkflowState:
        synthesis_phases.bind_facade(sys.modules[__name__])
        return await synthesis_phases._wait_for_user_input(self, state)

    async def _check_convergence(self, state: WorkflowState) -> WorkflowState:
        """Check if analysis has converged.

        Hard-stop conditions are handled before this node. Here we only decide
        whether soft convergence should end the workflow, unless pending user work
        explicitly requires another round.
        """
        analysis_state = state["analysis_state"]

        if _should_force_next_round(analysis_state):
            logger.info(
                f"[{self.session_id}] Convergence overridden by pending user work: "
                f"text={bool(analysis_state.pending_human_input)}, "
                f"files={bool(analysis_state.pending_uploaded_files)}, "
                f"queue={len(analysis_state.directions_queue)}"
            )
            return {
                **state,
                "should_continue": True,
            }

        should_stop = analysis_state.should_stop(self.config.convergence_rounds)

        logger.info(
            f"[{self.session_id}] Convergence check: "
            f"round={analysis_state.current_round}, should_stop={should_stop}"
        )

        return {
            **state,
            "should_continue": not should_stop,
        }
    
