"""Post-execution plan audit and the gated supplementary analysis pass.

Two features live here, deliberately independent of each other:

* the plan audit runs unconditionally after every bioinformatics execution and is
  the only writer of `PlanStep.status` / `executed_at`. It answers "what did this
  round actually run", which nothing could answer before;
* the supplementary pass is gated, bounded, and off by one config flag. It uses
  the audit as its trigger signal, asks the PI for at most N repair tasks, runs
  them through a restricted executor, and merges the result back into the SAME
  round result.

The pass is shaped after `_run_annotation_validation_round`: a trigger predicate,
an idempotency id, pure helpers doing the deterministic work, a hard iteration and
wall-clock cap, and a contract that it can never fail the round it reviews. It
runs inside the bioinformatics node rather than as a graph node so the round still
produces exactly one RoundResult for one direction — `max_rounds`, the direction
queue and the integration bookkeeping stay untouched.
"""
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models.schemas import AgentType
from .post_execution_review import (
    apply_audit_to_plan,
    audit_plan_execution,
    collect_failure_admissions,
    collect_structured_evidence_gaps,
    compute_review_attempt_id,
    empty_review_meta,
    sanitize_review_plan,
    sanitize_review_result,
    should_run_post_execution_review,
)

logger = logging.getLogger(__name__)

_MAX_STORED_REVIEWS = 40
_MAX_AGENT_SUGGESTIONS = 20
_MAX_AGENT_SUGGESTION_LEN = 600


class PostExecutionReviewMixin:
    """Plan audit + gated supplementary pass. Composed into SingleCellAnalysisWorkflow."""

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _post_execution_review_enabled(self) -> bool:
        try:
            from ..config import default_config
            cfg = getattr(self, "config", None) or default_config
            return bool(getattr(cfg, "enable_post_execution_review", True))
        except Exception:
            return True

    # Per-iteration allowance for the executor's own LLM call, on top of the code
    # execution budget. Generous: a code-generation call on a saturated round
    # context is the slowest kind this workflow makes.
    _REVIEW_LLM_ALLOWANCE_SECONDS = 120

    # Fallbacks for a config that cannot be read at all: `sandbox_timeout`'s own
    # default, put through the same derivation the configured path applies.
    _DEFAULT_SANDBOX_TIMEOUT = 5400
    _DEFAULT_STEP_TIMEOUT = _DEFAULT_SANDBOX_TIMEOUT + _REVIEW_LLM_ALLOWANCE_SECONDS

    def _post_execution_review_limits(self) -> Tuple[int, int, int, int]:
        """(max_iterations, step_timeout, total_timeout, max_tasks) from config.

        Clamped defensively — iterations in [1,8], timeouts >= 1, tasks in [0,10] —
        so a stray config cannot make the pass loop unboundedly or commission an
        open-ended study.

        The step budget is DERIVED from `sandbox_timeout` when unset, rather than
        being an independent constant: a supplementary step can legitimately be a
        long analysis, and a fixed ceiling below `sandbox_timeout` would abandon it
        mid-execution while the kernel kept working on it. Cells are also bounded
        and interrupted by the sandbox itself, so this only has to be large enough
        that it never becomes the thing that cuts one off. The total defaults to the
        sum of the step budgets for the same reason, and is floored at one step.
        """
        try:
            from ..config import default_config
            cfg = getattr(self, "config", None) or default_config
            iterations = max(1, min(int(getattr(cfg, "post_execution_review_max_iterations", 5)), 8))
            tasks = max(0, min(int(getattr(cfg, "post_execution_review_max_tasks", 5)), 10))

            step = int(getattr(cfg, "post_execution_review_timeout_seconds", 0) or 0)
            if step <= 0:
                sandbox_timeout = int(getattr(cfg, "sandbox_timeout", 0) or 0)
                if sandbox_timeout <= 0:
                    sandbox_timeout = int(
                        getattr(default_config, "sandbox_timeout", self._DEFAULT_SANDBOX_TIMEOUT)
                        or self._DEFAULT_SANDBOX_TIMEOUT
                    )
                step = sandbox_timeout + self._REVIEW_LLM_ALLOWANCE_SECONDS
            step = max(1, step)

            total = int(getattr(cfg, "post_execution_review_total_timeout_seconds", 0) or 0)
            if total <= 0:
                total = step * iterations
            return (iterations, step, max(step, total), tasks)
        except Exception:
            return (5, self._DEFAULT_STEP_TIMEOUT, self._DEFAULT_STEP_TIMEOUT * 5, 5)

    # ------------------------------------------------------------------
    # Last-direction detection
    # ------------------------------------------------------------------

    def _is_last_direction_round_of_batch(self, analysis_state) -> bool:
        """True when this round is the last DIRECTION of the active batch.

        This replaces an earlier "is this the terminal round" test, which was both
        unreachable and unanswerable:

        * unreachable, twice over. `directions_queue` is popped one round LATE —
          `_pi_planning` peeks `directions_queue[0]` and explicitly does not pop,
          `_initialize_round` pops it at the START of the next round — so the entry
          for the direction this round is running is still queued while it runs.
          A bare "queue is non-empty" test therefore never fired on a last
          direction. That is the same stale-entry trap 0.30.4 already fixed for the
          multi-direction terminal gate, and reintroducing it was a regression.
          Separately, the integration branch budgets one round MORE than the batch
          it plans (`max_rounds = current_round + len(selected_dirs)`, the spare
          slot being the integration round itself), so `current_round >= max_rounds`
          was false on every round of every integration batch even with an empty
          queue. Neither test is used here any more.
        * unanswerable. Whether any analysis round follows depends on a decision
          that has not been made yet: `integration_synthesis` increments
          `integration_cycle` and the PI then chooses how many new directions to
          open, both AFTER this node runs. So "no later round can carry follow-up
          work" cannot be established here, and pretending otherwise is what made
          the old predicate look conservative while being simply dead.

        What IS decidable is whether the active batch has another direction after
        this one. That is the honest approximation: it fires once per batch rather
        than once per session, and each firing is a round after which analysis may
        well stop. The in-flight entry is filtered out because the two queue
        writers disagree on whether it is present — `_pi_planning` stores
        `selected_dirs[1:]` while the user-selection route stores every chosen
        direction including the running one.
        """
        try:
            current = str(getattr(analysis_state, "current_round_direction", "") or "").strip()
            if not current:
                # No active direction means no batch at all: the exploration
                # round clears it, and the legacy next-steps path never sets
                # it. Neither is the last direction of a batch, and an empty
                # queue must not be read as "the batch just ended". Signals
                # 1-3 still cover those rounds.
                return False
            for entry in (getattr(analysis_state, "directions_queue", None) or []):
                direction = str((entry or {}).get("direction", "") or "").strip()
                if direction and direction != current:
                    return False
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Feature A — plan audit (always runs)
    # ------------------------------------------------------------------

    def _structured_method_families(self, analysis_state, round_number: int) -> set:
        """Families vouched for by structured state rather than by code text.

        Only mappings that cannot be wrong belong here: a registered score
        definition is direct proof that gene scoring ran. Everything vaguer is
        left to the execution-evidence channel, because a false `completed` is
        worse than a missed one — it silently retires a real gap.
        """
        families = set()
        try:
            for score in getattr(analysis_state, "score_definitions", None) or []:
                score_round = int(getattr(score, "round_number", 0) or 0)
                if not round_number or not score_round or score_round == int(round_number):
                    families.add("gene_scoring")
                    break
        except Exception as e:
            logger.debug(f"[{self.session_id}] structured family scan failed: {e}")
        return families

    def _collect_round_evidence_gaps(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Record this round's stated evidence gaps, with vocabulary labels.

        Observation only — nothing in the trigger path reads it, and it is kept out
        of the PI's synthesis context so that recording it cannot change what the
        round concludes. It exists so the distribution can be measured on real runs
        of the current code before an expansion trigger is designed around it. The
        labels are not a feasibility verdict; the verbatim items are the useful part.
        """
        try:
            gaps = collect_structured_evidence_gaps(result.get("mechanism_programs") or [])
        except Exception as e:
            logger.debug(f"[{self.session_id}] evidence-gap scan failed: {e}")
            return {}
        if not gaps.get("programs_total"):
            # No registered claim, so nothing states a gap. Returning an empty dict
            # keeps the audit summary free of a record that says nothing.
            return {}
        counts = gaps.get("counts") or {}
        logger.info(
            f"[{self.session_id}] Evidence gaps (observation only): "
            f"{gaps['programs_with_gaps']}/{gaps['programs_total']} programs state a gap; "
            f"mentions-computation={counts.get('mentions_computation', 0)}, "
            f"mentions-new-measurement={counts.get('mentions_new_measurement', 0)}, "
            f"unlabelled={counts.get('unlabelled', 0)}"
        )
        # A registered claim with no citable execution is the sharper signal of the
        # two: the tool schema requires evidence ids, so an empty list means the
        # claim cannot be traced to anything this round ran.
        if gaps.get("programs_without_evidence_ids"):
            logger.warning(
                f"[{self.session_id}] Mechanism programs with no citable execution: "
                f"{'; '.join(gaps['programs_without_evidence_ids'][:5])}"
            )
        return gaps

    def _audit_round_plan_execution(self, analysis_state, result: Dict[str, Any]) -> Dict[str, Any]:
        """Audit the approved plan against this round's evidence and record it.

        Writes `PlanStep.status` / `executed_at` on `analysis_state.current_plan`.
        Never raises: an audit failure must not fail the round it describes.
        Returns the audit summary (empty dict when there was no plan to audit).
        """
        evidence_gaps = self._collect_round_evidence_gaps(result)
        plan = getattr(analysis_state, "current_plan", None)
        if plan is None or not getattr(plan, "steps", None):
            return {"evidence_gaps": evidence_gaps} if evidence_gaps else {}
        try:
            round_number = int(getattr(analysis_state, "current_round", 0) or 0)
            audit = audit_plan_execution(
                plan,
                getattr(analysis_state, "code_history", None) or [],
                round_number=round_number,
                structured_families=self._structured_method_families(analysis_state, round_number),
            )
            updated = apply_audit_to_plan(plan, audit)
            summary = audit.summary()
            logger.info(
                f"[{self.session_id}] Plan audit: {summary['n_completed']}/{summary['n_steps']} "
                f"evidenced, {summary['n_skipped']} skipped, {summary['n_unmet']} unmet, "
                f"{summary['n_unverifiable']} unverifiable ({updated} step statuses written)"
            )
            if summary["n_unmet"]:
                logger.warning(
                    f"[{self.session_id}] Unmet approved steps: "
                    f"{'; '.join(summary['unmet_titles'][:5])}"
                )
            self._last_plan_audit = audit
            if evidence_gaps:
                summary["evidence_gaps"] = evidence_gaps
            return summary
        except Exception as e:
            logger.warning(f"[{self.session_id}] plan audit failed (non-critical): {e}")
            self._last_plan_audit = None
            return {"evidence_gaps": evidence_gaps} if evidence_gaps else {}

    # ------------------------------------------------------------------
    # Feature B — gated supplementary pass
    # ------------------------------------------------------------------

    def _find_stored_review(self, analysis_state, attempt_id: str) -> Optional[Dict[str, Any]]:
        for record in reversed(getattr(analysis_state, "post_execution_reviews", None) or []):
            if isinstance(record, dict) and record.get("review_attempt_id") == attempt_id:
                return record
        return None

    def _store_review(self, analysis_state, meta: Dict[str, Any]) -> None:
        try:
            records = getattr(analysis_state, "post_execution_reviews", None)
            if records is None:
                return
            records.append(meta)
            if len(records) > _MAX_STORED_REVIEWS:
                del records[:-_MAX_STORED_REVIEWS]
        except Exception as e:
            logger.debug(f"[{self.session_id}] could not store review record: {e}")

    async def _run_post_execution_review(
        self,
        analysis_state,
        result: Dict[str, Any],
        audit_summary: Dict[str, Any],
        planning_stage: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Review the completed execution and, if warranted, supplement it.

        Returns (review_meta, merge_payload). `merge_payload` carries what the
        caller folds into the round's result — findings, figures, code history and
        analysis log from the supplementary pass — so the round keeps a single
        result and downstream synthesis, evidence provenance and reporting need no
        change. Never raises into the caller.
        """
        audit = getattr(self, "_last_plan_audit", None)
        empty_merge: Dict[str, Any] = {}

        if not self._post_execution_review_enabled():
            return empty_review_meta(
                "skipped", "Post-execution review is disabled by configuration."), empty_merge

        # 1) Trigger inputs. The audit is advisory here: it names candidate gaps,
        #    it does not authorize spending on its own.
        findings = list(result.get("findings") or [])
        admissions = collect_failure_admissions(findings, result.get("analysis_log") or [])
        result_status = str(result.get("status", "completed") or "completed")
        is_terminal = self._is_last_direction_round_of_batch(analysis_state)
        has_runtime = bool(getattr(self, "bioinformatics_agent", None))

        plan = getattr(analysis_state, "current_plan", None)
        attempt_id = compute_review_attempt_id(
            int(getattr(analysis_state, "current_round", 0) or 0),
            str(getattr(plan, "id", "") or ""),
            planning_stage,
            result.get("evidence_execution_ids") or [],
            result_status,
        )

        def _decline(status: str, reason: str, **extra) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            meta = empty_review_meta(status, reason, attempt_id)
            meta["plan_audit"] = audit_summary
            meta.update(extra)
            self._store_review(analysis_state, meta)
            return meta, empty_merge

        # 2) Idempotency. Node retry (@with_retry, 3 attempts), websocket reconnect
        #    and Snapshot V2 restore all re-enter this node; a supplementary pass
        #    must not be paid for twice for the same execution.
        stored = self._find_stored_review(analysis_state, attempt_id)
        if stored is not None:
            logger.info(
                f"[{self.session_id}] reusing post-execution review for attempt {attempt_id[:8]}"
            )
            return stored, empty_merge

        if audit is None:
            from .post_execution_review import PlanAudit
            audit = PlanAudit()

        run, reason = should_run_post_execution_review(
            audit, admissions, result_status, is_terminal, has_runtime=has_runtime
        )
        if not run:
            logger.info(f"[{self.session_id}] Post-execution review skipped: {reason}")
            return _decline("skipped", reason)

        logger.info(f"[{self.session_id}] Post-execution review triggered: {reason}")
        max_iterations, step_timeout, total_timeout, max_tasks = \
            self._post_execution_review_limits()
        if max_tasks == 0:
            return _decline("skipped", "Supplementary task budget is zero by configuration.")

        # 3) PI reviews the completed result set and triages what is worth doing.
        await self._send_message("status", "PI is reviewing execution completeness...", AgentType.PI)
        agent_suggestions = self._collect_agent_suggestions(result)
        direction = str(getattr(analysis_state, "current_round_direction", "") or "")
        try:
            raw_plan = await self.pi_agent.review_round_execution(
                state=analysis_state,
                plan_audit=audit_summary,
                admissions=admissions,
                findings=findings,
                agent_suggestions=agent_suggestions,
                direction=direction,
                result_status=result_status,
                is_terminal_round=is_terminal,
                max_tasks=max_tasks,
                trigger_reason=reason,
            )
        except Exception as e:
            logger.warning(f"[{self.session_id}] post-execution review planning failed: {e}")
            return _decline("failed", f"Review planning failed: {e}")

        review_plan = sanitize_review_plan(raw_plan, max_tasks)
        if review_plan["verdict"] == "no_action" or not review_plan["supplementary_tasks"]:
            logger.info(f"[{self.session_id}] PI declined supplementary analysis")
            return _decline(
                "no_action",
                review_plan.get("rationale") or "PI found no supplementary analysis warranted.",
                review_plan=review_plan,
            )

        # 4) Restricted execution. Its own bounded loop, its own timeout; it cannot
        #    call report_findings, consult_pi or the figure pipeline, so it can
        #    neither end the round nor spend the round's figure budget.
        await self._send_message(
            "status",
            f"Bioinformatics agent is running {len(review_plan['supplementary_tasks'])} "
            f"supplementary task(s)...",
            AgentType.BIOINFORMATICS,
        )
        try:
            raw_result = await self.bioinformatics_agent.run_supplementary_tasks(
                tasks=review_plan["supplementary_tasks"],
                context_summary=self._build_supplementary_context(result, review_plan),
                direction=direction,
                max_iterations=max_iterations,
                timeout_seconds=step_timeout,
                total_timeout_seconds=total_timeout,
            )
        except Exception as e:
            logger.warning(f"[{self.session_id}] supplementary execution failed: {e}")
            raw_result = {"status": "failed", "summary": f"Supplementary execution failed: {e}",
                          "findings": [], "figures": [], "warnings": [str(e)]}

        supplementary = sanitize_review_result(raw_result)

        # 5) PI re-assesses: which gaps closed, which remain. PI sees the result
        #    even when execution failed, so the round's summary can state that.
        await self._send_message("status", "PI is assessing supplementary results...", AgentType.PI)
        try:
            assessment = await self.pi_agent.assess_supplementary_result(
                state=analysis_state,
                review_plan=review_plan,
                supplementary_result=supplementary,
                direction=direction,
            ) or {}
        except Exception as e:
            logger.warning(f"[{self.session_id}] supplementary assessment failed: {e}")
            assessment = {"verdict": "partial", "summary": "", "resolved_gaps": [],
                          "remaining_gaps": review_plan.get("gaps", [])}

        meta = {
            "review_attempt_id": attempt_id,
            "review_status": supplementary.get("status", "failed"),
            "review_reason": reason,
            "review_plan": review_plan,
            "review_result": {
                "status": supplementary.get("status"),
                "summary": supplementary.get("summary"),
                "warnings": supplementary.get("warnings", []),
            },
            "plan_audit": audit_summary,
            "supplementary_findings": supplementary.get("findings", []),
            "assessment": {
                "verdict": str(assessment.get("verdict", "partial") or "partial"),
                "summary": str(assessment.get("summary", "") or "")[:4000],
                "resolved_gaps": list(assessment.get("resolved_gaps") or [])[:12],
                "remaining_gaps": list(assessment.get("remaining_gaps") or [])[:12],
            },
            "round_number": int(getattr(analysis_state, "current_round", 0) or 0),
        }
        self._store_review(analysis_state, meta)

        merge_payload = {
            "findings": supplementary.get("findings", []),
            "figures": supplementary.get("figures", []),
            "code_history": raw_result.get("code_history") or [],
            "analysis_log": raw_result.get("analysis_log") or [],
            "assessment_summary": meta["assessment"]["summary"],
        }
        logger.info(
            f"[{self.session_id}] Post-execution review complete: "
            f"status={meta['review_status']}, verdict={meta['assessment']['verdict']}, "
            f"{len(merge_payload['findings'])} supplementary finding(s)"
        )
        await self._send_message("post_execution_review", meta, AgentType.PI)
        return meta, merge_payload

    # ------------------------------------------------------------------
    # Merging the pass back into the round
    # ------------------------------------------------------------------

    def _merge_supplementary_into_round(
        self,
        analysis_state,
        result: Dict[str, Any],
        all_findings: List[str],
        all_figures: List[Any],
        merge_payload: Dict[str, Any],
    ) -> Tuple[List[str], List[Any]]:
        """Fold a supplementary pass into the round's own result.

        Findings are tagged so a reader can tell repair work from the planned
        analysis, and executions are appended to `code_history` with the same
        identity-based dedup the main path uses — a supplementary execution is
        evidence like any other and must be citable. `result` is updated in place
        so a re-audit sees the new executions.
        """
        from ..models.schemas import AnalysisStep

        tagged = [f"[Supplementary] {f}" for f in (merge_payload.get("findings") or [])]
        merged_findings = list(all_findings) + tagged
        merged_figures = list(all_figures) + list(merge_payload.get("figures") or [])

        existing_code_ids = {c.id for c in analysis_state.code_history}
        added_executions = []
        for execution in merge_payload.get("code_history") or []:
            if getattr(execution, "id", None) is None or execution.id in existing_code_ids:
                continue
            analysis_state.code_history.append(execution)
            existing_code_ids.add(execution.id)
            added_executions.append(execution)

        existing_step_ids = {s.id for s in analysis_state.analysis_log}
        for step in merge_payload.get("analysis_log") or []:
            if isinstance(step, dict):
                try:
                    step = AnalysisStep(**step)
                except Exception:
                    continue
            if step.id in existing_step_ids:
                continue
            step.round_number = max(int(getattr(analysis_state, "current_round", 1) or 1), 1)
            analysis_state.analysis_log.append(step)
            existing_step_ids.add(step.id)

        # Keep the raw result consistent so the re-audit and any later reader see
        # the supplementary executions as part of this round.
        result["findings"] = merged_findings
        result["figures"] = merged_figures
        if added_executions:
            result["code_history"] = list(result.get("code_history") or []) + added_executions
            # Successful supplementary executions ARE citable evidence, so they join
            # the round's evidence ids — the report and the audit both need them.
            #
            # `evidence_verified` and `status` are deliberately left alone. It is
            # tempting to flip them, since a round whose main loop ended as
            # `max_steps_reached` or whose evidence gate degraded stays ineligible
            # for direction coverage even after a successful repair. But
            # `evidence_verified is False` records that THIS ROUND'S REPORTED
            # FINDINGS were not bound to successful executions, and running five
            # more cells does not make those findings bound. Overwriting it would
            # buy coverage eligibility by erasing the fact that earned the
            # ineligibility — exactly the hedging the gate exists to force.
            fresh_ids = [e.id for e in added_executions if not getattr(e, "error", None)]
            if fresh_ids:
                result["evidence_execution_ids"] = list(
                    result.get("evidence_execution_ids") or []
                ) + fresh_ids

        logger.info(
            f"[{self.session_id}] Merged supplementary pass: +{len(tagged)} finding(s), "
            f"+{len(added_executions)} execution(s)"
        )
        return merged_findings, merged_figures

    # ------------------------------------------------------------------
    # Prompt inputs
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_agent_suggestions(result: Dict[str, Any]) -> List[str]:
        """The executing agent's own follow-up suggestions.

        `report_findings` carries a `suggestions` array in every real call and the
        fixed-key result dict dropped it, so the PI never saw what the agent that
        ran the analysis thought was left undone. Read it back off the analysis log.
        """
        suggestions: List[str] = []
        seen: Set[str] = set()
        for step in result.get("analysis_log") or []:
            if isinstance(step, dict):
                if step.get("step_type") != "report_findings":
                    continue
                items = step.get("findings_suggestions") or []
            else:
                if getattr(step, "step_type", "") != "report_findings":
                    continue
                items = getattr(step, "findings_suggestions", None) or []
            for item in items:
                text = str(item or "").strip()
                if not text or text in seen:
                    continue
                text = text[:_MAX_AGENT_SUGGESTION_LEN]
                seen.add(text)
                suggestions.append(text)
                if len(suggestions) >= _MAX_AGENT_SUGGESTIONS:
                    return suggestions
        return suggestions

    @staticmethod
    def _build_supplementary_context(result: Dict[str, Any], review_plan: Dict[str, Any]) -> str:
        """Compact context for the restricted executor: what this round concluded
        and which gaps the tasks are meant to close."""
        parts: List[str] = []
        findings = [str(f) for f in (result.get("findings") or [])][:12]
        if findings:
            parts.append("Findings already established this round:\n"
                         + "\n".join(f"- {f[:400]}" for f in findings))
        gaps = review_plan.get("gaps") or []
        if gaps:
            parts.append("Gaps these tasks must close:\n"
                         + "\n".join(f"- {str(g)[:300]}" for g in gaps[:8]))
        return "\n\n".join(parts)
