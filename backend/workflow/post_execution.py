"""Post-execution observations and the PI-directed supplementary analysis pass.

Two features live here, deliberately independent of each other:

* the plan audit runs unconditionally after every bioinformatics execution and
  records technical linkage observations without deciding plan completion;
* the supplementary pass is PI-directed, bounded, and off by one config flag. It
  shows the observations to the PI, asks for at most N repair tasks, runs
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
import inspect
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models.schemas import AgentType
from ..models.schema_parts.enums import AnalysisPhase
from ..services.figure_contract import visual_plan_inheritance_for_plan
from ..services.figure_revision_tasks import (
    apply_pi_fallback_selection,
    candidate_figures_for_task,
    materialize_missing_figure_tasks,
)
from .post_execution_review import (
    audit_plan_execution,
    collect_failure_admissions,
    collect_structured_evidence_gaps,
    compute_review_attempt_id,
    empty_review_meta,
    sanitize_review_plan,
    sanitize_review_result,
    should_run_post_execution_review,
)
from .post_execution_review_parts.reconciliation import (
    reconcile_plan_evidence,
)
from .direction_coverage import (
    build_result_identity_key,
    current_lineage_id,
    normalize_direction_key,
)
from ..utils.direction_payload import (
    build_direction_projection,
    build_direction_projection_block,
    FOLLOWUP_DISPOSITIONS,
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
    _REVIEW_LLM_ALLOWANCE_SECONDS = 600

    # Fallbacks for a config that cannot be read at all: `sandbox_timeout`'s own
    # default, put through the same derivation the configured path applies.
    _DEFAULT_SANDBOX_TIMEOUT = 5400
    _DEFAULT_STEP_TIMEOUT = _DEFAULT_SANDBOX_TIMEOUT + _REVIEW_LLM_ALLOWANCE_SECONDS

    def _post_execution_review_limits(self) -> Tuple[int, int, int, int]:
        """(iteration_ceiling, step_timeout, total_timeout, max_tasks) from config.

        Clamped defensively — iterations in [1,20], step timeout >= 600, tasks in [0,10] —
        so a stray config cannot make the pass loop unboundedly or commission an
        open-ended study.

        The step budget is DERIVED from `sandbox_timeout` when unset, rather than
        being an independent constant: a supplementary step can legitimately be a
        long analysis, and a fixed ceiling below `sandbox_timeout` would abandon it
        mid-execution while the kernel kept working on it. Cells are also bounded
        and interrupted by the sandbox itself, so this only has to be large enough
        that it never becomes the thing that cuts one off. An unset total stays zero
        here so the executor can derive it from the approved task count and reserve
        a final reporting request after the PI has selected the tasks.
        """
        try:
            from ..config import default_config
            cfg = getattr(self, "config", None) or default_config
            iterations = max(1, min(int(getattr(cfg, "post_execution_review_max_iterations", 20)), 20))
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
            step = max(600, step)

            total = int(getattr(cfg, "post_execution_review_total_timeout_seconds", 0) or 0)
            return (iterations, step, max(step, total) if total > 0 else 0, tasks)
        except Exception:
            return (20, self._DEFAULT_STEP_TIMEOUT, 0, 5)

    async def _run_figure_followup(
        self, analysis_state, contract, planning_stage: str,
    ) -> Dict[str, Any]:
        """Close figure tasks through durable PI decisions and one bounded pass."""
        materialize_missing_figure_tasks(analysis_state, contract)
        from ..services.figure_followup import (
            MAX_SCOPE_TASKS,
            build_figure_followup_scope,
            create_needs_new_analysis_request,
            persist_figure_followup_scope,
        )
        plan_id = getattr(analysis_state.current_plan, "id", None)
        lineage = analysis_state.restore_branch_id or "main"
        tasks = [
            task for task in analysis_state.figure_revision_tasks
            if task.status in {"pending", "in_progress", "blocked", "exhausted_excluded"}
            and (task.plan_id or None) == plan_id
            and (task.lineage_id or "main") == lineage
            and (not task.round_number or task.round_number == analysis_state.current_round)
        ]
        if not tasks:
            return {"status": "not_needed", "dispositions": []}

        undecided = [task for task in tasks if not task.pi_disposition]
        candidate_counts = {
            task.id: len(candidate_figures_for_task(analysis_state, task))
            for task in undecided
        }
        for offset in range(0, len(undecided), 12):
            batch = undecided[offset:offset + 12]
            decisions = await self.pi_agent.decide_figure_followups(
                analysis_state, batch, candidate_counts,
            )
            by_id = {task.id: task for task in batch}
            for decision in decisions:
                task = by_id.get(decision["task_id"])
                if task is None:
                    continue
                task.pi_disposition = decision["disposition"]
                task.pi_instruction = decision.get("instruction") or None
                task.pi_rationale = decision.get("rationale") or None
                task.pi_report_note = decision.get("report_note") or None
                task.updated_at = datetime.now()

        targeted, select_existing = [], []
        for task in tasks:
            if task.followup_finished_at:
                continue
            if not task.pi_disposition:
                task.status = "blocked"
                task.pi_report_note = "No valid PI figure disposition was obtained."
            elif task.pi_disposition == "omit":
                task.status = "abandoned"
                task.followup_finished_at = datetime.now()
            elif task.pi_disposition == "accept_best":
                select_existing.append(task)
            elif task.pi_disposition == "needs_new_analysis":
                request = create_needs_new_analysis_request(analysis_state, task)
                task.status = "blocked"
                task.pi_report_note = (
                    "This figure requires new scientific analysis and was not run "
                    "during figure follow-up. A separately reviewable request was "
                    f"recorded as {request['id']}."
                )
                task.followup_finished_at = datetime.now()
            elif (
                task.pi_disposition == "targeted_redraw"
                and (task.followup_started_at or task.status == "in_progress")
            ):
                # A prior invocation may have dispatched code before interruption.
                # Resume closure using existing candidates, without granting a new
                # execution budget or replaying a possibly completed tool call.
                select_existing.append(task)
            elif task.pi_disposition == "targeted_redraw":
                task.status = "in_progress"
                task.followup_started_at = datetime.now()
                targeted.append(task)
            else:
                task.status = "blocked"
                task.pi_report_note = "The PI figure disposition is not supported."

        parent_run_scope = str(
            getattr(
                analysis_state.current_plan,
                "bioinformatics_run_scope_id",
                "",
            )
            or plan_id
            or "unplanned"
        )
        followup_scope = None
        if targeted:
            deferred_targeted = targeted[MAX_SCOPE_TASKS:]
            targeted = targeted[:MAX_SCOPE_TASKS]
            for task in deferred_targeted:
                task.status = "pending"
                task.followup_started_at = None
                task.updated_at = datetime.now()
            lineage = analysis_state.restore_branch_id or "main"
            followup_scope_id = "figure_scope_" + hashlib.sha256(
                (
                    f"{lineage}|{parent_run_scope}|{planning_stage}|"
                    + ",".join(sorted(task.id for task in targeted))
                ).encode("utf-8")
            ).hexdigest()
            followup_scope = build_figure_followup_scope(
                analysis_state, targeted, run_scope_id=followup_scope_id
            )
            task_entries = {
                entry.get("task_id"): entry
                for entry in followup_scope.get("tasks", [])
                if isinstance(entry, dict)
            }
            supported = []
            for task in targeted:
                entry = task_entries.get(task.id)
                if entry and entry.get("eligible_execution_ids"):
                    supported.append(task)
                else:
                    task.status = "blocked"
                    task.pi_report_note = (
                        "No pre-existing successful ordinary execution receipt is "
                        "bound to this task's plan step and execution scope; figure follow-up did "
                        "not dispatch an analysis or render."
                    )
                    task.followup_finished_at = datetime.now()
            targeted = supported
            if targeted:
                followup_scope = build_figure_followup_scope(
                    analysis_state,
                    targeted,
                    run_scope_id=str(followup_scope["run_scope_id"]),
                )
                persist_figure_followup_scope(analysis_state, followup_scope)
            else:
                followup_scope = None

        if not await self._persist_review_state(analysis_state):
            # Nothing has been dispatched. An unacknowledged lifecycle transition
            # must never authorize a possibly duplicated execution after restart.
            for task in targeted:
                task.status = "pending"
                task.followup_started_at = None
            return {"status": "persistence_failed", "dispositions": []}

        redraw_result: Dict[str, Any] = {}
        if targeted:
            from ..services.figure_followup import figure_followup_source_prompt
            from ..agents.bioinformatics.checkpoint_preflight import ensure_checkpoint_adata

            # Repair the owner namespace before the disposable renderer pins and
            # saves its generation. The checkpoint boundary also rechecks later.
            await ensure_checkpoint_adata(
                self.bioinformatics_agent, getattr(self, "sandbox", None), analysis_state,
                boundary="before_figure_followup",
                summary="\n".join(str(task.pi_instruction or "") for task in targeted),
            )
            lines = [
                f"- task_id={task.id}; logical_figure_id={task.logical_figure_id}; "
                f"plan_step_id={task.plan_step_id}; "
                f"plan_requirement_key={task.plan_requirement_key or 'supplementary'}; "
                f"coverage={task.found_count}/{task.required_count}\n"
                f"  PI instruction: {task.pi_instruction}"
                for task in targeted
            ]
            instruction = (
                "Complete only the PI-directed publication-figure tasks below. "
                "Use only objects already present in the committed runtime and the "
                "pre-existing execution receipts named in the task scope. Use the "
                "canonical adata for the primary AnnData when present; honor any "
                "explicit working binding in the source context. A filename such "
                "as adata_annotated.h5ad does not establish an adata_annotated "
                "variable. Do not guess alternative variable names. Do not "
                "load data, perform or persist scientific analysis, or introduce "
                "new scientific claims. Each task owns ONE DISTINCT output. Use a "
                "distinct filename for each task. Call generate_publication_figure "
                "with its exact figure_task_id, logical_figure_id, plan_step_id, "
                "and a supporting evidence_execution_id. Normal PI figure review "
                "applies. After attempting every task, call "
                "complete_figure_followup with an exact delivered/unresolved task "
                "partition and a concise render summary.\n\n"
                + figure_followup_source_prompt(analysis_state, followup_scope)
                + "\n\n" + "\n".join(lines)
            )
            try:
                redraw_result = await self.bioinformatics_agent.run_autonomous_analysis(
                    analysis_state, instruction,
                    max_steps=min(40, max(12, len(targeted) * 8)),
                    figure_review_callback=self._figure_review_callback,
                    supervisor_callback=None,
                    resume_from_checkpoint=True,
                    planning_stage=planning_stage,
                    figure_followup=True,
                    run_kind="figure_followup",
                    run_scope_id=str(followup_scope["run_scope_id"]),
                )
            except Exception as exc:
                logger.warning("[%s] figure follow-up failed: %s", self.session_id, exc)
                redraw_result = {"status": "failed", "error": str(exc)}

            select_existing.extend(targeted)

        for task in select_existing:
            if task.status != "resolved":
                candidates = candidate_figures_for_task(analysis_state, task)
                selection = await self.pi_agent.select_fallback_figure(
                    analysis_state, task, candidates
                )
                selected = apply_pi_fallback_selection(
                    analysis_state, task, **selection
                ) if selection else None
                if selected is None:
                    task.status = "blocked"
                    task.pi_report_note = (
                        "The figure task remains incomplete: no usable candidate "
                        "was selected by the PI after the bounded attempt. "
                        + (task.pi_report_note or "")
                    )[:4000]
            task.followup_finished_at = datetime.now()
            task.updated_at = datetime.now()
        persisted = await self._persist_review_state(analysis_state)
        remaining = [task.id for task in tasks if task.status not in {"resolved", "abandoned"}]
        return {
            "status": "persistence_failed" if not persisted else (
                "completed" if not remaining else "incomplete"
            ),
            "dispositions": [
                {"task_id": task.id, "disposition": task.pi_disposition,
                 "status": task.status, "report_note": task.pi_report_note,
                 "selected_figure_id": task.selected_figure_id}
                for task in tasks
            ],
            "remaining_task_ids": remaining,
            "targeted_task_ids": [task.id for task in targeted],
            "redraw_status": redraw_result.get("status"),
            "figures": list(redraw_result.get("figures", []) or []),
            "figure_attempt_execution_ids": list(
                redraw_result.get("figure_attempt_execution_ids") or []
            ),
            "executor_run_scope_id": redraw_result.get("executor_run_scope_id"),
            "executor_settlement_scope_id": redraw_result.get(
                "executor_settlement_scope_id"
            ),
            "executor_checkpoint_name": redraw_result.get(
                "executor_checkpoint_name"
            ),
        }

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

    async def _reconcile_plan_evidence(
        self, analysis_state, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await reconcile_plan_evidence(self, analysis_state, result)
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

    def _audit_round_plan_execution(
        self,
        analysis_state,
        result: Dict[str, Any],
        reconciliation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Observe the approved plan against this round's evidence.

        Never writes PI-owned plan completion state and never raises: an audit
        failure must not fail the round it describes.
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
                publication_figures=getattr(analysis_state, "publication_figures", None) or [],
                inherited_plan_ids_by_step=visual_plan_inheritance_for_plan(
                    analysis_state, plan
                ),
                target_lineage_id=(
                    getattr(analysis_state, "restore_branch_id", None) or "main"
                ),
                semantic_reconciliation=reconciliation,
                analysis_state=analysis_state,
            )
            summary = audit.summary()
            logger.info(
                f"[{self.session_id}] Plan audit: {summary['n_completed']}/{summary['n_steps']} "
                f"evidenced, {summary['n_skipped']} skipped, {summary['n_unmet']} unmet, "
                f"{summary['n_unverifiable']} unverifiable (observation only)"
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

    def _find_stored_direction_review(
        self, analysis_state, direction_pass_id: str
    ) -> Optional[Dict[str, Any]]:
        """Find the one durable review record for an innovation direction pass."""
        if not direction_pass_id:
            return None
        for record in reversed(getattr(analysis_state, "post_execution_reviews", None) or []):
            if isinstance(record, dict) and record.get("direction_pass_id") == direction_pass_id:
                return record
        return None

    def _store_review(self, analysis_state, meta: Dict[str, Any]) -> None:
        try:
            records = getattr(analysis_state, "post_execution_reviews", None)
            if records is None:
                return
            stable_id = str(meta.get("direction_pass_id") or "").strip()
            attempt_id = str(meta.get("review_attempt_id") or "").strip()
            if stable_id or attempt_id:
                for index in range(len(records) - 1, -1, -1):
                    existing = records[index]
                    if not isinstance(existing, dict):
                        continue
                    same_direction = bool(
                        stable_id and existing.get("direction_pass_id") == stable_id
                    )
                    same_attempt = bool(
                        not stable_id
                        and attempt_id
                        and existing.get("review_attempt_id") == attempt_id
                    )
                    if same_direction or same_attempt:
                        records[index] = meta
                        return
            records.append(meta)
            if len(records) > _MAX_STORED_REVIEWS:
                del records[:-_MAX_STORED_REVIEWS]
        except Exception as e:
            logger.debug(f"[{self.session_id}] could not store review record: {e}")

    async def _persist_review_state(self, analysis_state) -> bool:
        """Force-save a review lifecycle transition before/after work."""
        try:
            callback = getattr(self, "state_callback", None)
            if callback is not None:
                try:
                    supports_force = "force" in inspect.signature(callback).parameters
                except (TypeError, ValueError):
                    supports_force = False
                outcome = (
                    callback(analysis_state, force=True)
                    if supports_force
                    else callback(analysis_state)
                )
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                return outcome is not False
            from ..services.session_store import async_save_session
            outcome = await async_save_session(self.session_id, analysis_state, force=True)
            return outcome is not False
        except Exception as exc:
            logger.warning(
                f"[{self.session_id}] Could not persist post-execution review lifecycle: {exc}"
            )
            return False

    async def _finalize_review_technical_snapshots(
        self,
        analysis_state,
        review_meta: Dict[str, Any],
        *,
        figure_contract_at_review: Dict[str, Any],
        final_figure_contract: Dict[str, Any],
        figure_followup: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Attach final objective figure state without rewriting PI judgement."""
        if not isinstance(review_meta, dict) or not review_meta:
            return review_meta
        updated = dict(review_meta)
        snapshots = dict(updated.get("technical_snapshots") or {})
        snapshots.setdefault("at_review", {
            "plan_audit": self._serialize_review_record(
                updated.get("plan_audit") or {}
            ),
            "figure_contract": self._serialize_review_record(
                figure_contract_at_review or {}
            ),
        })
        snapshots["after_figure_followup"] = {
            "figure_contract": self._serialize_review_record(
                final_figure_contract or {}
            ),
            "figure_followup": self._serialize_review_record({
                "status": figure_followup.get("status"),
                "remaining_task_ids": list(
                    figure_followup.get("remaining_task_ids") or []
                )[:40],
                "targeted_task_ids": list(
                    figure_followup.get("targeted_task_ids") or []
                )[:40],
                "redraw_status": figure_followup.get("redraw_status"),
                "figures": list(figure_followup.get("figures") or [])[:40],
            }),
            "persistence_status": "acknowledged",
            "finalized_at": datetime.now().isoformat(),
        }
        updated["technical_snapshots"] = snapshots
        if not (
            str(updated.get("direction_pass_id") or "").strip()
            or str(updated.get("review_attempt_id") or "").strip()
        ):
            return updated
        self._store_review(analysis_state, updated)
        if not await self._persist_review_state(analysis_state):
            snapshots["after_figure_followup"]["persistence_status"] = "pending"
            updated["technical_snapshots"] = snapshots
            self._store_review(analysis_state, updated)
        return updated

    @staticmethod
    def _serialize_review_record(value: Any) -> Any:
        """Serialize a bounded provenance object for replayable review state."""
        if isinstance(value, dict):
            return {
                str(key): PostExecutionReviewMixin._serialize_review_record(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [PostExecutionReviewMixin._serialize_review_record(item) for item in value]
        dumper = getattr(value, "model_dump", None)
        if callable(dumper):
            try:
                return PostExecutionReviewMixin._serialize_review_record(dumper(mode="json"))
            except Exception:
                try:
                    return PostExecutionReviewMixin._serialize_review_record(dumper())
                except Exception:
                    return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _bounded_assessment_text(value: Any, limit: int) -> Optional[str]:
        """Accept only bounded scalar prose at the PI assessment boundary."""
        if not isinstance(value, str):
            return None
        text = value.strip()
        return text[:limit] if text else None

    @staticmethod
    def _bounded_assessment_items(
        value: Any, *, limit: int = 12, item_limit: int = 500,
    ) -> List[str]:
        """Read list-shaped PI fields without iterating malformed scalars."""
        if not isinstance(value, (list, tuple)):
            return []
        output: List[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            output.append(text[:item_limit])
            if len(output) >= limit:
                break
        return output

    @staticmethod
    def _bounded_assessment_verdict(value: Any) -> str:
        """Keep the persisted assessment verdict within its finite contract."""
        verdict = value.strip().lower() if isinstance(value, str) else ""
        return verdict if verdict in {"resolved", "partial", "unresolved"} else "partial"

    @staticmethod
    def _known_review_execution_ids(
        raw_result: Any, supplementary: Dict[str, Any],
    ) -> List[str]:
        """Return only execution IDs present in the attempted pass evidence."""
        values: List[str] = []
        seen: Set[str] = set()

        def add(value: Any) -> None:
            if isinstance(value, dict):
                value = value.get("id") or value.get("execution_id")
            else:
                value = getattr(value, "id", None) or getattr(value, "execution_id", None)
            if not isinstance(value, (str, int)):
                return
            text = str(value).strip()[:120]
            if text and text not in seen:
                seen.add(text)
                values.append(text)

        for record in (supplementary.get("evidence_records") or []):
            if isinstance(record, dict) and str(record.get("status") or "").lower() in {
                "completed", "failed",
            }:
                add(record)
        raw_mapping = raw_result if isinstance(raw_result, dict) else {}
        for execution in raw_mapping.get("code_history") or []:
            add(execution)
        return values[:12]

    @staticmethod
    def _shared_review_outcome(
        supplementary: Dict[str, Any], known_execution_ids: List[str],
    ) -> str:
        """Describe pass-level execution without claiming each task ran."""
        status = str(supplementary.get("status") or "unknown").strip().lower()[:80]
        if not status:
            status = "unknown"
        evidence_note = (
            " Stored execution evidence is available."
            if known_execution_ids else
            " No stored execution evidence record is available."
        )
        return (
            f"Shared supplementary pass outcome: {status}. This is a pass-level "
            "status, not a task-specific execution claim for this follow-up."
            + evidence_note
        )

    @staticmethod
    def _shared_review_reference(
        attempt_id: str, direction_pass_id: str, known_execution_ids: List[str],
    ) -> str:
        parts = []
        if direction_pass_id:
            parts.append(f"pass:{direction_pass_id}")
        elif attempt_id:
            parts.append(f"attempt:{attempt_id}")
        if known_execution_ids:
            parts.append("evidence:" + ",".join(known_execution_ids))
        return "; ".join(parts)[:1200]

    @classmethod
    def _attach_shared_review_outcome(
        cls,
        followups: List[Dict[str, Any]],
        *,
        supplementary: Dict[str, Any],
        known_execution_ids: List[str],
        attempt_id: str,
        direction_pass_id: str,
    ) -> List[Dict[str, Any]]:
        """Associate attempted technical evidence while preserving PI IDs."""
        if not followups:
            return followups
        outcome = cls._shared_review_outcome(supplementary, known_execution_ids)
        reference = cls._shared_review_reference(
            attempt_id, direction_pass_id, known_execution_ids,
        )
        for item in followups:
            # A PI-linked reference was admitted only after the caller checked it
            # against known pass evidence. Otherwise expose the honest shared-pass
            # association, which cannot be mistaken for task-level execution.
            item.setdefault("execution_outcome", outcome)
            item.setdefault("execution_reference", reference)
        return followups

    @classmethod
    def _merge_followup_assessment(
        cls,
        followups: List[Dict[str, Any]],
        assessment: Any,
        *,
        known_execution_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Apply PI assessment updates while preserving server-owned IDs."""
        current = [dict(item) for item in (followups or []) if isinstance(item, dict)]
        by_id = {
            str(item.get("followup_id")): item
            for item in current
            if str(item.get("followup_id") or "").strip()
        }
        updates = assessment.get("followups") if isinstance(assessment, dict) else None
        if not isinstance(updates, list):
            return current
        allowed = {
            "question", "why_it_matters", "disposition", "reason",
            "evidence_references", "suggested_instruction", "required_input",
            "execution_outcome", "execution_reference",
        }
        for update in updates:
            if not isinstance(update, dict):
                continue
            followup_id = str(update.get("followup_id") or "").strip()
            target = by_id.get(followup_id)
            if target is None:
                continue
            if "disposition" in update and str(update.get("disposition") or "").strip().lower() not in FOLLOWUP_DISPOSITIONS:
                continue
            for key in allowed:
                if key not in update:
                    continue
                value = update[key]
                if key == "evidence_references":
                    if isinstance(value, (list, tuple)):
                        refs = []
                        for ref in value:
                            if not isinstance(ref, str):
                                continue
                            text = ref.strip()[:160]
                            if text and text not in refs:
                                refs.append(text)
                            if len(refs) >= 12:
                                break
                        target[key] = refs
                    continue
                if key == "execution_reference":
                    # A PI may tie a question to one real pass execution, but
                    # cannot mint an identity through this mutable field.
                    if (
                        isinstance(value, str)
                        and known_execution_ids is not None
                        and value.strip() in known_execution_ids
                    ):
                        target[key] = value.strip()[:160]
                    continue
                limit = {
                    "question": 1000,
                    "why_it_matters": 1200,
                    "reason": 1200,
                    "suggested_instruction": 1400,
                    "required_input": 1200,
                    "execution_outcome": 1200,
                }.get(key, 1200)
                bounded = cls._bounded_assessment_text(value, limit)
                if bounded is not None:
                    target[key] = bounded
            target["followup_id"] = followup_id
        return current

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

        phase_value = getattr(getattr(analysis_state, "analysis_phase", ""), "value", None)
        phase_value = str(phase_value or getattr(analysis_state, "analysis_phase", "") or "").lower()
        round_number = int(getattr(analysis_state, "current_round", 0) or 0)
        direction = str(getattr(analysis_state, "current_round_direction", "") or "").strip()
        direction_key = normalize_direction_key(direction)
        plan_phase = str(getattr(plan, "phase", "") or "").lower()
        innovation_direction = bool(direction_key) and (
            phase_value == AnalysisPhase.INNOVATION.value
            or plan_phase == AnalysisPhase.INNOVATION.value
            or (not phase_value and str(planning_stage or "").lower() == "full")
        )
        lineage_id = current_lineage_id(analysis_state)
        direction_pass_id = (
            build_result_identity_key(lineage_id, AnalysisPhase.INNOVATION.value, round_number, direction_key)
            if innovation_direction else ""
        )

        attempt_id = compute_review_attempt_id(
            int(getattr(analysis_state, "current_round", 0) or 0),
            str(getattr(plan, "id", "") or ""),
            planning_stage,
            result.get("evidence_execution_ids") or [],
            result_status,
        )

        async def _decline(status: str, reason: str, **extra) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            meta = empty_review_meta(status, reason, attempt_id)
            meta["plan_audit"] = audit_summary
            meta["round_number"] = round_number
            if direction_pass_id:
                meta.update({
                    "direction_pass_id": direction_pass_id,
                    "direction_key": direction_key,
                    "direction": direction,
                    "lineage_id": lineage_id,
                    "lifecycle": status,
                    "lifecycle_status": status,
                })
            meta.update(extra)
            self._store_review(analysis_state, meta)
            await self._persist_review_state(analysis_state)
            return meta, empty_merge

        # 2) Idempotency. Node retry (@with_retry, 3 attempts), websocket reconnect
        #    and Snapshot V2 restore all re-enter this node; a supplementary pass
        #    must not be paid for twice for the same execution.
        stored = self._find_stored_direction_review(analysis_state, direction_pass_id)
        if stored is None:
            stored = self._find_stored_review(analysis_state, attempt_id)
        if stored is not None:
            logger.info(
                f"[{self.session_id}] reusing post-execution review for attempt {attempt_id[:8]}"
            )
            lifecycle = str(stored.get("lifecycle") or stored.get("lifecycle_status") or "").lower()
            replay_payload = stored.get("merge_payload")
            if lifecycle == "terminal_persistence_pending":
                # The paid pass already ran.  Retry the same terminal record and
                # payload, releasing it to the caller only after the forced save
                # is acknowledged; never call either agent again.
                if not isinstance(replay_payload, dict):
                    return stored, empty_merge
                retry_record = dict(stored)
                terminal_lifecycle = str(
                    stored.get("terminal_lifecycle")
                    or ("completed" if stored.get("review_status") == "completed" else "failed")
                ).lower()
                if terminal_lifecycle not in {"completed", "failed"}:
                    terminal_lifecycle = "failed"
                retry_record["lifecycle"] = terminal_lifecycle
                retry_record["lifecycle_status"] = terminal_lifecycle
                retry_record["persistence_status"] = "acknowledged"
                retry_record.pop("persistence_failure", None)
                self._store_review(analysis_state, retry_record)
                if not await self._persist_review_state(analysis_state):
                    retry_record["lifecycle"] = "terminal_persistence_pending"
                    retry_record["lifecycle_status"] = "terminal_persistence_pending"
                    retry_record["persistence_status"] = "pending"
                    retry_record["persistence_failure"] = (
                        "Terminal supplementary review state could not be durably "
                        "acknowledged; the retained merge remains pending replay."
                    )
                    self._store_review(analysis_state, retry_record)
                    return retry_record, empty_merge
                retry_record["persistence_status"] = "acknowledged"
                retry_record.pop("persistence_failure", None)
                self._store_review(analysis_state, retry_record)
                return retry_record, replay_payload
            if lifecycle in {"reserved", "running", "reservation_failed"}:
                interrupted = dict(stored)
                interrupted["review_status"] = "interrupted"
                interrupted["lifecycle"] = "interrupted"
                interrupted["lifecycle_status"] = "interrupted"
                interrupted["interruption_reason"] = (
                    "A durable supplementary reservation has no terminal result; "
                    "automatic rerun was not attempted."
                )
                interrupted["review_reason"] = interrupted["interruption_reason"]
                self._store_review(analysis_state, interrupted)
                await self._persist_review_state(analysis_state)
                return interrupted, empty_merge
            if isinstance(replay_payload, dict):
                return stored, replay_payload
            return stored, empty_merge

        if audit is None:
            from .post_execution_review import PlanAudit
            audit = PlanAudit()

        if innovation_direction:
            review_pi = True
            reason = (
                "Selected innovation direction receives one PI review so completed "
                "evidence and useful remaining questions are considered together."
            )
        else:
            review_pi, reason = should_run_post_execution_review(
                audit, admissions, result_status, is_terminal, has_runtime=has_runtime
            )
        if not review_pi:
            logger.info(f"[{self.session_id}] Post-execution review skipped: {reason}")
            return await _decline("skipped", reason)

        logger.info(f"[{self.session_id}] PI post-execution review requested: {reason}")
        max_iterations, step_timeout, total_timeout, max_tasks = \
            self._post_execution_review_limits()
        if not has_runtime:
            max_tasks = 0

        # 3) PI reviews the completed result set and triages what is worth doing.
        await self._send_message("status", "PI is reviewing execution completeness...", AgentType.PI)
        agent_suggestions = self._collect_agent_suggestions(result)
        session_names = await self._collect_session_names(analysis_state, result, has_runtime)
        direction_context = build_direction_projection_block(
            build_direction_projection(analysis_state, direction, lineage_id=lineage_id)
        )
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
                session_names=session_names,
                direction_context=direction_context if innovation_direction else "",
            )
        except Exception as e:
            logger.warning(f"[{self.session_id}] post-execution review planning failed: {e}")
            return await _decline("failed", f"Review planning failed: {e}")

        # An empty/unparseable PI response means review was unavailable.  Keep
        # the ordinary round/report path moving, but do not persist it as a PI
        # no-action decision for an innovation direction.
        if innovation_direction and (
            not isinstance(raw_plan, dict)
            or not raw_plan
            or "parse_error" in raw_plan
        ):
            return await _decline(
                "failed",
                "Review returned no structured innovation decision; no scientific closure was recorded.",
            )

        review_plan = sanitize_review_plan(
            raw_plan,
            max_tasks,
            valid_step_ids={step.step_id for step in audit.steps},
            direction_pass_id=direction_pass_id,
            source_round=round_number,
            direction=direction,
            lineage_id=lineage_id,
        )
        raw_verdict = raw_plan.get("verdict") if isinstance(raw_plan, dict) else None
        explicit_no_action = (
            isinstance(raw_verdict, str)
            and raw_verdict.strip().lower() == "no_action"
        )
        decisions = {
            item["plan_step_id"]: item
            for item in review_plan.get("step_decisions") or []
        }
        now = datetime.now()
        for step in getattr(plan, "steps", None) or []:
            decision = decisions.get(int(getattr(step, "id", 0) or 0))
            if decision is None:
                continue
            step.status = decision["status"]
            if step.status == "completed" and getattr(step, "executed_at", None) is None:
                step.executed_at = now

        if review_plan["verdict"] == "no_action" and explicit_no_action:
            logger.info(f"[{self.session_id}] PI declined supplementary analysis")
            return await _decline(
                "no_action",
                review_plan.get("rationale") or "PI found no supplementary analysis warranted.",
                review_plan=review_plan,
                followups=review_plan.get("followups") or [],
            )
        if review_plan["verdict"] == "no_action":
            technical_status = "unavailable" if not has_runtime else "failed"
            technical_reason = (
                "Supplementary execution was unavailable because no live analysis "
                "runtime is attached; the PI review did not provide an explicit "
                "no_action decision."
                if not has_runtime else
                "Supplementary execution was technically skipped because the PI "
                "review did not provide an explicit no_action or supplement decision."
            )
            return await _decline(
                technical_status,
                technical_reason,
                review_plan=review_plan,
                followups=review_plan.get("followups") or [],
                technical_skip=True,
            )
        if not review_plan["supplementary_tasks"]:
            technical_status = "unavailable" if not has_runtime else "skipped"
            technical_reason = (
                "Supplementary execution was unavailable because no live analysis "
                "runtime is attached; the PI request and follow-up records were retained."
                if not has_runtime else
                "Supplementary execution was technically skipped because its configured "
                "task budget admitted no task; the PI request and follow-up records were retained."
            )
            return await _decline(
                technical_status,
                technical_reason,
                review_plan=review_plan,
                followups=review_plan.get("followups") or [],
                technical_skip=True,
            )

        # Reserve the single direction-level pass before invoking the restricted
        # executor. A plan/evidence-sensitive attempt id is retained for legacy
        # records, but cannot authorize a second pass after a revision or retry.
        reservation = {
            "review_attempt_id": attempt_id,
            "direction_pass_id": direction_pass_id,
            "direction_key": direction_key,
            "direction": direction,
            "lineage_id": lineage_id,
            "round_number": round_number,
            "lifecycle": "reserved",
            "lifecycle_status": "reserved",
            "review_status": "reserved",
            "review_reason": reason,
            "review_plan": review_plan,
            "followups": list(review_plan.get("followups") or []),
            "review_result": {},
            "plan_audit": audit_summary,
            "supplementary_findings": [],
            "evidence_records": [],
        }
        if direction_pass_id:
            self._store_review(analysis_state, reservation)
            if not await self._persist_review_state(analysis_state):
                reservation["review_status"] = "failed"
                reservation["lifecycle"] = "reservation_failed"
                reservation["lifecycle_status"] = "reservation_failed"
                reservation["review_reason"] = (
                    "The supplementary reservation could not be durably acknowledged; "
                    "automatic execution was skipped."
                )
                self._store_review(analysis_state, reservation)
                return reservation, empty_merge

        # 4) Restricted execution. Its own bounded loop, its own timeout; it cannot
        #    call report_findings, consult_pi or the publication-figure pipeline,
        #    so required visual deliverables remain owned by the main executor.
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
                source_context={
                    **dict(result.get("resolved_adata_source_context") or {}),
                    **{
                        key: value
                        for key, value in dict(session_names.get("analysis_source") or {}).items()
                        if key != "selection_error"
                    },
                },
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
        if not isinstance(assessment, dict):
            assessment = {
                "verdict": "partial",
                "summary": "",
                "resolved_gaps": [],
                "remaining_gaps": review_plan.get("gaps") or [],
            }

        raw_result_mapping = raw_result if isinstance(raw_result, dict) else {}
        known_execution_ids = self._known_review_execution_ids(
            raw_result_mapping, supplementary,
        )
        final_followups = self._merge_followup_assessment(
            list(review_plan.get("followups") or []),
            assessment,
            known_execution_ids=set(known_execution_ids),
        )
        final_followups = self._attach_shared_review_outcome(
            final_followups,
            supplementary=supplementary,
            known_execution_ids=known_execution_ids,
            attempt_id=attempt_id,
            direction_pass_id=direction_pass_id,
        )
        shared_outcome = self._shared_review_outcome(
            supplementary, known_execution_ids,
        )
        shared_reference = self._shared_review_reference(
            attempt_id, direction_pass_id, known_execution_ids,
        )
        merge_payload = {
            "findings": supplementary.get("findings", []),
            "finding_records": supplementary.get("finding_records", []),
            # Restricted supplementary work repairs computational evidence only.
            # Any visual gap is returned to the main reviewed figure pipeline.
            "figures": [],
            "code_history": raw_result_mapping.get("code_history") or [],
            "analysis_log": raw_result_mapping.get("analysis_log") or [],
            "evidence_records": supplementary.get("evidence_records", []),
            "assessment_summary": self._bounded_assessment_text(
                assessment.get("summary"), 4000,
            ) or "",
        }
        serialized_merge = self._serialize_review_record(merge_payload)

        meta = {
            "review_attempt_id": attempt_id,
            "direction_pass_id": direction_pass_id,
            "direction_key": direction_key,
            "direction": direction,
            "lineage_id": lineage_id,
            "review_status": supplementary.get("status", "failed"),
            "lifecycle": "completed" if supplementary.get("status") == "completed" else "failed",
            "lifecycle_status": "completed" if supplementary.get("status") == "completed" else "failed",
            "review_reason": reason,
            "review_plan": review_plan,
            "followups": final_followups,
            "review_result": {
                "status": supplementary.get("status"),
                "summary": supplementary.get("summary"),
                "warnings": supplementary.get("warnings", []),
                "evidence_records": supplementary.get("evidence_records", []),
                "finding_records": supplementary.get("finding_records", []),
                "active_object_selection": supplementary.get(
                    "active_object_selection", {}
                ),
            },
            "plan_audit": audit_summary,
            "supplementary_findings": supplementary.get("findings", []),
            "evidence_records": supplementary.get("evidence_records", []),
            "merge_payload": serialized_merge,
            "assessment": {
                "verdict": self._bounded_assessment_verdict(
                    assessment.get("verdict"),
                ),
                "summary": self._bounded_assessment_text(
                    assessment.get("summary"), 4000,
                ) or "",
                "resolved_gaps": self._bounded_assessment_items(
                    assessment.get("resolved_gaps"),
                ),
                "remaining_gaps": self._bounded_assessment_items(
                    assessment.get("remaining_gaps"),
                ),
                "execution_outcome": shared_outcome,
                "execution_reference": shared_reference,
            },
            "round_number": round_number,
        }
        self._store_review(analysis_state, meta)
        if direction_pass_id:
            if not await self._persist_review_state(analysis_state):
                # Keep every terminal field and the serialized replay payload in
                # the existing review record.  The caller must not merge paid
                # evidence until a later re-entry acknowledges this exact record.
                meta["terminal_lifecycle"] = meta["lifecycle"]
                meta["lifecycle"] = "terminal_persistence_pending"
                meta["lifecycle_status"] = "terminal_persistence_pending"
                meta["persistence_status"] = "pending"
                meta["persistence_failure"] = (
                    "Terminal supplementary review state could not be durably "
                    "acknowledged; the retained merge remains pending replay."
                )
                self._store_review(analysis_state, meta)
                logger.warning(
                    f"[{self.session_id}] Terminal supplementary review persistence "
                    "is pending; merge held for replay"
                )
                return meta, empty_merge
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
        from ..models.schemas import AnalysisStep, CodeExecution

        def _identity(value: Any) -> str:
            return str(getattr(value, "id", None) or (value.get("id") if isinstance(value, dict) else "") or "").strip()

        def _record_key(record: Any) -> Tuple[str, str, Tuple[str, ...]]:
            if isinstance(record, dict):
                return (
                    str(record.get("id") or ""),
                    str(record.get("text") or "").strip(),
                    tuple(sorted(str(item) for item in (record.get("evidence_execution_ids") or []))),
                )
            return (
                str(getattr(record, "id", "") or ""),
                str(getattr(record, "text", "") or "").strip(),
                tuple(sorted(str(item) for item in (getattr(record, "evidence_execution_ids", None) or []))),
            )

        existing_record_keys = {
            _record_key(record)
            for record in (result.get("finding_records") or [])
            if isinstance(record, dict)
        }
        tagged_records = []
        for record in (merge_payload.get("finding_records") or []):
            if not isinstance(record, dict) or not str(record.get("text", "")).strip():
                continue
            original_key = _record_key(record)
            tagged_record = {
                **record,
                "text": f"[Supplementary] {record.get('text', '')}",
            }
            key = _record_key(tagged_record)
            if key in existing_record_keys or original_key in existing_record_keys:
                continue
            existing_record_keys.add(key)
            tagged_records.append(tagged_record)
        tagged = [record["text"] for record in tagged_records]
        merged_findings = list(all_findings)
        for finding in (merge_payload.get("findings") or []):
            text = f"[Supplementary] {str(finding).strip()}"
            if text.strip() and text not in merged_findings:
                merged_findings.append(text)
        for finding in tagged:
            if finding not in merged_findings:
                merged_findings.append(finding)
        # Defense in depth for old checkpoints or mocked executors carrying the
        # pre-isolation payload shape: unreviewed supplementary images never join
        # the round's archival figure collection.
        merged_figures = list(all_figures)

        existing_code_ids = {_identity(c) for c in analysis_state.code_history if _identity(c)}
        added_executions = []
        replay_executions = []
        replay_ids: Set[str] = set()
        for execution in merge_payload.get("code_history") or []:
            if isinstance(execution, dict):
                try:
                    execution = CodeExecution(**execution)
                except Exception:
                    continue
            execution_id = _identity(execution)
            if not execution_id:
                continue
            if execution_id in replay_ids:
                continue
            replay_ids.add(execution_id)
            replay_executions.append(execution)
            if execution_id in existing_code_ids:
                continue
            analysis_state.code_history.append(execution)
            existing_code_ids.add(execution.id)
            added_executions.append(execution)

        existing_step_ids = {_identity(s) for s in analysis_state.analysis_log if _identity(s)}
        for step in merge_payload.get("analysis_log") or []:
            if isinstance(step, dict):
                try:
                    step = AnalysisStep(**step)
                except Exception:
                    continue
            if not _identity(step) or step.id in existing_step_ids:
                continue
            step.round_number = max(int(getattr(analysis_state, "current_round", 1) or 1), 1)
            analysis_state.analysis_log.append(step)
            existing_step_ids.add(step.id)

        # Keep the raw result consistent so the re-audit and any later reader see
        # the supplementary executions as part of this round.
        result["findings"] = merged_findings
        result["finding_records"] = list(result.get("finding_records") or []) + tagged_records
        result["figures"] = merged_figures
        if replay_executions:
            current_result_ids = {
                _identity(item) for item in (result.get("code_history") or []) if _identity(item)
            }
            result["code_history"] = list(result.get("code_history") or []) + [
                item for item in replay_executions if _identity(item) not in current_result_ids
            ]
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
            successful_ids = [
                _identity(e) for e in replay_executions if not getattr(e, "error", None)
            ]
            if successful_ids:
                existing_evidence_ids = list(result.get("evidence_execution_ids") or [])
                result["evidence_execution_ids"] = list(
                    dict.fromkeys(existing_evidence_ids + successful_ids)
                )

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

    async def _collect_session_names(
        self, analysis_state, result: Dict[str, Any], has_runtime: bool
    ) -> Dict[str, Any]:
        """The identifiers this session actually holds, for the review prompt.

        The PI is asked for tasks specific enough to execute unmodified, and is
        otherwise given only prose to write them from — so it names what a task
        of that shape would plausibly be called. Every such guess is an iteration
        of a budget in single digits. The names are already in `AnalysisState` and
        one kernel round-trip away; this assembles them.

        Never raises and never blocks the review: an empty dict renders as no
        block at all, which leaves the previous behaviour exactly as it was.
        """
        names: Dict[str, Any] = {}

        try:
            scores = [
                str(getattr(s, "score_name", "") or "")
                for s in (getattr(analysis_state, "score_definitions", None) or [])
            ]
            registered = [s for s in scores if s]
            if registered:
                # Reverse order: the scores this round registered are the ones a
                # repair task will reach for, and they are appended last.
                names["registered_scores"] = list(dict.fromkeys(registered[::-1]))
        except Exception as e:
            logger.debug(f"[{self.session_id}] score-name collection failed: {e}")

        try:
            artifacts = []
            for figure in (result.get("figures") or []):
                label = figure.get("filename") or figure.get("name") if isinstance(figure, dict) else figure
                if label:
                    artifacts.append(str(label))
            for table in (result.get("tables") or []):
                label = table.get("filename") or table.get("name") if isinstance(table, dict) else table
                if label:
                    artifacts.append(str(label))
            if artifacts:
                names["artifacts"] = list(dict.fromkeys(artifacts))
        except Exception as e:
            logger.debug(f"[{self.session_id}] artifact-name collection failed: {e}")

        if not has_runtime:
            return names

        try:
            source = dict(result.get("resolved_adata_source_context") or {})
            if not source and not getattr(getattr(analysis_state, "current_plan", None), "adata_source_preparation_id", None):
                source = dict(getattr(getattr(analysis_state, "annotation_state", None), "annotation_source", None) or {})
            metadata = await self.bioinformatics_agent.resolve_analysis_source_metadata(
                source, require_cluster_key=False,
            )
        except Exception as e:
            logger.debug(f"[{self.session_id}] object introspection for review failed: {e}")
            return names
        if isinstance(metadata, dict) and metadata:
            names["analysis_source"] = {
                key: metadata[key] for key in (
                    "var", "cluster_key", "partition_digest", "preparation_id",
                    "candidate_names", "selection_error",
                )
                if metadata.get(key)
            }
            for source_key, target_key in (
                ("obs_columns", "obs_columns"),
                ("layers", "layers"),
            ):
                values = [str(v) for v in (metadata.get(source_key) or [])]
                if values:
                    names[target_key] = values
            variable = str(metadata.get("var") or "").strip()
            if variable:
                names["variables"] = [variable]
        return names

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
