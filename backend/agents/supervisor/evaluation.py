"""Rule-based and LLM evaluation of bioinformatics steps, plus PI escalation.

Moved verbatim out of backend/agents/supervisor_agent.py; composed into SupervisorAgent.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import logging
import re
import hashlib

from ...constants import REASONING_MODEL_MIN_TOKENS

logger = logging.getLogger(__name__)


from ...models.schemas import AnalysisStep, PlanStage
from ...utils.error_parsing import CODE_GEN_ERROR_TYPES, extract_error_type, TECHNICAL_ERROR_TYPES


# ---------------------------------------------------------------------------
# Rich HTML report CSS (extracted to utils/report_css.py for separation of concerns)
# ---------------------------------------------------------------------------
from ...utils.report_css import REPORT_CSS as _REPORT_CSS  # noqa: E402

from .shared import (  # noqa: F401
    _SPECIFIC_SECTION_IDS,
    _SECTION_ID_TO_DEFAULT_TITLE,
    _PLACEHOLDER_SECTION_TITLES,
    SupervisorRecord,
    SupervisorEvalResult,
    _CRITICAL_OUTPUT_PATTERNS,
    _PERCENTAGE_PATTERN,
    _INF_NAN_PATTERN,
    _SUPERVISOR_SYSTEM_PROMPT,
    _serialize_record,
    persist_graphic_abstract,
)


class EvaluationMixin:
    """Rule-based and LLM evaluation of bioinformatics steps, plus PI escalation."""

    async def evaluate_step(
        self,
        step: AnalysisStep,
        figures: List[str],
    ) -> SupervisorEvalResult:
        """
        Evaluate a single analysis step. Called after every tool-call iteration.

        Args:
            step: The latest AnalysisStep from analysis_log
            figures: List of base64-encoded PNG images generated in this step (may be empty)

        Returns:
            SupervisorEvalResult with feedback string and force_skip flag.
        """
        self.refresh_runtime_config_if_needed()
        self._step_count += 1
        record = SupervisorRecord(
            step_number=step.step_number,
            step_type=step.step_type,
            timestamp=step.timestamp,
            code=step.code[-2000:] if step.code else None,
            code_output=step.code_output[-2000:] if step.code_output else None,
            code_error=step.code_error[:400] if step.code_error else None,
            saved_figure_filenames=list(step.saved_figure_filenames),
            agent_reasoning=step.agent_reasoning if step.agent_reasoning else None,
            literature_topic=step.literature_topic,
            literature_result=step.literature_result[:500] if step.literature_result else None,
        )

        feedback_parts: List[str] = []

        # Phase 1: Rule-based checks (no LLM, instant)
        rule_issues = self._rule_check(step)
        if rule_issues:
            record.technical_issues.extend(rule_issues)

            # Skip injecting feedback for first-occurrence technical errors:
            # the agent already sees the full traceback from the tool result
            # and should attempt self-correction before supervisor intervenes.
            is_first_technical_error = (
                step.code_error
                and self._is_technical_error(step.code_error)
                and len(self._recent_errors) < 2
            )

            if not is_first_technical_error:
                # De-duplicate rule feedback to avoid repeating the same message
                for issue in rule_issues:
                    issue_hash = hash(issue[:100])
                    if issue_hash not in self._recent_rule_feedback_hashes:
                        feedback_parts.append(issue)
                        self._recent_rule_feedback_hashes.add(issue_hash)

        # Resolve error type once — used by Phase 1.5 and Phase 2 below
        _err_type = extract_error_type(step.code_error) if step.code_error else None

        # Phase 1.5: Technique Agent for repeated API errors
        if self._should_trigger_technique(step):
            try:
                # Syntax/gen errors get a static hint; API doc lookup is irrelevant.
                if _err_type in CODE_GEN_ERROR_TYPES:
                    logger.info(
                        f"[{self.session_id}] Triggering Technique Agent for repeated "
                        f"{_err_type} error at step {step.step_number}"
                    )
                    feedback_parts.append(
                        f"[Technique Support] Repeated syntax error ({_err_type}) detected. "
                        f"Check for: (1) unclosed parentheses, brackets, or quotes; "
                        f"(2) newline characters inside single/double-quoted strings — "
                        f"use triple quotes for multi-line strings; "
                        f"(3) mixed tabs and spaces for indentation. "
                        f"Simplify: break complex code into smaller cells and test incrementally."
                    )
                else:
                    logger.info(
                        f"[{self.session_id}] Triggering Technique Agent for repeated "
                        f"{_err_type} error at step {step.step_number}"
                    )
                    await self._notify("status", "Looking up API documentation...")
                    doc_hint = await self.technique_agent.lookup_from_error(
                        self._sandbox,
                        step.code_error,
                        step.code or "",
                    )
                    if doc_hint:
                        feedback_parts.append(
                            f"[Technique Support] Repeated API error detected. "
                            f"Here is the correct API documentation:\n{doc_hint}"
                        )
            except Exception as e:
                logger.warning(f"[{self.session_id}] Technique Agent lookup failed: {e}")

        # Phase 2: LLM evaluation (sampled to reduce cost)
        # Skip LLM for first-occurrence technical errors — the agent can self-correct.
        # Only invoke LLM when the agent appears stuck (consecutive same-type errors)
        # or the error is non-technical (potential data/biological issue).
        # Detect stuck: same error type appearing >= 3 times in recent history,
        # OR two consecutive identical types (original check).
        # (_err_type is hoisted above Phase 1.5 — no redundant re-parse here.)
        _same_type_count = sum(
            1 for e in self._recent_errors
            if extract_error_type(e) == _err_type
        ) if _err_type else 0
        is_stuck = (
            _same_type_count >= 3
            or (
                len(self._recent_errors) >= 2
                and extract_error_type(self._recent_errors[-1])
                    == extract_error_type(self._recent_errors[-2])
            )
        )
        should_llm_evaluate = (
            (bool(step.code_error) and (not self._is_technical_error(step.code_error) or is_stuck))
            or bool(figures)                          # Has figure output
            or step.step_type == "annotate_clusters"  # Critical biological decision
            or self._step_count % 5 == 0              # Sample every 5 steps
        )

        llm_result = None
        if not should_llm_evaluate:
            logger.debug(f"[{self.session_id}] Skipping LLM evaluation for step {step.step_number} (sampling)")
        else:
            try:
                llm_result = await self._llm_evaluate(step, figures)
            except Exception as e:
                logger.warning(f"[{self.session_id}] Supervisor LLM evaluation failed: {e}")

        if llm_result:
            if llm_result.get("parse_failed"):
                # The no-op verdict below is a parsing artefact, not a finding.
                # Logged loudly, but deliberately NOT appended to
                # record.technical_issues: that list feeds
                # steps_with_technical_issues, which the detailed report prints
                # to the reader as "found N technical issues" — a supervisor
                # that could not be parsed is not a defect in the analysis, and
                # counting it as one misinforms the reader about their results.
                logger.warning(
                    f"[{self.session_id}] Step {step.step_number} was not evaluated: "
                    f"supervisor response was not parseable JSON"
                )
            record.llm_severity = llm_result.get("severity", "none")
            llm_issues = llm_result.get("issues", [])
            if llm_issues:
                record.technical_issues.extend(llm_issues)

            severity = llm_result.get("severity", "none")
            message = llm_result.get("message", "")

            # Critical → inject feedback + conditionally escalate to PI
            if severity == "critical":
                if message:
                    feedback_parts.append(f"CRITICAL: {message}")
                redirect = llm_result.get("redirect_instruction", "")
                if redirect:
                    feedback_parts.append(f"Suggested action: {redirect}")
                # Only escalate to PI for non-technical errors — unless the agent
                # is stuck (same error type repeated >= 3 times), in which case
                # PI should be asked even for technical errors.
                # Layer 1: deterministic check on Python exception type
                if self._is_technical_error(step.code_error) and not is_stuck:
                    logger.info(
                        f"[{self.session_id}] Skipping PI escalation for technical error "
                        f"(exception type) at step {step.step_number}"
                    )
                # Layer 2: LLM classification catches logic errors with no exception
                # (e.g. blank plot due to wrong column name, code ran without error)
                elif llm_result.get("issue_type", "ambiguous") == "technical" and not is_stuck:
                    logger.info(
                        f"[{self.session_id}] Skipping PI escalation for technical issue "
                        f"(LLM classified) at step {step.step_number}"
                    )
                else:
                    pi_resp = await self._escalate_to_pi(
                        f"Critical issue at step {step.step_number}: {message}",
                        self._build_step_context(step),
                    )
                    if pi_resp:
                        record.pi_escalated = True
                        record.pi_response = pi_resp
                        feedback_parts.append(f"PI guidance: {pi_resp}")

            elif severity == "warning" and message:
                feedback_parts.append(message)

            # Biological checkpoint check
            is_checkpoint = llm_result.get("is_biological_checkpoint", False)
            pi_question = llm_result.get("pi_question", "")
            if is_checkpoint and pi_question:
                record.checkpoint_type = "llm_detected"
                pi_resp = await self._escalate_to_pi(
                    pi_question,
                    self._build_checkpoint_context(step),
                )
                if pi_resp:
                    record.pi_escalated = True
                    record.pi_response = pi_resp
                    feedback_parts.append(f"PI guidance: {pi_resp}")

        # Hard-coded: annotate_clusters escalates to PI only on first occurrence per round
        if step.step_type == "annotate_clusters" and not record.pi_escalated:
            already_annotated = any(
                r.step_type == "annotate_clusters" and r.pi_escalated
                for r in self.supervision_log
            )
            if not already_annotated:
                record.checkpoint_type = "annotate_clusters"
                if self.planning_stage == "stage1":
                    escalation_msg = (
                        "Cell type annotation has been completed. "
                        "Please confirm the annotations are biologically reasonable. "
                        "Stage 1 is now complete — the agent should call report_findings "
                        "to finalize Stage 1 before Stage 2 downstream analysis begins."
                    )
                else:
                    escalation_msg = (
                        "Cell type annotation has been completed. Please review the annotation results "
                        "and provide guidance on whether they are biologically reasonable."
                    )
                pi_resp = await self._escalate_to_pi(
                    escalation_msg,
                    self._build_checkpoint_context(step),
                )
                if pi_resp:
                    record.pi_escalated = True
                    record.pi_response = pi_resp
                    feedback_parts.append(f"PI guidance: {pi_resp}")

        # If the agent is stuck (same error type repeated), escalate to PI
        # regardless of LLM severity classification — the cheap supervisor LLM
        # may underrate repeated technical errors as "warning" when PI-level
        # strategic guidance is needed to break the cycle.
        if is_stuck and step.code_error and not record.pi_escalated:
            logger.info(
                f"[{self.session_id}] Agent stuck (same error type repeated), "
                f"escalating to PI at step {step.step_number}"
            )
            await self._notify("status", "Agent stuck — consulting PI for strategic guidance...")
            stuck_question = (
                f"Agent is stuck: the same error type ({_err_type or 'unknown'}) "
                f"has occurred repeatedly at step {step.step_number}. "
                f"The agent has attempted self-correction but keeps failing. "
                f"Please provide strategic guidance: should the agent try a completely "
                f"different approach, simplify the analysis, or skip this step?"
            )
            pi_resp = await self._escalate_to_pi(
                stuck_question,
                self._build_step_context(step),
            )
            if pi_resp:
                record.pi_escalated = True
                record.pi_response = pi_resp
                feedback_parts.append(f"[PI stuck intervention] {pi_resp}")

        # Assemble final feedback
        feedback = None
        if feedback_parts:
            feedback = "\n".join(feedback_parts)
            record.feedback_injected = feedback[:1000]
            await self._notify("supervisor_feedback", feedback[:500])

        self.supervision_log.append(record)
        # force_skip: agent is stuck AND PI already intervened on this step,
        # meaning the LLM has received strategic guidance but still cannot fix it.
        return SupervisorEvalResult(
            feedback=feedback,
            force_skip=is_stuck and record.pi_escalated,
        )

    async def _llm_evaluate(
        self, step: AnalysisStep, figures: List[str]
    ) -> Optional[Dict]:
        """Call cheap LLM to evaluate the step. Uses vision when figures are available."""
        # Build text context
        parts = [f"Step {step.step_number} | Type: {step.step_type}"]
        if step.agent_reasoning:
            parts.append(f"Agent reasoning: {step.agent_reasoning}")
        if step.code:
            if len(step.code) > 2000:
                parts.append(f"Code (last 2000 chars of {len(step.code)}):\n```python\n{step.code[-2000:]}\n```")
            else:
                parts.append(f"Code:\n```python\n{step.code}\n```")
        if step.code_output:
            if len(step.code_output) > 2000:
                parts.append(f"Output (last 2000 chars of {len(step.code_output)}):\n{step.code_output[-2000:]}")
            else:
                parts.append(f"Output:\n{step.code_output}")
        if step.code_error:
            parts.append(f"Error:\n{step.code_error[:400]}")
        if step.step_type == "execute_code":
            if step.code_error:
                parts.append("Execution status: FAILED (see Error above)")
            elif step.code_output is not None:
                parts.append("Execution status: SUCCESS")
            else:
                parts.append("Execution status: No output recorded")
        if step.literature_topic:
            parts.append(f"Literature search topic: {step.literature_topic}")
        if step.literature_result:
            parts.append(f"Literature result: {step.literature_result[:300]}")
        if step.annotation_result_summary:
            parts.append(f"Annotation result: {step.annotation_result_summary}")
        if step.findings:
            parts.append(f"Findings: {'; '.join(step.findings[:3])}")

        text_context = "\n\n".join(parts)

        # Build effective system prompt: user-custom prompt prepended to built-in default
        effective_prompt = (
            f"{self.system_prompt}\n\n{_SUPERVISOR_SYSTEM_PROMPT}"
            if self.system_prompt
            else _SUPERVISOR_SYSTEM_PROMPT
        )

        # If we have figures and vision is enabled, try vision evaluation
        if figures and self.has_vision:
            try:
                compressed = []
                for fig_b64 in figures[:3]:  # Max 3 figures per evaluation
                    compressed.append(self._compress_base64_figure(fig_b64))

                n_figs = len(compressed)
                # This note sits immediately before the images and is what the
                # model actually acts on, so it restates the full image
                # checklist rather than a shortened one — an abbreviated
                # reminder in the highest-recency position quietly narrows the
                # system prompt's criteria.
                text_with_fig_note = (
                    f"{text_context}\n\n"
                    f"[{n_figs} figure(s) attached for visual inspection. "
                    f"Check for blank/empty plots, missing or nonsensical labels, "
                    f"text or legends drawn on top of other text, labels that run "
                    f"into each other, fonts too small to read, and obviously wrong "
                    f"visualizations.]"
                )

                self._set_llm_context()
                response = await self.llm_client.chat_with_vision(
                    model=self.model,
                    text_prompt=f"{effective_prompt}\n\n{text_with_fig_note}",
                    images=compressed,
                    temperature=self.temperature,
                    max_tokens=REASONING_MODEL_MIN_TOKENS,  # reasoning models need headroom or content comes back empty
                )
                return self._parse_supervisor_response(response.content if response else "")

            except Exception as e:
                logger.warning(
                    f"[{self.session_id}] Vision evaluation failed, falling back to text-only: {e}"
                )
                # Fall through to text-only evaluation

        # Text-only evaluation
        self._set_llm_context()
        response = await self.llm_client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": effective_prompt},
                {"role": "user", "content": text_context},
            ],
            temperature=self.temperature,
            max_tokens=REASONING_MODEL_MIN_TOKENS,  # reasoning models need headroom or content comes back empty
        )
        return self._parse_supervisor_response(response.content if response else "")

    def _rule_check(self, step: AnalysisStep) -> List[str]:
        """Run instant rule-based checks on a step. Returns list of issue descriptions."""
        issues = []

        # Code error
        if step.code_error:
            err_snippet = step.code_error[:200]
            issues.append(f"Code error: {err_snippet}")

            # Track recent errors for stuck detection.
            # Keep the TAIL of the traceback (error type appears at the end).
            self._recent_errors.append(step.code_error[-300:])
            if len(self._recent_errors) > 5:
                self._recent_errors = self._recent_errors[-5:]

            # Check for 2 consecutive errors of the same type
            if len(self._recent_errors) >= 2:
                last_2 = self._recent_errors[-2:]
                error_types = [extract_error_type(e) for e in last_2]
                if all(e == last_2[0] for e in last_2):
                    issues.append(
                        "WARNING: Agent appears stuck — same error occurred 2 times in a row. "
                        "Consider a different approach."
                    )
                elif all(t == error_types[0] for t in error_types):
                    issues.append(
                        f"WARNING: Agent appears stuck — same error type ({error_types[0]}) "
                        "occurred 2 times in a row. Consider a completely different approach "
                        "or skip this analysis step."
                    )
        else:
            # Reset error streak and feedback dedup on success
            if self._recent_errors:
                self._recent_errors = []
                self._recent_rule_feedback_hashes.clear()

        # Check output patterns
        if step.code_output:
            output = step.code_output
            for pat in _CRITICAL_OUTPUT_PATTERNS:
                m = pat.search(output)
                if m:
                    if "n_obs" in pat.pattern:
                        try:
                            n = int(m.group(1))
                            if n < 50:
                                code_lower = (step.code or "").lower()
                                is_bulk_analysis = ("bulk" in code_lower
                                                    or "merged_counts" in code_lower
                                                    or "pydeseq" in code_lower
                                                    or "deseqdataset" in code_lower)
                                if not is_bulk_analysis:
                                    issues.append(f"Very low cell count: n_obs={n}")
                        except (ValueError, IndexError):
                            pass
                    elif "0" in pat.pattern and "cells" in pat.pattern.lower():
                        issues.append("Zero cells detected in output")
                    elif "Empty DataFrame" in pat.pattern:
                        issues.append("Empty DataFrame in output")
                    elif "MemoryError" in pat.pattern:
                        issues.append("MemoryError detected")
                    elif "obs_names" in pat.pattern.lower():
                        issues.append("obs_names are not unique — call adata.obs_names_make_unique()")
                    elif "InvalidIndexError" in pat.pattern:
                        issues.append("InvalidIndexError — likely caused by duplicate obs_names")

        # Cell-type annotation is a mandatory biological checkpoint. If the
        # annotate_clusters tool returned an error summary, surface it as a
        # high-severity issue so the agent is redirected to retry instead of
        # silently pivoting to other work.
        if step.step_type == "annotate_clusters":
            summary = step.annotation_result_summary or ""
            if summary.lstrip().lower().startswith("error"):
                issues.append(
                    "annotate_clusters failed — user-facing cell type review was not "
                    "triggered. You MUST fix the underlying cause (ensure a clustering "
                    "column and rank_genes_groups exist, or pass cluster_key/rank_key "
                    "explicitly) and call annotate_clusters again before any "
                    "report_findings or downstream analysis."
                )

        return issues

    @staticmethod
    def _parse_supervisor_response(text: str) -> Dict:
        """Parse JSON from LLM response with multi-level fallback.

        The fallback is a no-op verdict, which is indistinguishable from "the
        supervisor looked and found nothing wrong". They are not the same thing,
        so an exhausted fallback is logged at WARNING and the returned verdict
        carries ``parse_failed`` — a silent unparseable response reads in every
        downstream view as a clean bill of health.
        """
        # Try direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            logger.debug(
                "Supervisor response: parsed as %s, not an object", type(parsed).__name__
            )
        except (json.JSONDecodeError, TypeError):
            logger.debug("Supervisor response: direct JSON parse failed")
        # Try extracting from markdown code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                logger.debug("Supervisor response: markdown code block JSON parse failed")
        # Try finding any JSON object in the text
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                logger.debug("Supervisor response: regex-extracted JSON parse failed")
        # Default: no intervention, but say so out loud
        snippet = (text or "").strip().replace("\n", " ")[:200]
        logger.warning(
            "Supervisor response could not be parsed as JSON; treating the step as "
            "un-evaluated rather than clean. Response began: %r",
            snippet or "<empty>",
        )
        return {
            "severity": "none", "issues": [], "action": "none", "message": "",
            "is_biological_checkpoint": False, "pi_question": "",
            "issue_type": "ambiguous", "parse_failed": True,
        }

    @staticmethod
    def _is_technical_error(code_error: Optional[str]) -> bool:
        """Return True if the error is a pure Python/technical exception (no PI help needed)."""
        if not code_error:
            return False
        err_type = extract_error_type(code_error)
        return err_type in TECHNICAL_ERROR_TYPES

    async def _escalate_to_pi(self, question: str, context: str) -> Optional[str]:
        """Escalate a question to PI, with deduplication."""
        if not self.pi_callback:
            return None

        # De-duplicate: skip if we already asked the same question
        q_hash = hashlib.md5(question.encode()).hexdigest()
        if q_hash in self._pi_question_hashes:
            return None
        self._pi_question_hashes.add(q_hash)

        try:
            await self._notify("supervisor_checkpoint", question[:200])
            response = await self.pi_callback(question, context)
            return response
        except Exception as e:
            logger.warning(f"[{self.session_id}] PI escalation failed: {e}")
            return None

    def _should_trigger_technique(self, step: AnalysisStep) -> bool:
        """
        Check if we should trigger the Technique Agent for documentation lookup.

        Returns True when:
        1. technique_agent and sandbox are available
        2. Current step has a code error
        3. Previous step had the same type of error (consecutive same-type)
        """
        if not self.technique_agent or not self._sandbox:
            return False
        if not step.code_error:
            return False

        # Trigger on any consecutive same-type error (not just API errors)
        # so the Technique Agent can look up relevant docs for repeated failures
        if len(self._recent_errors) < 2:
            return False
        current_type = extract_error_type(step.code_error)
        prev_type = extract_error_type(self._recent_errors[-2])
        return current_type == prev_type

    def reset_for_round(self, planning_stage: PlanStage = "full"):
        """Archive current round log and reset for a new round."""
        if self.supervision_log:
            self.all_rounds_log.append(self.supervision_log.copy())
        self.supervision_log = []
        self._step_count = 0
        self._recent_errors = []
        self._pi_question_hashes = set()
        self.planning_stage = planning_stage

    def get_summary_stats(self) -> Dict[str, Any]:
        """Return summary statistics for the current analysis."""
        all_records = []
        for round_log in self.all_rounds_log:
            all_records.extend(round_log)
        all_records.extend(self.supervision_log)

        total = len(all_records)
        with_issues = sum(1 for r in all_records if r.technical_issues)
        pi_escalations = sum(1 for r in all_records if r.pi_escalated)
        critical = sum(1 for r in all_records if r.llm_severity == "critical")

        # Build checkpoints with round identifiers
        checkpoints_reached = []
        for round_idx, round_log in enumerate(self.all_rounds_log):
            for r in round_log:
                if r.checkpoint_type:
                    checkpoints_reached.append({"step": r.step_number, "type": r.checkpoint_type, "round": round_idx + 1})
        current_round = len(self.all_rounds_log) + 1
        for r in self.supervision_log:
            if r.checkpoint_type:
                checkpoints_reached.append({"step": r.step_number, "type": r.checkpoint_type, "round": current_round})

        return {
            "total_steps_monitored": total,
            "steps_with_technical_issues": with_issues,
            "pi_escalations": pi_escalations,
            "checkpoints_reached": checkpoints_reached,
            "critical_issues": critical,
            "rounds_completed": len(self.all_rounds_log),
        }

    def get_supervision_log(self) -> List[Dict]:
        """Return serializable log of all rounds + current round, with round_number."""
        result = []
        for round_idx, round_log in enumerate(self.all_rounds_log):
            for r in round_log:
                result.append(_serialize_record(r, round_idx + 1))
        current_round = len(self.all_rounds_log) + 1
        for r in self.supervision_log:
            result.append(_serialize_record(r, current_round))
        return result

    def restore_log_from_saved(self, records: List[Dict]):
        """Restore supervision log from saved analysis_log dicts, grouping by round_number.
        
        Records with round_number == max(round_number) go to supervision_log (current round).
        All other rounds are grouped into all_rounds_log.
        Old format records without round_number default to round 1.
        Empty records list short-circuits to empty logs.
        """
        if not records:
            self.all_rounds_log = []
            self.supervision_log = []
            return

        max_round = max(r.get("round_number", 1) for r in records)

        rounds: Dict[int, List[SupervisorRecord]] = {}
        for d in records:
            ts = d.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(ts) if ts else datetime.now()
            except (ValueError, TypeError):
                timestamp = datetime.now()
            record = SupervisorRecord(
                step_number=d.get("step_number", 0),
                step_type=d.get("step_type", ""),
                timestamp=timestamp,
                technical_issues=d.get("technical_issues", []),
                checkpoint_type=d.get("checkpoint_type"),
                llm_severity=d.get("llm_severity"),
                pi_escalated=d.get("pi_escalated", False),
                pi_response=d.get("pi_response"),
                feedback_injected=d.get("feedback_injected"),
                literature_topic=d.get("literature_topic"),
                literature_result=d.get("literature_result"),
                code=d.get("code"),
                code_output=d.get("code_output"),
                code_error=d.get("code_error"),
                saved_figure_filenames=d.get("saved_figure_filenames", []),
                agent_reasoning=d.get("agent_reasoning"),
            )
            rn = d.get("round_number", 1)
            rounds.setdefault(rn, []).append(record)

        self.all_rounds_log = []
        self.supervision_log = []
        for rn in sorted(rounds.keys()):
            if rn == max_round:
                self.supervision_log = rounds[rn]
            else:
                self.all_rounds_log.append(rounds[rn])

        logger.info(f"Restored {len(records)} records: {len(self.all_rounds_log)} completed rounds, "
                     f"{len(self.supervision_log)} current round records (round {max_round})")
