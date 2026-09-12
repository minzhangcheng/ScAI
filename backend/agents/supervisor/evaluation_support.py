"""Deterministic classification and ordered evaluation support."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

from ...models.schemas import AnalysisStep
from ...utils.error_parsing import (
    CODE_GEN_ERROR_TYPES, TECHNICAL_ERROR_TYPES, compact_traceback, extract_error_type,
)
from .shared import SupervisorEvalResult, SupervisorRecord, _CRITICAL_OUTPUT_PATTERNS

logger = logging.getLogger(__name__)

_SUPERVISOR_EVALUATION_TOOL_NAME = "submit_supervisor_evaluation"
_SUPERVISOR_EVALUATION_FIELDS = {
    "severity", "issues", "action", "message", "redirect_instruction",
    "issue_type", "is_biological_checkpoint", "pi_question",
}
_SUPERVISOR_REQUIRED_FIELDS = {"severity"}
_SUPERVISOR_OPTIONAL_DEFAULTS: Dict[str, Any] = {
    "issues": [], "action": "none", "message": "",
    "redirect_instruction": "", "issue_type": "ambiguous",
    "is_biological_checkpoint": False, "pi_question": "",
}
_SUPERVISOR_STRUCTURED_ATTEMPTS = 2
_MISSING_MODULE_PATTERN = re.compile(
    r"(?:ModuleNotFoundError|ImportError).*?No module named ['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)

def _error_fingerprint(error: Optional[str]) -> str:
    """Fingerprint the stable final exception line, not changing traceback frames."""
    if not error:
        return ""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    exception_lines = [
        line for line in lines
        if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):", line)
    ]
    final = exception_lines[-1] if exception_lines else (lines[-1] if lines else "")
    final = re.sub(r"Cell In\[\d+\]", "Cell In[]", final)
    final = re.sub(r"\bline\s+\d+\b", "line N", final, flags=re.IGNORECASE)
    return hashlib.sha256(final.encode("utf-8", errors="replace")).hexdigest()

def _platform_fault_reason(step: AnalysisStep) -> Optional[str]:
    """Recognize failures produced by ScAI's own execution wrappers."""
    error = str(step.code_error or "")
    output = str(step.code_output or "")
    if (
        "SyntaxError" in error
        and "_scai_expression_contract_call" in error
        and (
            "Generator expression must be parenthesized" in error
            or "f-string: unmatched" in error
        )
    ):
        return "ScAI expression-contract instrumentation generated invalid Python"
    if "Expression contract violation" in error and (
        re.search(r"integer[-_ ]like\s*[:=]\s*False", output, re.IGNORECASE)
        or re.search(
            r"noninteger[_ ]fraction\s*[:=]\s*(?:0?\.\d*[1-9]\d*|1(?:\.0+)?)",
            output,
            re.IGNORECASE,
        )
    ):
        return (
            "ScAI expression-contract validation rejected a matrix whose "
            "printed audit proves it is not integer-valued"
        )
    return None

async def evaluate_step(
    self,
    step: AnalysisStep,
    figures: List[str],
) -> SupervisorEvalResult:
    self.refresh_runtime_config_if_needed()
    self._step_count += 1
    record = SupervisorRecord(
        step_number=step.step_number,
        step_type=step.step_type,
        timestamp=step.timestamp,
        code=step.code[-2000:] if step.code else None,
        code_output=step.code_output[-2000:] if step.code_output else None,
        code_error=compact_traceback(step.code_error, max_chars=1200) if step.code_error else None,
        saved_figure_filenames=list(step.saved_figure_filenames),
        agent_reasoning=step.agent_reasoning if step.agent_reasoning else None,
        literature_topic=step.literature_topic,
        literature_result=step.literature_result[:500] if step.literature_result else None,
    )

    feedback_parts: List[str] = []

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

    _err_type = extract_error_type(step.code_error) if step.code_error else None
    platform_fault = _platform_fault_reason(step)
    current_fingerprint = _error_fingerprint(step.code_error)
    same_fingerprint_count = sum(
        _error_fingerprint(error) == current_fingerprint
        for error in self._recent_errors
    ) if current_fingerprint else 0
    if platform_fault:
        feedback_parts.append(
            "[Platform fault] " + platform_fault + ". Do not rebuild an "
            "equivalent object or spend another execution retry on this "
            "operation; preserve the checkpoint and release the step as "
            "unavailable pending a platform repair."
        )

    # Missing packages are an actionable environment observation. Ask once
    # per top-level package, and keep the guidance ecosystem-neutral.
    missing_package = self._missing_package_name(step.code_error)
    if missing_package:
        diagnosed = getattr(self, "_diagnosed_packages", None)
        if diagnosed is None:
            diagnosed = set()
            self._diagnosed_packages = diagnosed
        if missing_package not in diagnosed:
            diagnosed.add(missing_package)
            package_hint = await self._diagnose_missing_package(
                missing_package, step.code or ""
            )
            if package_hint:
                feedback_parts.append(package_hint)

    # Phase 1.5: Technique Agent for repeated API errors
    if not platform_fault and self._should_trigger_technique(step):
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
    is_stuck = (
        bool(platform_fault)
        or same_fingerprint_count >= 2
    )
    should_llm_evaluate = (
        (bool(step.code_error) and (not self._is_technical_error(step.code_error) or is_stuck))
        or bool(figures)                          # Has figure output
        or step.step_type == "annotate_clusters"  # Critical biological decision
        or self._step_count % 5 == 0              # Sample every 5 steps
    )

    llm_result = None
    if platform_fault:
        logger.warning(
            "[%s] Step %s classified as platform fault: %s",
            self.session_id,
            step.step_number,
            platform_fault,
        )
    elif not should_llm_evaluate:
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
                    "The platform will render and verify the required Stage 1 baseline "
                    "figures after this pass. The agent should call report_findings now; "
                    "Stage 2 downstream analysis will not begin unless the baseline "
                    "visual contract is satisfied."
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
    # meaning a prior same-fingerprint executor turn received guidance and
    # still failed. Current-step guidance is not enough to skip work.
    prior_guidance = False
    if step.code_error and is_stuck:
        fingerprint = self._error_fingerprint(step.code_error)
        prior_guidance = any(
            record.step_number < step.step_number
            and record.pi_escalated
            and self._error_fingerprint(record.code_error) == fingerprint
            for record in self.supervision_log
        )
    return SupervisorEvalResult(
        feedback=feedback,
        force_skip=bool(platform_fault) or (is_stuck and prior_guidance),
        failure_class=(
            "platform_fault" if platform_fault
            else ("repeated_execution_failure" if is_stuck else None)
        ),
    )

def _rule_check(self, step: AnalysisStep) -> List[str]:
    """Run instant rule-based checks on a step. Returns list of issue descriptions."""
    issues = []

    # Code error
    if step.code_error:
        err_snippet = compact_traceback(step.code_error, max_chars=600)
        issues.append(f"Code error: {err_snippet}")

        # Track recent errors for stuck detection.
        # Keep the TAIL of the traceback (error type appears at the end).
        self._recent_errors.append(
            compact_traceback(step.code_error, max_chars=800)
        )
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

def _validate_supervisor_result(
    value: Any,
) -> tuple[Optional[Dict[str, Any]], str]:
    """Validate provider output against the closed evaluation schema."""
    if not isinstance(value, dict):
        return None, "verdict is not an object"
    keys = set(value)
    missing = _SUPERVISOR_REQUIRED_FIELDS - keys
    extra = keys - _SUPERVISOR_EVALUATION_FIELDS
    if missing:
        return None, f"missing fields: {', '.join(sorted(missing))}"
    if extra:
        return None, f"unexpected fields: {', '.join(sorted(extra))}"

    normalized_value = dict(value)
    for field, default in _SUPERVISOR_OPTIONAL_DEFAULTS.items():
        if normalized_value.get(field) is None:
            normalized_value[field] = list(default) if isinstance(default, list) else default
    value = normalized_value

    # Provider tool calls occasionally return a lone string instead of the
    # schema's single-item list. Accept that one mechanical shape mismatch, trim
    # and bound the entries, but leave every non-string-typed value for the
    # strict checks below.
    issues = value["issues"]
    if isinstance(issues, str):
        issues = [issues]
    if isinstance(issues, list) and all(isinstance(item, str) for item in issues):
        normalized_issues: List[str] = []
        for item in issues:
            item = item.strip()
            if not item:
                continue
            normalized_issues.append(item[:2000])
            if len(normalized_issues) >= 20:
                break
        issues = normalized_issues
    value["issues"] = issues

    enums = {
        "severity": {"none", "info", "warning", "critical"},
        "action": {"none", "warn", "redirect"},
        "issue_type": {"technical", "biological", "ambiguous"},
    }
    issues = value["issues"]
    if (
        not isinstance(issues, list)
        or len(issues) > 20
        or any(not isinstance(item, str) or len(item) > 2000 for item in issues)
    ):
        return None, "issues must be a bounded list of strings"

    normalized_enums: Dict[str, str] = {}
    for field, allowed in enums.items():
        item = value[field]
        if not isinstance(item, str):
            return None, f"invalid {field}"
        normalized = item.strip().casefold()
        if (
            field == "issue_type"
            and normalized in {"", "none", "n/a", "null"}
            and not issues
            and normalized_enums.get("severity") in {"none", "info"}
        ):
            normalized = "ambiguous"
        if normalized not in allowed:
            return None, f"invalid {field}"
        normalized_enums[field] = normalized

    for field in ("message", "redirect_instruction", "pi_question"):
        item = value[field]
        if item is None:
            item = ""
        if not isinstance(item, str) or len(item) > 4000:
            return None, f"invalid {field}"
    if not isinstance(value["is_biological_checkpoint"], bool):
        return None, "is_biological_checkpoint must be a boolean"

    # Copy the validated fields so callers never retain a mutable mapping
    # owned by an SDK/provider response object.
    return {
        "severity": normalized_enums["severity"],
        "issues": list(issues),
        "action": normalized_enums["action"],
        "message": value["message"] or "",
        "redirect_instruction": value["redirect_instruction"] or "",
        "issue_type": normalized_enums["issue_type"],
        "is_biological_checkpoint": value["is_biological_checkpoint"],
        "pi_question": value["pi_question"] or "",
    }, ""

def _parse_supervisor_response(
    text: str,
    *,
    warn: bool = True,
) -> Dict[str, Any]:
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
        validated, error = _validate_supervisor_result(parsed)
        if validated is not None:
            return validated
        if isinstance(parsed, dict):
            logger.debug("Supervisor response: direct JSON schema failed: %s", error)
        else:
            logger.debug(
                "Supervisor response: parsed as %s, not an object",
                type(parsed).__name__,
            )
    except (json.JSONDecodeError, TypeError):
        logger.debug("Supervisor response: direct JSON parse failed")
    # Try extracting from markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            validated, error = _validate_supervisor_result(parsed)
            if validated is not None:
                return validated
            logger.debug(
                "Supervisor response: markdown JSON schema failed: %s", error
            )
        except (json.JSONDecodeError, TypeError):
            logger.debug("Supervisor response: markdown code block JSON parse failed")
    # Try finding any JSON object in the text
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            validated, error = _validate_supervisor_result(parsed)
            if validated is not None:
                return validated
            logger.debug(
                "Supervisor response: extracted JSON schema failed: %s", error
            )
        except (json.JSONDecodeError, TypeError):
            logger.debug("Supervisor response: regex-extracted JSON parse failed")
    # Default: no intervention, but say so out loud
    snippet = (text or "").strip().replace("\n", " ")[:200]
    if warn:
        logger.warning(
            "Supervisor response could not be parsed as schema-valid JSON; "
            "treating the step as un-evaluated rather than clean. Response "
            "began: %r",
            snippet or "<empty>",
        )
    return {
        "severity": "none", "issues": [], "action": "none", "message": "",
        "redirect_instruction": "",
        "is_biological_checkpoint": False, "pi_question": "",
        "issue_type": "ambiguous", "parse_failed": True,
    }

def _is_technical_error(code_error: Optional[str]) -> bool:
    """Return True if the error is a pure Python/technical exception (no PI help needed)."""
    if not code_error:
        return False
    err_type = extract_error_type(code_error)
    return err_type in TECHNICAL_ERROR_TYPES

def _missing_package_name(code_error: Optional[str]) -> Optional[str]:
    if not code_error:
        return None
    match = _MISSING_MODULE_PATTERN.search(str(code_error))
    if not match:
        return None
    return match.group(1).split(".", 1)[0].strip() or None

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
