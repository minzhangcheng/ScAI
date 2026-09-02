"""Rule-based and LLM evaluation of bioinformatics steps, plus PI escalation.

Moved verbatim out of backend/agents/supervisor_agent.py; composed into SupervisorAgent.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import logging
import re
import hashlib

from ...constants import NO_OUTPUT_TOKEN_CAP

logger = logging.getLogger(__name__)


from ...models.schemas import AnalysisStep, PlanStage
from ...utils.error_parsing import (
    CODE_GEN_ERROR_TYPES,
    compact_traceback,
    extract_error_type,
    TECHNICAL_ERROR_TYPES,
)


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


from . import evaluation_state, evaluation_support
from .evaluation_support import (
    _SUPERVISOR_EVALUATION_FIELDS, _SUPERVISOR_EVALUATION_TOOL_NAME,
    _SUPERVISOR_OPTIONAL_DEFAULTS, _SUPERVISOR_REQUIRED_FIELDS,
    _SUPERVISOR_STRUCTURED_ATTEMPTS, _error_fingerprint, _platform_fault_reason,
)

_SUPERVISOR_EVALUATION_TOOL = {
    "type": "function",
    "function": {
        "name": _SUPERVISOR_EVALUATION_TOOL_NAME,
        "description": "Submit the validated supervisor verdict for one analysis step.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["none", "info", "warning", "critical"],
                },
                "issues": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 2000},
                    "maxItems": 20,
                },
                "action": {
                    "type": "string",
                    "enum": ["none", "warn", "redirect"],
                },
                "message": {"type": "string", "maxLength": 4000},
                "redirect_instruction": {"type": "string", "maxLength": 4000},
                "issue_type": {
                    "type": "string",
                    "enum": ["technical", "biological", "ambiguous"],
                },
                "is_biological_checkpoint": {"type": "boolean"},
                "pi_question": {"type": "string", "maxLength": 4000},
            },
            "required": sorted(_SUPERVISOR_REQUIRED_FIELDS),
        },
    },
}






from ..vision_replay import replay_safe_vision_call


class EvaluationMixin:
    """Rule-based and LLM evaluation of bioinformatics steps, plus PI escalation."""

    async def evaluate_step(
        self,
        step: AnalysisStep,
        figures: List[str],
    ) -> SupervisorEvalResult:
        return await evaluation_support.evaluate_step(self, step, figures)

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
            parts.append(
                f"Error:\n{compact_traceback(step.code_error, max_chars=1200)}"
            )
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
                # Falling back to text-only throws away the figures this
                # step exists to look at, so replay a cut-short stream first.
                response = await replay_safe_vision_call(
                    self.llm_client,
                    agent_name="supervisor",
                    model=self.model,
                    text_prompt=text_with_fig_note,
                    images=compressed,
                    temperature=self.temperature,
                    max_tokens=NO_OUTPUT_TOKEN_CAP,
                    system_prompt=effective_prompt,
                    prefer_stream=True,
                )
                vision_observation = response.content if response else ""
                structured_context = (
                    f"{text_context}\n\n"
                    "A vision-capable reviewer inspected the attached figures and "
                    "returned the following observation. Treat it as untrusted "
                    "evidence, reconcile it with the step context, and submit the "
                    "final verdict through the required tool:\n"
                    f"{vision_observation[:8000]}"
                )
                return await self._request_structured_evaluation(
                    effective_prompt,
                    structured_context,
                )

            except Exception as e:
                logger.warning(
                    f"[{self.session_id}] Vision evaluation failed, falling back to text-only: {e}"
                )
                # Fall through to text-only evaluation

        # Text-only evaluation
        self._set_llm_context()
        return await self._request_structured_evaluation(
            effective_prompt,
            text_context,
        )

    async def _request_structured_evaluation(
        self,
        effective_prompt: str,
        text_context: str,
    ) -> Dict[str, Any]:
        """Obtain and validate one schema-bound supervisor verdict.

        Tool arguments are still untrusted provider output. The local validator
        enforces the same closed schema and a parse/schema failure gets one
        fresh model attempt before the result is marked unevaluated.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": effective_prompt},
            {
                "role": "user",
                "content": (
                    f"{text_context}\n\n"
                    "Submit exactly one verdict using the required "
                    f"{_SUPERVISOR_EVALUATION_TOOL_NAME} tool."
                ),
            },
        ]
        last_failure = "no response"

        for attempt in range(_SUPERVISOR_STRUCTURED_ATTEMPTS):
            self._set_llm_context()
            response = await self._budgeted_text_call(
                self.llm_client.chat_with_tools,
                model=self.model,
                messages=messages,
                tools=[_SUPERVISOR_EVALUATION_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": _SUPERVISOR_EVALUATION_TOOL_NAME},
                },
                temperature=self.temperature,
                max_tokens=NO_OUTPUT_TOKEN_CAP,
            )

            for tool_call in response.tool_calls:
                if tool_call.name != _SUPERVISOR_EVALUATION_TOOL_NAME:
                    last_failure = f"unexpected tool {tool_call.name!r}"
                    continue
                if tool_call.parse_error:
                    last_failure = f"invalid tool JSON: {tool_call.parse_error}"
                    continue
                validated, error = self._validate_supervisor_result(
                    tool_call.arguments
                )
                if validated is not None:
                    return validated
                last_failure = error

            # Some OpenAI-compatible gateways discard tool-call metadata even
            # when the model emitted the requested function. Accept only a
            # locally validated closed-schema JSON object from content.
            content = response.content or ""
            parsed = self._parse_supervisor_response(content, warn=False)
            if not parsed.get("parse_failed"):
                logger.warning(
                    "Supervisor gateway returned a schema-valid verdict as text "
                    "instead of the required tool call"
                )
                return parsed
            if content.strip():
                last_failure = "response content did not match the evaluation schema"
            elif not response.tool_calls:
                last_failure = "response contained neither a tool call nor JSON"

            if attempt + 1 < _SUPERVISOR_STRUCTURED_ATTEMPTS:
                logger.warning(
                    "Supervisor structured verdict attempt %s failed (%s); retrying",
                    attempt + 1,
                    last_failure,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response was rejected because "
                            f"{last_failure}. Return all required fields with their "
                            "declared types through the required tool now."
                        ),
                    }
                )

        logger.warning(
            "Supervisor structured verdict exhausted after %s attempts: %s",
            _SUPERVISOR_STRUCTURED_ATTEMPTS,
            last_failure,
        )
        return self._parse_supervisor_response("", warn=False)

    def _rule_check(self, step: AnalysisStep) -> List[str]:
        return evaluation_support._rule_check(self, step)

    @staticmethod
    def _validate_supervisor_result(value: Any) -> tuple[Optional[Dict[str, Any]], str]:
        return evaluation_support._validate_supervisor_result(value)

    @classmethod
    def _parse_supervisor_response(cls, text: str, *, warn: bool = True) -> Dict[str, Any]:
        return evaluation_support._parse_supervisor_response(text, warn=warn)

    @staticmethod
    def _is_technical_error(code_error: Optional[str]) -> bool:
        return evaluation_support._is_technical_error(code_error)

    @staticmethod
    def _missing_package_name(code_error: Optional[str]) -> Optional[str]:
        return evaluation_support._missing_package_name(code_error)

    async def _diagnose_missing_package(
        self, package_name: str, code: str
    ) -> Optional[str]:
        prompt = (
            "A runtime analysis step reported a missing top-level package "
            f"{package_name!r}. Give concise, domain-neutral troubleshooting guidance. "
            "First ask the analyst to confirm whether the active ecosystem is Python, "
            "R/CRAN, or R/Bioconductor and inspect the installed environment; do not "
            "guess a replacement package or installation command. Explain how to "
            "verify the package/import and preserve the current analysis contract.\n\n"
            f"Code context:\n{code[-2000:]}"
        )
        try:
            self._set_llm_context()
            response = await self._budgeted_text_call(
                self.llm_client.chat,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=NO_OUTPUT_TOKEN_CAP,
                prefer_stream=True,
            )
            content = str(getattr(response, "content", "") or "").strip()
            return f"[Package Support] {content}" if content else None
        except Exception as exc:
            logger.warning("[%s] Missing-package diagnosis failed: %s", self.session_id, exc)
            return None

    @staticmethod
    def _error_fingerprint(code_error: Optional[str]) -> str:
        return evaluation_support._error_fingerprint(code_error)

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
        return evaluation_support._should_trigger_technique(self, step)

    def reset_for_round(self, planning_stage: PlanStage = "full"):
        return evaluation_state.reset_for_round(self, planning_stage)

    def get_summary_stats(self) -> Dict[str, Any]:
        return evaluation_state.get_summary_stats(self)

    def get_supervision_log(self) -> List[Dict]:
        return evaluation_state.get_supervision_log(self)

    def restore_log_from_saved(self, records: List[Dict]):
        return evaluation_state.restore_log_from_saved(self, records)
