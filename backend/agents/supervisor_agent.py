"""
Supervisor Agent - Lightweight real-time monitoring of bioinformatics analysis.

Uses a cheap vision LLM to evaluate every analysis step, detect technical issues,
identify biological checkpoints, and escalate to PI when needed. Also generates
the detailed report from its accumulated supervision log.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Awaitable
import asyncio
import json
import logging
import os
import re
import hashlib
import uuid
import io
import base64
import shutil
from pathlib import Path
from urllib.parse import quote as url_quote

logger = logging.getLogger(__name__)

from openai import APIConnectionError

from ..models.schemas import AgentType, AnalysisState, AnalysisStep
from ..utils.error_parsing import API_ERROR_TYPES, extract_error_type

# Python exception types that are purely technical (no PI guidance needed)
_TECHNICAL_ERROR_TYPES = frozenset({
    "KeyError", "TypeError", "IndexError", "ValueError",
    "ModuleNotFoundError", "AttributeError", "NameError",
    "SyntaxError", "ImportError", "FileNotFoundError",
    "ZeroDivisionError", "RuntimeError", "OSError",
    "IOError", "PermissionError", "StopIteration",
    "UnicodeDecodeError", "UnicodeEncodeError",
    "RecursionError", "OverflowError", "NotImplementedError",
    "UnboundLocalError",
    # Library-specific technical errors (no PI guidance needed)
    "InvalidIndexError", "AnnDataReadError", "SparseDataError",
})
from ..utils.llm_client import OpenAICompatibleClient

# Section IDs that represent specific/targeted analysis
_SPECIFIC_SECTION_IDS = frozenset({
    "differential_expression", "key_findings", "integrated_findings",
    "exploration_findings",
})

# ---------------------------------------------------------------------------
# Rich HTML report CSS (extracted to utils/report_css.py for separation of concerns)
# ---------------------------------------------------------------------------
from ..utils.report_css import REPORT_CSS as _REPORT_CSS  # noqa: E402


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SupervisorRecord:
    """One record per monitored step."""
    step_number: int
    step_type: str
    timestamp: datetime
    technical_issues: List[str] = field(default_factory=list)
    checkpoint_type: Optional[str] = None
    llm_severity: Optional[str] = None  # "none"|"info"|"warning"|"critical"
    pi_escalated: bool = False
    pi_response: Optional[str] = None
    feedback_injected: Optional[str] = None
    # Literature fields
    literature_topic: Optional[str] = None
    literature_result: Optional[str] = None
    # Raw data for report generation
    code: Optional[str] = None
    code_output: Optional[str] = None
    code_error: Optional[str] = None
    saved_figure_filenames: List[str] = field(default_factory=list)
    agent_reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Rule-based patterns
# ---------------------------------------------------------------------------

_CRITICAL_OUTPUT_PATTERNS = [
    re.compile(r"\b0\s+cells?\b", re.IGNORECASE),
    re.compile(r"\bEmpty DataFrame\b"),
    re.compile(r"\bMemoryError\b"),
    re.compile(r"\bn_obs\s*[:=]\s*([0-9]+)"),
    re.compile(r"obs_names are not unique", re.IGNORECASE),
    re.compile(r"InvalidIndexError"),
]

# Patterns for data integrity checks on code output
_PERCENTAGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*%", re.IGNORECASE
)
_INF_NAN_PATTERN = re.compile(
    r"\b(inf|-inf|nan)\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# System prompt for the cheap LLM evaluator
# ---------------------------------------------------------------------------

_SUPERVISOR_SYSTEM_PROMPT = """\
You are a bioinformatics analysis supervisor. Your job is to evaluate each analysis step \
for technical correctness and identify important biological decision points.

Respond ONLY with a JSON object (no markdown, no explanation):
{
  "severity": "none|info|warning|critical",
  "issues": ["list of specific technical issues found, empty if none"],
  "action": "none|warn|redirect",
  "message": "short feedback for the analyst (1-2 sentences), empty string if no issues",
  "redirect_instruction": "specific instruction if action=redirect, empty string otherwise",
  "issue_type": "technical|biological|ambiguous",
  "is_biological_checkpoint": false,
  "pi_question": "question to ask PI if is_biological_checkpoint=true, empty string otherwise"
}

IMPORTANT: All responses must be in English.

Severity guide:
- none: step looks fine
- info: minor note, no action needed
- warning: potential issue that analyst should be aware of
- critical: blank/broken figure, repeated errors, zero cells, memory errors, obviously wrong results

IMPORTANT: Code and output may be truncated for display (only the last 2000 characters are shown for long content). If content is marked as truncated, do NOT report it as syntactically incomplete or missing. Only report syntax errors if the Execution status is FAILED and the Error indicates a SyntaxError.

Biological checkpoint criteria (set is_biological_checkpoint=true when):
- A major analysis phase completed (clustering, DE, annotation, trajectory, batch correction)
- Results require biological interpretation or domain expertise
- Analysis reached a decision fork (multiple directions possible)
- PI guidance would improve the next analysis direction

IMPORTANT: Do NOT set is_biological_checkpoint=true for purely technical issues such as:
- Sandbox security blocks (e.g., "Dangerous pattern detected")
- Code validation failures or syntax errors
- Missing package / ModuleNotFoundError
- Memory errors or kernel crashes
These are technical problems the analyst can fix on their own without PI guidance.

Issue type guide (only relevant when severity is "warning" or "critical"):
- technical: coding/programming problem that the analyst can fix alone. Examples: \
wrong column name used for plotting (blank plot), incorrect API usage, data wrangling \
logic error, missing data transformation step, wrong plot type parameters.
- biological: data quality or biological interpretation issue needing PI expertise. \
Examples: all cells filtered out (QC thresholds too strict), suspicious clustering \
patterns, unexpected cell type proportions, batch effect concerns, annotation disagreement.
- ambiguous: unclear whether technical or biological. Default to this when unsure.

If images are provided, check for:
- Blank or empty plots (no data visible)
- Axis labels missing or nonsensical
- Plots with no meaningful variation (flat lines, uniform colors)
- Legends or annotations that are cut off or overlapping
"""


class SupervisorAgent:
    """Lightweight analysis supervisor using a cheap vision-capable LLM."""

    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        model: str,
        pi_callback: Optional[Callable[[str, str], Awaitable[str]]] = None,
        message_callback: Optional[Callable[[str, str], Awaitable[None]]] = None,
        session_id: str = "default",
        temperature: float = 0.1,
        system_prompt: str = "",
        technique_agent=None,
    ):
        self.llm_client = llm_client
        self.model = model
        self.pi_callback = pi_callback
        self.message_callback = message_callback
        self.session_id = session_id
        self.temperature = temperature
        # Use custom system_prompt if provided, otherwise fall back to built-in default
        self.system_prompt = system_prompt.strip() if system_prompt else ""
        self.technique_agent = technique_agent
        self._sandbox = None  # Set by workflow when sandbox is available

        # Current round state
        self.supervision_log: List[SupervisorRecord] = []
        self._step_count: int = 0
        self._recent_errors: List[str] = []
        self._pi_question_hashes: set = set()  # De-duplicate PI escalations
        self._recent_rule_feedback_hashes: set = set()  # De-duplicate rule-based feedback

        # All rounds archive
        self.all_rounds_log: List[List[SupervisorRecord]] = []

    def set_sandbox(self, sandbox):
        """Set the sandbox reference for Technique Agent doc lookups."""
        self._sandbox = sandbox

    def _set_llm_context(self):
        """Set LLM client context before each call (since client is shared)"""
        self.llm_client.set_context(self.session_id, "supervisor")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate_step(
        self,
        step: AnalysisStep,
        figures: List[str],
    ) -> Optional[str]:
        """
        Evaluate a single analysis step. Called after every tool-call iteration.

        Args:
            step: The latest AnalysisStep from analysis_log
            figures: List of base64-encoded PNG images generated in this step (may be empty)

        Returns:
            Feedback string to inject into analyst's messages, or None if no feedback needed.
        """
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

        # Phase 1.5: Technique Agent for repeated API errors
        if self._should_trigger_technique(step):
            try:
                logger.info(
                    f"[{self.session_id}] Triggering Technique Agent for repeated API error "
                    f"at step {step.step_number}"
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
        _current_err_type = extract_error_type(step.code_error) if step.code_error else None
        _same_type_count = sum(
            1 for e in self._recent_errors
            if extract_error_type(e) == _current_err_type
        ) if _current_err_type else 0
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
                pi_resp = await self._escalate_to_pi(
                    "Cell type annotation has been completed. Please review the annotation results "
                    "and provide guidance on whether they are biologically reasonable.",
                    self._build_checkpoint_context(step),
                )
                if pi_resp:
                    record.pi_escalated = True
                    record.pi_response = pi_resp
                    feedback_parts.append(f"PI guidance: {pi_resp}")

        # Assemble final feedback
        feedback = None
        if feedback_parts:
            feedback = "\n".join(feedback_parts)
            record.feedback_injected = feedback[:1000]
            await self._notify("supervisor_feedback", feedback[:500])

        self.supervision_log.append(record)
        return feedback

    def reset_for_round(self):
        """Archive current round log and reset for a new round."""
        if self.supervision_log:
            self.all_rounds_log.append(self.supervision_log.copy())
        self.supervision_log = []
        self._step_count = 0
        self._recent_errors = []
        self._pi_question_hashes = set()

    def restore_log_from_saved(self, records: List[Dict]):
        """Restore supervision log from saved analysis_log dicts (e.g., from report.json)."""
        restored = []
        for d in records:
            ts = d.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(ts) if ts else datetime.now()
            except (ValueError, TypeError):
                timestamp = datetime.now()
            restored.append(SupervisorRecord(
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
            ))
        # Store all restored records in supervision_log (flat, not split by rounds)
        self.supervision_log = restored

    def get_supervision_log(self) -> List[Dict]:
        """Return serializable log of all rounds + current round."""
        all_records = []
        for round_log in self.all_rounds_log:
            all_records.extend(round_log)
        all_records.extend(self.supervision_log)
        result = []
        for r in all_records:
            d = asdict(r)
            d["timestamp"] = r.timestamp.isoformat()
            result.append(d)
        return result

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

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    async def generate_detailed_report(
        self,
        analysis_state: AnalysisState,
        report_model: str,
        output_dir: str,
        workspace_dir: str = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Generate a detailed report from the supervisor's accumulated analysis records.

        Uses the supervision log (every step's code, output, figures, PI guidance,
        technical issues) as the primary source. Does NOT re-scan figures or re-run
        PI review — just summarizes what was recorded.

        Args:
            analysis_state: Full analysis state
            report_model: Strong model for report writing (e.g. PI model)
            output_dir: Directory to save report files
            workspace_dir: Workspace directory containing generated figures
            progress_callback: Optional progress reporting callback

        Returns:
            Report dict compatible with existing frontend expectations.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        workspace_path = Path(workspace_dir) if workspace_dir else None

        language = "en"

        async def _progress(pct: float, step: str):
            if progress_callback:
                await progress_callback({"progress": pct, "step": step, "status": "generating"})

        await _progress(5, "Preparing analysis records...")

        # Collect all records
        all_records = self.get_supervision_log()
        stats = self.get_summary_stats()

        # Build structured analysis narrative for the LLM
        narrative = self._build_analysis_narrative(all_records, analysis_state, workspace_path)

        # Collect figure paths from workspace and publication figures
        figure_map = self._collect_figure_paths(all_records, workspace_path, state=analysis_state)

        # Collect table paths from workspace
        table_list = self._collect_table_paths(workspace_path)

        await _progress(15, "Generating report content...")

        # Generate report sections using strong model
        report_prompt = self._build_report_prompt(
            narrative, stats, analysis_state, figure_map, language,
            table_list=table_list,
        )

        try:
            # Set LLM context for correct logging label (shared client)
            self.llm_client.set_context(self.session_id, "report_generator")
            for attempt in range(2):  # Single retry on connection error
                try:
                    response = await self.llm_client.chat(
                        model=report_model,
                        messages=[
                            {"role": "system", "content": self._get_report_system_prompt(language)},
                            {"role": "user", "content": report_prompt},
                        ],
                        temperature=0.3,
                        max_tokens=32768,
                        timeout=900.0,  # 15 min — large narratives need long generation time
                    )
                    report_content = response.content or ""
                    break
                except APIConnectionError:
                    if attempt == 0:
                        logger.warning(
                            f"[{self.session_id}] Report generation connection error, "
                            f"retrying in 5s"
                        )
                        await asyncio.sleep(5)
                        continue
                    logger.exception(
                        f"[{self.session_id}] Report generation failed after retry"
                    )
                    report_content = self._fallback_report(narrative, stats, language)
                except Exception:
                    logger.exception(
                        f"[{self.session_id}] Report generation LLM call failed"
                    )
                    report_content = self._fallback_report(narrative, stats, language)
                    break
        finally:
            # Restore supervisor context in case of subsequent LLM calls
            self._set_llm_context()

        await _progress(60, "Parsing report structure...")

        # Parse the LLM output into sections
        sections = self._parse_report_sections(report_content, language)

        # Extract abstract and conclusions
        abstract = ""
        conclusions = ""
        filtered_sections = []
        for sec in sections:
            sid = sec.get("id", "")
            if sid == "abstract":
                abstract = sec.get("content", "")
            elif sid == "conclusions":
                conclusions = sec.get("content", "")
            elif sid in ("supervisor_log", "figure_appendix"):
                continue  # Skip monitoring and figure appendix sections
            else:
                filtered_sections.append(sec)

        await _progress(70, "Embedding figures...")

        # Build figures list for report
        report_figures = []
        for fig_name, fig_path in figure_map.items():
            fig_entry = {
                "filename": fig_name,
                "path": str(fig_path),
                "category": self._categorize_figure(fig_name),
            }
            # Add PDF path if exists
            pdf_path = fig_path.with_suffix(".pdf")
            if pdf_path.exists():
                fig_entry["pdf_path"] = str(pdf_path)
            report_figures.append(fig_entry)

        await _progress(80, "Building report...")

        report_id = str(uuid.uuid4())
        appendix_data = self._build_appendix(analysis_state, stats)

        report = {
            "id": report_id,
            "session_id": analysis_state.id,
            "title": analysis_state.name,
            "abstract": abstract,
            "sections": filtered_sections,
            "conclusions": conclusions,
            "figures": report_figures,
            "tables": table_list,
            "graphic_abstract_path": None,  # GraphicAbstract only has image_base64, no file path
            "generated_at": datetime.now().isoformat(),
            "status": "completed",
            "progress": 100,
            "current_step": "Complete",
            "appendix": appendix_data,
            "analysis_log": all_records,
            "score_definitions": [sd.model_dump(mode="json") for sd in analysis_state.score_definitions] if analysis_state.score_definitions else [],
        }

        # Sanitize NaN/Infinity
        report = self._sanitize(report)

        # Save JSON
        report_path = output_path / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        await _progress(90, "Generating HTML report...")

        # Temporarily attach analysis_state for HTML generation (not serialized to JSON)
        report["_analysis_state"] = analysis_state

        # Generate HTML
        html_path = self._generate_html_report(
            report, filtered_sections, abstract, conclusions,
            report_figures, figure_map, stats, language, output_path, workspace_path,
        )
        report.pop("_analysis_state", None)
        report["html_path"] = str(html_path)

        await _progress(100, "Complete")
        return report

    # ------------------------------------------------------------------
    # Internal: Error classification
    # ------------------------------------------------------------------

    @staticmethod
    def _is_technical_error(code_error: Optional[str]) -> bool:
        """Return True if the error is a pure Python/technical exception (no PI help needed)."""
        if not code_error:
            return False
        err_type = extract_error_type(code_error)
        return err_type in _TECHNICAL_ERROR_TYPES

    # ------------------------------------------------------------------
    # Internal: Rule-based checks
    # ------------------------------------------------------------------

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

        return issues

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

    # ------------------------------------------------------------------
    # Internal: LLM evaluation
    # ------------------------------------------------------------------

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
                text_with_fig_note = (
                    f"{text_context}\n\n"
                    f"[{n_figs} figure(s) attached for visual inspection. "
                    f"Check for blank/empty plots, missing labels, or obviously wrong visualizations.]"
                )

                self._set_llm_context()
                response = await self.llm_client.chat_with_vision(
                    model=self.model,
                    text_prompt=f"{effective_prompt}\n\n{text_with_fig_note}",
                    images=compressed,
                    temperature=self.temperature,
                    max_tokens=512,
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
            max_tokens=512,
        )
        return self._parse_supervisor_response(response.content if response else "")

    # ------------------------------------------------------------------
    # Internal: PI escalation
    # ------------------------------------------------------------------

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

    def _build_step_context(self, step: AnalysisStep) -> str:
        """Build context string for a step (used in PI escalation for critical issues)."""
        parts = [f"Step {step.step_number} ({step.step_type})"]
        if step.code_output:
            parts.append(f"Output: {step.code_output}")
        if step.code_error:
            parts.append(f"Error: {step.code_error[:300]}")
        return "\n".join(parts)

    def _build_checkpoint_context(self, step: AnalysisStep) -> str:
        """Build rich context for biological checkpoint PI escalation."""
        parts = [f"Checkpoint at step {step.step_number} ({step.step_type})"]

        # Include recent literature searches
        recent_lit = [
            r for r in self.supervision_log
            if r.step_type == "search_literature" and r.literature_topic
        ]
        if recent_lit:
            parts.append("\nLiterature searched before this checkpoint:")
            for lit in recent_lit[-3:]:
                parts.append(f"  Topic: {lit.literature_topic}")
                if lit.literature_result:
                    parts.append(f"  Key findings: {lit.literature_result[:300]}")

        if step.code_output:
            parts.append(f"\nMost recent output:\n{step.code_output[-1000:]}")
        if step.annotation_result_summary:
            parts.append(f"\nAnnotation result: {step.annotation_result_summary}")
        if step.findings:
            parts.append(f"\nFindings: {'; '.join(step.findings[:5])}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal: Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compress_base64_figure(b64_png: str, max_width: int = 512) -> str:
        """Compress a base64 PNG to a smaller JPEG base64 for vision model review."""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(base64.b64decode(b64_png)))
            if img.width > max_width:
                ratio = max_width / img.width
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS
                img = img.resize((max_width, int(img.height * ratio)), resample)
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.warning(f"Image compression failed: {e}")
            return b64_png  # Return original if compression fails

    @staticmethod
    def _parse_supervisor_response(text: str) -> Dict:
        """Parse JSON from LLM response with multi-level fallback."""
        # Try direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Supervisor response: direct JSON parse failed")
        # Try extracting from markdown code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, TypeError):
                logger.debug("Supervisor response: markdown code block JSON parse failed")
        # Try finding any JSON object in the text
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                logger.debug("Supervisor response: regex-extracted JSON parse failed")
        # Default: no intervention
        return {
            "severity": "none", "issues": [], "action": "none", "message": "",
            "is_biological_checkpoint": False, "pi_question": "",
            "issue_type": "ambiguous",
        }

    async def _notify(self, msg_type: str, content: str):
        """Send notification via callback."""
        if self.message_callback:
            try:
                await self.message_callback(msg_type, content)
            except Exception as e:
                logger.warning(f"[{self.session_id}] Supervisor notify failed: {e}")

    # ------------------------------------------------------------------
    # Report generation helpers
    # ------------------------------------------------------------------

    def _build_analysis_narrative(
        self,
        records: List[Dict],
        state: AnalysisState,
        workspace_path: Optional[Path],
    ) -> str:
        """Build a structured text narrative from all supervision records for the report LLM."""
        lines = []
        lines.append(f"Project: {state.name}")
        lines.append(f"Instruction: {state.initial_instruction[:500]}")
        lines.append(f"Total rounds: {len(state.round_results)}")
        lines.append(f"Total steps monitored: {len(records)}")
        lines.append("")

        # Split records into per-round blocks using step number resets
        round_blocks: List[List[Dict]] = []
        current_block: List[Dict] = []
        prev_step = -1
        for rec in records:
            step_num = rec.get("step_number", 0)
            if step_num <= prev_step and current_block:
                # Step number reset → new round
                round_blocks.append(current_block)
                current_block = []
            current_block.append(rec)
            prev_step = step_num
        if current_block:
            round_blocks.append(current_block)

        # Build narrative with round/phase labels
        for block_idx, block in enumerate(round_blocks):
            round_num = block_idx + 1
            # Determine phase from round_results
            phase = "exploration"
            direction = ""
            if block_idx < len(state.round_results):
                rr = state.round_results[block_idx]
                phase = rr.phase or "exploration"
            if phase == "innovation":
                # Get direction from round result (stored at synthesis time)
                if block_idx < len(state.round_results):
                    direction = state.round_results[block_idx].direction or ""
                if not direction and getattr(state, "user_chosen_direction", None):
                    direction = state.user_chosen_direction

            header = f"========== ROUND {round_num} ({phase.upper()}) =========="
            if direction:
                header += f"\nResearch Direction: {direction}"
            lines.append(header)
            lines.append("")

            for rec in block:
                step_line = f"--- Step {rec['step_number']} [{rec['step_type']}] ---"
                lines.append(step_line)
                if rec.get("agent_reasoning"):
                    lines.append(f"Reasoning: {rec['agent_reasoning'][:300]}")
                if rec.get("code"):
                    lines.append(f"Code:\n```python\n{rec['code'][-600:]}\n```")
                if rec.get("code_output"):
                    lines.append(f"Output: {rec['code_output'][-400:]}")
                if rec.get("code_error"):
                    lines.append(f"Error: {rec['code_error'][:200]}")
                if rec.get("saved_figure_filenames"):
                    lines.append(f"Figures generated: {', '.join(rec['saved_figure_filenames'])}")
                if rec.get("literature_topic"):
                    lines.append(f"Literature search: {rec['literature_topic']}")
                    if rec.get("literature_result"):
                        lines.append(f"Literature findings: {rec['literature_result'][:200]}")
                if rec.get("technical_issues"):
                    lines.append(f"Technical issues: {'; '.join(rec['technical_issues'][:3])}")
                if rec.get("pi_escalated") and rec.get("pi_response"):
                    lines.append(f"PI guidance: {rec['pi_response'][:300]}")
                lines.append("")

        # Exploration report and user-chosen directions
        if state.exploration_report and hasattr(state.exploration_report, "research_direction_candidates"):
            candidates = state.exploration_report.research_direction_candidates or []
            if candidates:
                lines.append("=== Exploration Report: Proposed Research Directions ===")
                for idx, c in enumerate(candidates):
                    lines.append(f"{idx+1}. {c.get('direction', '')}")
                    lines.append(f"   Rationale: {c.get('rationale', '')[:200]}")
                lines.append("")

        if state.user_chosen_directions:
            lines.append("=== User-Selected Research Directions ===")
            for d in state.user_chosen_directions:
                lines.append(f"- {d.get('direction', '')} (Notes: {d.get('notes', '')})")
            lines.append("")

        # Publication figures
        if state.publication_figures:
            lines.append("=== Publication Figures ===")
            for pf in state.publication_figures:
                lines.append(f"- {pf.name}: {pf.description or ''}")
                if pf.png_path:
                    lines.append(f"  Path: {pf.png_path}")

        # Truncate to fit context
        full_text = "\n".join(lines)
        if len(full_text) > 80000:
            full_text = full_text[:80000] + "\n\n[... truncated ...]"
        return full_text

    def _collect_figure_paths(
        self, records: List[Dict], workspace_path: Optional[Path],
        state: Optional[AnalysisState] = None,
    ) -> Dict[str, Path]:
        """Collect all figure file paths from supervision records, publication figures,
        and workspace directory scan.

        The workspace scan fallback is essential for report-only generation after
        session restore, where a fresh SupervisorAgent has empty supervision logs.
        """
        figure_map: Dict[str, Path] = {}
        if not workspace_path:
            return figure_map

        for rec in records:
            for fname in rec.get("saved_figure_filenames", []):
                if fname in figure_map:
                    continue
                # Search for the figure in workspace
                for candidate in [
                    workspace_path / fname,
                    workspace_path / "figures" / fname,
                ]:
                    if candidate.exists():
                        figure_map[fname] = candidate
                        break

        # Also include publication figures (PI-reviewed, may not be in supervision records)
        if state and state.publication_figures:
            for pf in state.publication_figures:
                if not pf.png_path:
                    continue
                fname = os.path.basename(pf.png_path)
                if fname in figure_map:
                    continue
                fig_path = Path(pf.png_path)
                if not fig_path.exists():
                    fig_path = workspace_path / "figures" / fname
                if fig_path.exists():
                    figure_map[fname] = fig_path

        # Fallback: scan workspace directory for any figures not already collected.
        scan_dirs = [workspace_path]
        figures_subdir = workspace_path / "figures"
        if figures_subdir.exists():
            scan_dirs.append(figures_subdir)

        seen_stems = {Path(f).stem for f in figure_map}
        for scan_dir in scan_dirs:
            for img_file in sorted(scan_dir.glob("*.jpg")) + sorted(scan_dir.glob("*.png")):
                if img_file.stem in seen_stems:
                    continue
                seen_stems.add(img_file.stem)
                figure_map[img_file.name] = img_file

        # Deduplicate versioned figures: when both base and _v2/_v3/_final/_revised
        # exist, keep only the highest version and remove the base figure.
        _version_re = re.compile(r'^(.+?)_(v(\d+)|final|revised)$')
        base_to_versioned: Dict[str, List[str]] = {}
        for stem in list(seen_stems):
            m = _version_re.match(stem)
            if m:
                base_to_versioned.setdefault(m.group(1), []).append(stem)
        for base, versions in base_to_versioned.items():
            if base in seen_stems:
                # Remove the base version from figure_map, keep the versioned one(s)
                for fname in list(figure_map.keys()):
                    if Path(fname).stem == base:
                        del figure_map[fname]
                        logger.debug(f"Removed base figure {fname} in favor of versioned: {versions}")
                        break

        logger.info(f"Collected {len(figure_map)} figures for report "
                     f"({len(records)} supervision records, workspace={workspace_path})")

        return figure_map

    def _collect_table_paths(
        self, workspace_path: Optional[Path],
    ) -> List[Dict[str, Any]]:
        """Scan workspace for CSV tables generated during analysis."""
        tables = []
        if not workspace_path:
            return tables

        tables_subdir = workspace_path / "tables"
        csv_search_dir = tables_subdir if tables_subdir.exists() else workspace_path

        for csv_file in sorted(csv_search_dir.glob("*.csv")):
            table_info: Dict[str, Any] = {
                "filename": csv_file.name,
                "path": str(csv_file),
                "name": csv_file.stem.replace("_", " ").title(),
            }
            # Read preview
            try:
                import pandas as pd
                df_preview = pd.read_csv(csv_file, nrows=5)
                table_info["columns"] = list(df_preview.columns)
                table_info["preview_rows"] = len(df_preview)
                table_info["total_rows"] = sum(1 for _ in open(csv_file)) - 1
            except Exception:
                logger.debug("Failed to read CSV preview for table collection: %s", csv_file)
            tables.append(table_info)

        if tables:
            logger.info(f"Collected {len(tables)} tables from workspace")
        return tables

    def _build_report_prompt(
        self,
        narrative: str,
        stats: Dict,
        state: AnalysisState,
        figure_map: Dict[str, Path],
        language: str,
        table_list: Optional[List[Dict]] = None,
    ) -> str:
        """Build the prompt for the report-writing LLM."""
        figure_list = "\n".join(
            f"- {name} ({self._categorize_figure(name)})" for name in figure_map.keys()
        ) or ("None")

        # Collect per-round findings with phase labels
        findings_parts = []
        has_innovation = False
        innovation_directions = []
        for rr in state.round_results:
            phase_label = (rr.phase or "exploration").upper()
            if rr.phase == "innovation":
                has_innovation = True
            header = f"### Round {rr.round_number} ({phase_label})"

            # Get direction for innovation rounds (stored at synthesis time)
            direction = ""
            if rr.phase == "innovation":
                direction = getattr(rr, "direction", "") or ""
                if not direction and getattr(state, "user_chosen_direction", None):
                    direction = state.user_chosen_direction
                if direction:
                    header += f" — Direction: {direction}"
                    innovation_directions.append((rr.round_number, direction))

            findings_parts.append(header)
            if rr.pi_summary:
                findings_parts.append(f"**PI Summary**: {rr.pi_summary[:400]}")
            for agent_name, findings_list in [
                ("Bioinformatics", rr.bioinformatics_findings),
                ("Pathology", rr.pathology_findings),
                ("Biochemistry", rr.biochemistry_findings),
                ("Literature", rr.literature_findings),
            ]:
                if findings_list:
                    findings_parts.append(f"**{agent_name}**:")
                    for f in findings_list[:8]:
                        findings_parts.append(f"- {f[:300]}")
            findings_parts.append("")

        findings_text = "\n".join(findings_parts) or ("None")

        # Build dynamic section template
        base_sections = [
            "[abstract] - Research summary (background, methods, key findings, conclusions)",
            "[analysis_overview] - Analysis methodology overview (study design, data processing pipeline, computational methods, key parameters)",
            "[data_qc] - Data overview and quality control",
            "[clustering_annotation] - Clustering and cell type annotation (include PI checkpoint guidance if available)",
            "[differential_expression] - Differential expression analysis (include PI checkpoint guidance if available)",
        ]

        if has_innovation:
            base_sections.append(
                "[exploration_findings] - Phase 1 exploration findings summary and proposed research directions"
            )
            for i, (round_num, direction) in enumerate(innovation_directions):
                short_dir = direction[:80] if direction else f"Direction {i+1}"
                base_sections.append(
                    f"[innovation_{i+1}] - Innovation Phase (Round {round_num}): {short_dir} "
                    f"(hypothesis, targeted analyses such as GSEA/ORA/gene modules, key findings, "
                    f"biological significance). This section MUST be comprehensive and include ALL "
                    f"findings from this research direction."
                )
            base_sections.append(
                "[integrated_findings] - Integrated findings across all phases (cross-phase connections, "
                "how innovation findings validate or extend exploration discoveries)"
            )
        else:
            base_sections.append("[key_findings] - Key biological findings")

        base_sections.append(
            "[conclusions] - Conclusions and future directions"
        )

        sections_text = "\n".join(base_sections)

        # Build table list text
        table_list = table_list or []
        table_list_text = "\n".join(
            f"- {tbl['filename']}: {tbl.get('name', '')} ({tbl.get('total_rows', '?')} rows, columns: {', '.join(tbl.get('columns', [])[:8])})"
            for tbl in table_list
        ) if table_list else "None"

        prompt = f"""Based on the following complete analysis records, generate a detailed scientific research report.

## Analysis Narrative
{narrative}

## Key Findings (Per Round)
{findings_text}

## Available Figures
{figure_list}

## Available Tables (CSV)
{table_list_text}

## Instructions
Write the report in English.
Generate the report with the following sections (use the exact section IDs in brackets):

{sections_text}

Format each section as:
## [section_id] Section Title
Content...

When referencing a figure, use this exact format (each marker on its own line).
IMPORTANT: Use the EXACT filename from the Available Figures list above (including the correct extension — most are .jpg, NOT .png):
[FIGURE: exact_filename_from_list_above.jpg]
[FIGURE_CAPTION: A concise scientific caption (1-2 sentences) describing what this figure shows and key observations]

When referencing a data table, use this exact format (each marker on its own line):
[TABLE: exact_filename.csv]
[TABLE_CAPTION: A concise scientific caption (1-2 sentences) describing key data in this table and its significance]

Each figure and table should appear exactly once in the most relevant section.
Do NOT create separate figure/table appendix or listing sections.

IMPORTANT:
- Write as a formal scientific research report, NOT a monitoring or system log.
- Make the report thorough and comprehensive with detailed scientific content (300-500 words minimum per section).
- Innovation phase sections are CRITICAL — they contain the novel research findings and MUST be written in detail.
- Reference specific figures, genes, pathways, cell types, statistical results, and quantitative findings.
- Include ALL available figures — assign each figure to the most relevant section using [FIGURE: filename] markers.
- Include ALL available tables — assign each table to the most relevant section using [TABLE: filename.csv] markers.
- Every [FIGURE:] and [TABLE:] marker MUST be followed by its corresponding [FIGURE_CAPTION:] or [TABLE_CAPTION:] marker on the next line.
"""
        return prompt

    def _get_report_system_prompt(self, language: str) -> str:
        return (
            "You are a senior bioinformatics researcher writing a detailed scRNA-seq analysis report. "
            "The report should be professional, rigorous, and comprehensive, including key analysis "
            "decisions and PI guidance. Use Markdown format, referencing specific figure filenames "
            "and analysis step numbers."
        )

    def _parse_report_sections(self, content: str, language: str) -> List[Dict]:
        """Parse LLM-generated report content into structured sections."""
        sections = []
        # Split by ## [section_id] pattern
        pattern = re.compile(r'^##\s*\[(\w+)\]\s*(.+?)$', re.MULTILINE)
        matches = list(pattern.finditer(content))

        for i, match in enumerate(matches):
            section_id = match.group(1)
            title = match.group(2).strip()
            # Content is between this match and the next
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()

            analysis_type = (
                "specific"
                if section_id in _SPECIFIC_SECTION_IDS or section_id.startswith("innovation_")
                else "general"
            )
            sections.append({
                "id": section_id,
                "title": title,
                "content": section_content,
                "analysis_type": analysis_type,
                "description": "",
                "figures": [],
                "tables": [],
            })

        if not sections:
            # Fallback: treat entire content as one section
            sections.append({
                "id": "full_report",
                "title": "Analysis Report",
                "content": content,
                "analysis_type": "general",
                "description": "",
                "figures": [],
                "tables": [],
            })

        return sections

    def _fallback_report(self, narrative: str, stats: Dict, language: str) -> str:
        """Generate a minimal fallback report when LLM call fails.

        Extracts only step headers and reasoning from the raw narrative,
        stripping code blocks and outputs to keep the fallback readable.

        Note: Currently generates English text only. The language parameter is
        accepted for interface consistency but not used in this fallback path.
        """
        # Extract step summaries (reasoning only, no code/output)
        summary_lines = []
        for line in narrative.split('\n'):
            stripped = line.strip()
            if stripped.startswith('--- Step') or stripped.startswith('Reasoning:'):
                summary_lines.append(stripped)
            elif stripped.startswith('========== ROUND'):
                summary_lines.append('')
                summary_lines.append(stripped)
            if len(summary_lines) >= 60:
                break

        if summary_lines:
            overview = '\n'.join(summary_lines)
        else:
            overview = (
                "Report content generation failed due to API connection error. "
                "Please refer to the analysis log in the appendix for detailed records."
            )

        return f"""## [abstract] Abstract
This report was generated by the automated monitoring system. Analysis monitored {stats['total_steps_monitored']} steps, found {stats['steps_with_technical_issues']} technical issues, with {stats['pi_escalations']} PI escalations.

Note: The detailed narrative report could not be generated due to an API connection error. Below is a summary of the analysis steps performed. For full details including code and outputs, please refer to the appendix.

## [analysis_overview] Analysis Overview
{overview}

## [conclusions] Conclusions
Report generation model call failed due to API connection error. Please refer to the analysis log in the appendix for complete information, including all figures and tables generated during the analysis.
"""

    @staticmethod
    def _categorize_figure(filename: str) -> str:
        """Categorize a figure by its filename."""
        name = filename.lower()
        if "umap" in name:
            return "umap"
        if "qc" in name or "quality" in name or "violin_qc" in name:
            return "qc"
        if "cluster" in name:
            return "clustering"
        if "marker" in name or "dotplot" in name:
            return "markers"
        if "heatmap" in name:
            return "heatmap"
        if "de_" in name or "volcano" in name or "differential" in name:
            return "de"
        if "violin" in name:
            return "violin"
        if "trajectory" in name or "paga" in name:
            return "trajectory"
        return "other"

    def _build_appendix(self, state: AnalysisState, stats: Dict) -> Dict:
        """Build appendix metadata for the report."""
        # Build user interaction records
        cluster_annotations_data = []
        if state.annotation_state and state.annotation_state.cluster_annotations:
            for ca in state.annotation_state.cluster_annotations:
                cluster_annotations_data.append({
                    "cluster_id": ca.cluster_id,
                    "cell_type": ca.cell_type,
                    "user_override": ca.user_override or "",
                    "confirmed": ca.user_confirmed,
                    "confidence": ca.confidence,
                })

        direction_selection_data = []
        if state.user_chosen_directions:
            for d in state.user_chosen_directions:
                direction_selection_data.append({
                    "direction": d.get("direction", ""),
                    "notes": d.get("notes", ""),
                })

        return {
            "project_name": state.name,
            "project_description": state.initial_instruction[:500],
            "created_at": datetime.now().isoformat(),
            "agent_models": dict(state.custom_models),
            "agent_temperatures": dict(state.custom_temperatures),
            "treatment_groups": [
                {"id": g.id, "name": g.name, "description": g.description, "sample_count": g.sample_count}
                for g in state.treatment_groups
            ],
            "data_files": [
                {"name": f.filename, "type": f.data_type, "section": f.section}
                for f in state.data_files
            ],
            "round_results": [
                {"round": rr.round_number, "findings_count": (
                    len(rr.bioinformatics_findings) + len(rr.pathology_findings) +
                    len(rr.biochemistry_findings) + len(rr.literature_findings)
                )}
                for rr in state.round_results
            ],
            "user_interactions": {
                "cluster_annotations": cluster_annotations_data,
                "direction_selection": direction_selection_data,
            },
            "related_reports": {
                "exploration_report": "exploration_report.html",
                "annotation_report": "annotation_context_report.html",
            },
            "notebook_filename": f"{state.id}.ipynb",
        }

    @staticmethod
    def _sanitize(obj):
        """Sanitize NaN/Infinity values for JSON serialization."""
        import math
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: SupervisorAgent._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [SupervisorAgent._sanitize(v) for v in obj]
        return obj

    def _generate_html_report(
        self,
        report: Dict,
        sections: List[Dict],
        abstract: str,
        conclusions: str,
        report_figures: List[Dict],
        figure_map: Dict[str, Path],
        stats: Dict,
        language: str,
        output_path: Path,
        workspace_path: Optional[Path],
    ) -> Path:
        """Generate a standalone HTML report with embedded images and download links."""
        import html as html_module

        def _maybe_escape_html(text: str) -> str:
            """Return *text* as-is if it already contains HTML tags, otherwise escape it.

            Caption text from [FIGURE_CAPTION:]/[TABLE_CAPTION:] markers has been
            through _markdown_to_html() and may contain safe HTML.  Captions from
            pub_fig_descriptions are plain text and need escaping.
            """
            if '<' in text and '>' in text:
                return text
            return html_module.escape(text)

        # Build figure base64 cache for embedding
        figure_b64_cache: Dict[str, str] = {}
        for fig_name, fig_path in figure_map.items():
            try:
                with open(fig_path, "rb") as f:
                    raw = f.read()
                figure_b64_cache[fig_name] = base64.b64encode(raw).decode()
            except Exception as e:
                logger.warning(f"Could not read figure {fig_path}: {e}")

        # Build TOC
        toc_items = []
        for sec in sections:
            a_type = sec.get("analysis_type", "general")
            type_class = "general" if a_type == "general" else "specific"
            type_label = "General" if a_type == "general" else "Specific"
            toc_items.append(
                f'<li><a href="#{html_module.escape(sec["id"])}">'
                f'{html_module.escape(sec["title"])}</a>'
                f'<span class="section-type {type_class}">{type_label}</span></li>'
            )

        # Build publication figure description lookup
        pub_fig_descriptions: Dict[str, Dict[str, str]] = {}
        analysis_state = report.get("_analysis_state")
        if analysis_state and analysis_state.publication_figures:
            for pf in analysis_state.publication_figures:
                fname = os.path.basename(pf.png_path) if pf.png_path else ""
                if fname:
                    pub_fig_descriptions[fname] = {
                        "name": pf.name,
                        "description": pf.description or "",
                        "interpretation": pf.interpretation or "",
                    }

        def _alt_ext_name(name: str) -> str:
            """Return filename with alternate image extension (.png↔.jpg/.jpeg)."""
            stem, ext = os.path.splitext(name)
            lower_ext = ext.lower()
            if lower_ext in (".jpg", ".jpeg"):
                return stem + ".png"
            return stem + ".jpg"

        def _build_figure_html(fig_name: str, caption: str = "") -> str:
            """Build inline figure HTML block for a given figure filename."""
            fig_b64 = figure_b64_cache.get(fig_name, "")
            # Fallback: try alternate extension (.png ↔ .jpg/.jpeg) if exact name not found
            if not fig_b64:
                alt_name = _alt_ext_name(fig_name)
                fig_b64 = figure_b64_cache.get(alt_name, "")
                if fig_b64:
                    logger.info(f"Figure extension fallback: '{fig_name}' → '{alt_name}'")
                    fig_name = alt_name
            if not fig_b64:
                return ""
            # Determine MIME type from extension
            mime = "image/jpeg" if fig_name.lower().endswith((".jpg", ".jpeg")) else "image/png"

            # Figure description: prefer LLM caption, fallback to publication figures
            fig_desc = pub_fig_descriptions.get(fig_name, {})
            display_name = fig_desc.get("name", fig_name.replace("_", " ").rsplit(".", 1)[0].title())
            description = fig_desc.get("description", "")
            interpretation = caption or fig_desc.get("interpretation", "")

            # Description paragraph
            desc_html = ""
            if description:
                desc_html = f'<p><em>{html_module.escape(description)}</em></p>'

            # Interpretation block
            interp_html = ""
            if interpretation:
                interp_content = _maybe_escape_html(interpretation)
                interp_html = (
                    f'<div class="figure-interpretation">'
                    f'<strong>Interpretation</strong>'
                    f'<p>{interp_content}</p>'
                    f'</div>'
                )

            # Download buttons — label matches actual file extension
            fig_path = figure_map.get(fig_name) or figure_map.get(_alt_ext_name(fig_name))
            img_ext = os.path.splitext(fig_name)[1].lstrip('.').upper() or "PNG"
            img_btn = (
                f'<a href="figures/{html_module.escape(fig_name)}" '
                f'download class="download-btn png">Download {img_ext}</a>'
            )
            pdf_btn = ""
            if fig_path:
                pdf_p = fig_path.with_suffix(".pdf")
                if pdf_p.exists():
                    pdf_btn = (
                        f'<a href="figures/{html_module.escape(pdf_p.name)}" '
                        f'download class="download-btn pdf">Download PDF</a>'
                    )

            return f'''
                <div class="figure-container">
                    <div class="figure-image">
                        <img src="data:{mime};base64,{fig_b64}"
                             alt="{html_module.escape(fig_name)}"/>
                    </div>
                    <div class="figure-caption">
                        <h4>{html_module.escape(display_name)}</h4>
                        {desc_html}
                        {interp_html}
                        <div class="figure-actions">
                            {img_btn}
                            {pdf_btn}
                        </div>
                    </div>
                </div>
            '''

        # Track which figures have been placed
        placed_figures: set = set()

        # Build table lookup for inline embedding
        table_lookup: Dict[str, Dict] = {}
        for tbl in report.get("tables", []):
            table_lookup[tbl["filename"]] = tbl
        placed_tables: set = set()

        def _build_table_html(tbl_filename: str, caption: str = "") -> str:
            """Build inline table HTML block for a given CSV filename."""
            tbl_info = table_lookup.get(tbl_filename)
            if not tbl_info:
                return ""
            tbl_path = tbl_info.get("path", "")
            if not tbl_path or not Path(tbl_path).exists():
                return ""

            display_name = tbl_info.get("name", tbl_filename.replace("_", " ").rsplit(".", 1)[0].title())
            csv_url = url_quote(tbl_filename)

            # Read preview data (up to 10 rows) from CSV
            columns = tbl_info.get("columns", [])
            total_rows = tbl_info.get("total_rows", 0)
            preview_data = []
            try:
                import pandas as pd
                df_preview = pd.read_csv(tbl_path, nrows=10)
                columns = list(df_preview.columns)
                preview_data = df_preview.to_dict('records')
                if not total_rows:
                    total_rows = sum(1 for _ in open(tbl_path)) - 1
            except Exception:
                logger.debug("Failed to read CSV preview for HTML table: %s", tbl_path)

            if not columns:
                return ""

            table_header = "".join(
                f"<th>{html_module.escape(str(col))}</th>" for col in columns
            )
            table_rows = ""
            for row in preview_data[:10]:
                row_cells = "".join(
                    f"<td>{html_module.escape(str(row.get(col, '-')))}</td>" for col in columns
                )
                table_rows += f"<tr>{row_cells}</tr>"

            row_info = ""
            if total_rows > 10:
                row_info = f'<div class="table-info">Showing 10 of {total_rows} rows</div>'

            caption_html = ""
            if caption:
                caption_html = (
                    f'<div class="table-interpretation">'
                    f'<strong>Interpretation</strong>'
                    f'<p>{_maybe_escape_html(caption)}</p>'
                    f'</div>'
                )

            return f'''
                <div class="table-container">
                    <div class="table-header">
                        <h4>{html_module.escape(display_name)}</h4>
                        <a href="tables/{csv_url}" download class="download-btn csv">
                            Download CSV
                        </a>
                    </div>
                    <div class="table-wrapper">
                        <table>
                            <thead><tr>{table_header}</tr></thead>
                            <tbody>{table_rows}</tbody>
                        </table>
                    </div>
                    {row_info}
                    {caption_html}
                </div>
            '''

        # Build sections HTML with inline figure embedding
        sections_html = []
        for sec in sections:
            raw_content = sec.get("content", "")
            content_html = self._markdown_to_html(raw_content)

            # Replace [FIGURE: filename] + optional [FIGURE_CAPTION: ...] markers
            def _replace_figure_marker(m):
                fname = m.group(1).strip()
                caption = (m.group(2) or "").strip() if m.lastindex >= 2 else ""
                fig_html = _build_figure_html(fname, caption=caption)
                if fig_html:
                    placed_figures.add(fname)
                    return fig_html
                logger.info(f"Figure referenced in report but not found, skipping: {fname}")
                return ""

            # Match [FIGURE: name] optionally followed by [FIGURE_CAPTION: text]
            content_html = re.sub(
                r'\[FIGURE:\s*([^\]]+)\](?:\s*(?:<[^>]*>\s*)*\[FIGURE_CAPTION:\s*([^\]]+)\])?',
                _replace_figure_marker,
                content_html,
            )

            # Replace [TABLE: filename] + optional [TABLE_CAPTION: ...] markers
            def _replace_table_marker(m):
                fname = m.group(1).strip()
                caption = (m.group(2) or "").strip() if m.lastindex >= 2 else ""
                tbl_html = _build_table_html(fname, caption=caption)
                if tbl_html:
                    placed_tables.add(fname)
                    return tbl_html
                logger.info(f"Table referenced in report but not found, skipping: {fname}")
                return ""

            # Match [TABLE: name] optionally followed by [TABLE_CAPTION: text]
            content_html = re.sub(
                r'\[TABLE:\s*([^\]]+)\](?:\s*(?:<[^>]*>\s*)*\[TABLE_CAPTION:\s*([^\]]+)\])?',
                _replace_table_marker,
                content_html,
            )

            # Fallback: embed figures referenced by filename in text but without markers
            remaining_fig_parts = []
            for fig_name in figure_b64_cache:
                if fig_name in placed_figures:
                    continue
                if fig_name.lower() in raw_content.lower():
                    fig_html = _build_figure_html(fig_name)
                    if fig_html:
                        placed_figures.add(fig_name)
                        remaining_fig_parts.append(fig_html)

            # Fallback: embed tables referenced by filename in text but without markers
            remaining_tbl_parts = []
            for tbl_filename in table_lookup:
                if tbl_filename in placed_tables:
                    continue
                if tbl_filename.lower() in raw_content.lower():
                    tbl_html = _build_table_html(tbl_filename)
                    if tbl_html:
                        placed_tables.add(tbl_filename)
                        remaining_tbl_parts.append(tbl_html)

            a_type = sec.get("analysis_type", "general")
            type_class = "general" if a_type == "general" else "specific"
            type_label = "General" if a_type == "general" else "Specific"

            sec_html = f'''
            <div class="section" id="{html_module.escape(sec["id"])}">
                <div class="section-header">
                    <h2>
                        {html_module.escape(sec["title"])}
                        <span class="analysis-type-badge {type_class}">{type_label}</span>
                    </h2>
                </div>
                <div class="section-content">
                    {content_html}
                    {"".join(remaining_fig_parts)}
                    {"".join(remaining_tbl_parts)}
                </div>
            </div>
            '''
            sections_html.append(sec_html)

        # Append any unplaced figures as a supplementary section
        unplaced = [fn for fn in figure_b64_cache if fn not in placed_figures and not fn.startswith("graphic_abstract.")]
        if unplaced:
            supp_title = "Supplementary Figures"
            supp_figs = "".join(_build_figure_html(fn) for fn in unplaced)
            sections_html.append(f'''
            <div class="section" id="supplementary_figures">
                <div class="section-header">
                    <h2>{supp_title}</h2>
                </div>
                <div class="section-content">
                    {supp_figs}
                </div>
            </div>
            ''')

        # Append any unplaced tables as a supplementary section
        unplaced_tables = [fn for fn in table_lookup if fn not in placed_tables]
        if unplaced_tables:
            supp_tbl_title = "Supplementary Data Tables"
            supp_tbl_parts = []
            for fn in unplaced_tables:
                tbl_html = _build_table_html(fn)
                if tbl_html:
                    supp_tbl_parts.append(tbl_html)
            if supp_tbl_parts:
                sections_html.append(f'''
                <div class="section" id="supplementary_tables">
                    <div class="section-header">
                        <h2>{supp_tbl_title}</h2>
                    </div>
                    <div class="section-content">
                        <p>The following data tables were generated during analysis and are provided for download and reference.</p>
                        {"".join(supp_tbl_parts)}
                    </div>
                </div>
                ''')

        title = report.get("title", "Analysis Report")

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_module.escape(title)}</title>
<style>
{_REPORT_CSS}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>{html_module.escape(title)}</h1>
        <div class="meta">Generated: {generated_at}</div>
    </div>
    <div class="content">
        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>{"".join(toc_items)}</ul>
        </div>

        <div class="abstract">
            <h2>Abstract</h2>
            {self._markdown_to_html(abstract)}
        </div>

        {"".join(sections_html)}

        <div class="conclusions">
            <h2>Conclusions</h2>
            {self._markdown_to_html(conclusions)}
        </div>

        {self._build_score_definitions_html(report)}

        {self._build_appendix_html_section(report)}
    </div>
</div>
</body>
</html>'''

        html_file = output_path / "report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Copy figures to output for download links
        figures_out = output_path / "figures"
        figures_out.mkdir(exist_ok=True)
        for fig_name, fig_path in figure_map.items():
            try:
                shutil.copy2(fig_path, figures_out / fig_name)
                # Copy PDF if exists
                pdf_path = fig_path.with_suffix(".pdf")
                if pdf_path.exists():
                    shutil.copy2(pdf_path, figures_out / pdf_path.name)
            except Exception as e:
                logger.warning(f"Could not copy figure {fig_name}: {e}")

        # Copy tables to output for download links
        tables_out = output_path / "tables"
        tables_out.mkdir(exist_ok=True)
        for tbl in report.get("tables", []):
            tbl_path = tbl.get("path", "")
            if tbl_path and Path(tbl_path).exists():
                try:
                    shutil.copy2(tbl_path, tables_out / os.path.basename(tbl_path))
                except Exception as e:
                    logger.warning(f"Could not copy table {tbl_path}: {e}")

        # Copy notebook to output for download
        _analysis_state = report.get("_analysis_state")
        if _analysis_state:
            from ..config import get_session_states
            notebook_src = Path(get_session_states(_analysis_state.id)) / f"{_analysis_state.id}.ipynb"
            if notebook_src.exists():
                try:
                    shutil.copy2(notebook_src, output_path / f"{_analysis_state.id}.ipynb")
                except Exception as e:
                    logger.warning(f"Could not copy notebook: {e}")

        logger.info(f"HTML report generated: {html_file}")
        return html_file

    @staticmethod
    def _build_score_definitions_html(report: Dict) -> str:
        """Build score definitions HTML table from report data."""
        import html as html_module
        score_defs = report.get("score_definitions", [])
        if not score_defs:
            return ""
        rows = ""
        for sd in score_defs:
            genes_raw = sd.get("gene_list", [])
            genes = html_module.escape(", ".join(genes_raw[:10]))
            if len(genes_raw) > 10:
                genes += f" ... ({len(genes_raw)} total)"
            params = sd.get("parameters", {})
            params_str = html_module.escape(
                ", ".join(f"{k}={v}" for k, v in params.items()) if params else "default"
            )
            rows += f"""<tr>
                <td>{html_module.escape(sd.get("score_name", ""))}</td>
                <td><code>{html_module.escape(sd.get("method", ""))}</code></td>
                <td>{html_module.escape(sd.get("description", ""))}</td>
                <td>{params_str}</td>
                <td>{genes}</td>
                <td>{html_module.escape(sd.get("gene_list_source", "") or "N/A")}</td>
            </tr>"""
        return f"""
        <div class="score-definitions" id="score-definitions">
            <div class="section-header">
                <h2>Score Definitions</h2>
            </div>
            <div class="section-content">
                <p>The following scores were computed during the analysis. Each score's methodology is documented for transparency and reproducibility.</p>
                <table>
                    <thead>
                        <tr>
                            <th>Score Name</th>
                            <th>Method</th>
                            <th>Description</th>
                            <th>Parameters</th>
                            <th>Gene List</th>
                            <th>Source</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
        </div>"""

    @staticmethod
    def _build_appendix_html_section(report: Dict) -> str:
        """Build appendix HTML section from report data."""
        import html as html_module
        appendix = report.get("appendix")
        if not appendix:
            return ""

        esc = html_module.escape

        parts = []
        parts.append('<div class="appendix" id="appendix">')
        parts.append('<h2>Appendix: Analysis Settings &amp; Records</h2>')

        # Project Information
        parts.append('<h3>Project Information</h3>')
        parts.append('<table>')
        parts.append(f'<tr><td><strong>Project Name</strong></td>'
                     f'<td>{esc(appendix.get("project_name", "-"))}</td></tr>')
        if appendix.get("created_at"):
            parts.append(f'<tr><td><strong>Created At</strong></td>'
                         f'<td>{esc(appendix.get("created_at", "-"))}</td></tr>')
        if report.get("generated_at"):
            parts.append(f'<tr><td><strong>Report Generated</strong></td>'
                         f'<td>{esc(report.get("generated_at", "-"))}</td></tr>')
        parts.append('</table>')

        # Agent Configuration
        models = appendix.get("agent_models", {})
        temps = appendix.get("agent_temperatures", {})
        if models:
            parts.append('<h3>Agent Configuration</h3>')
            parts.append('<table>')
            parts.append('<thead><tr><th>Agent</th>'
                         '<th>Model</th>'
                         '<th>Temperature</th></tr></thead><tbody>')
            for key in sorted(models.keys()):
                parts.append(f'<tr><td>{esc(key.title())}</td>'
                             f'<td><code>{esc(str(models[key]))}</code></td>'
                             f'<td>{temps.get(key, "-")}</td></tr>')
            parts.append('</tbody></table>')

        # Data Files
        data_files = appendix.get("data_files", [])
        if data_files:
            parts.append('<h3>Data Files</h3>')
            parts.append('<table>')
            parts.append('<thead><tr><th>Filename</th>'
                         '<th>Type</th>'
                         '<th>Section</th></tr></thead><tbody>')
            for f in data_files:
                parts.append(f'<tr><td><code>{esc(f.get("name", "-"))}</code></td>'
                             f'<td>{esc(f.get("type", "-"))}</td>'
                             f'<td>{esc(f.get("section", "-"))}</td></tr>')
            parts.append('</tbody></table>')

        # Treatment Groups
        groups = appendix.get("treatment_groups", [])
        if groups:
            parts.append('<h3>Treatment Groups</h3>')
            parts.append('<table>')
            parts.append('<thead><tr><th>Group</th>'
                         '<th>Description</th>'
                         '<th>Samples</th></tr></thead><tbody>')
            for g in groups:
                parts.append(f'<tr><td>{esc(g.get("name", "-"))}</td>'
                             f'<td>{esc(g.get("description", "-"))}</td>'
                             f'<td>{g.get("sample_count", 0)}</td></tr>')
            parts.append('</tbody></table>')

        # Analysis Rounds
        round_results = appendix.get("round_results", [])
        if round_results:
            parts.append('<h3>Analysis Rounds</h3>')
            parts.append('<table>')
            parts.append('<thead><tr><th>Round</th>'
                         '<th>Findings</th></tr></thead><tbody>')
            for rr in round_results:
                parts.append(f'<tr><td>{rr.get("round", "-")}</td>'
                             f'<td>{rr.get("findings_count", 0)}</td></tr>')
            parts.append('</tbody></table>')

        # User Interaction History
        user_interactions = appendix.get("user_interactions", {})
        cluster_anns = user_interactions.get("cluster_annotations", [])
        dir_selections = user_interactions.get("direction_selection", [])

        if cluster_anns or dir_selections:
            parts.append('<h3>User Interaction History</h3>')

            if cluster_anns:
                parts.append('<h4>Cell Annotation Decisions</h4>')
                parts.append('<table>')
                parts.append('<thead><tr>'
                             '<th>Cluster</th>'
                             '<th>Cell Type</th>'
                             '<th>User Override</th>'
                             '<th>Confirmed</th>'
                             '<th>Confidence</th>'
                             '</tr></thead><tbody>')
                for ca in cluster_anns:
                    confirmed_icon = "✓" if ca.get("confirmed") else "-"
                    parts.append(
                        f'<tr><td>{esc(str(ca.get("cluster_id", "-")))}</td>'
                        f'<td>{esc(ca.get("cell_type", "-"))}</td>'
                        f'<td>{esc(ca.get("user_override", "-") or "-")}</td>'
                        f'<td>{confirmed_icon}</td>'
                        f'<td>{esc(ca.get("confidence", "-"))}</td></tr>'
                    )
                parts.append('</tbody></table>')

            if dir_selections:
                parts.append('<h4>Research Direction Selections</h4>')
                parts.append('<table>')
                parts.append('<thead><tr>'
                             '<th>Direction</th>'
                             '<th>User Notes</th>'
                             '</tr></thead><tbody>')
                for d in dir_selections:
                    parts.append(
                        f'<tr><td>{esc(d.get("direction", "-"))}</td>'
                        f'<td>{esc(d.get("notes", "-") or "-")}</td></tr>'
                    )
                parts.append('</tbody></table>')

        # Related Reports & Downloads
        related = appendix.get("related_reports", {})
        notebook = appendix.get("notebook_filename", "")

        if related or notebook:
            parts.append('<h3>Related Reports &amp; Downloads</h3>')
            parts.append('<ul>')
            if related.get("exploration_report"):
                parts.append(f'<li><a href="{esc(related["exploration_report"])}" download>'
                             f'Exploration Report</a></li>')
            if related.get("annotation_report"):
                parts.append(f'<li><a href="{esc(related["annotation_report"])}" download>'
                             f'Annotation Context Report</a></li>')
            if notebook:
                parts.append(f'<li><a href="{esc(notebook)}" download>'
                             f'Analysis Notebook (.ipynb)</a></li>')
            parts.append('</ul>')

        parts.append('</div>')
        return "\n".join(parts)

    @staticmethod
    def _markdown_to_html(text: str) -> str:
        """Simple markdown to HTML conversion for report content."""
        if not text:
            return ""
        import html as html_module

        # Extract code blocks and inline code before escaping to avoid double-escaping
        code_blocks = []
        def _save_code_block(m):
            placeholder = f"\x00CODEBLOCK{len(code_blocks)}\x00"
            code_blocks.append(f'<pre><code>{html_module.escape(m.group(2))}</code></pre>')
            return placeholder
        text = re.sub(r'```(\w*)\n(.*?)```', _save_code_block, text, flags=re.DOTALL)

        inline_codes = []
        def _save_inline_code(m):
            placeholder = f"\x00INLINECODE{len(inline_codes)}\x00"
            inline_codes.append(f'<code>{html_module.escape(m.group(1))}</code>')
            return placeholder
        text = re.sub(r'`([^`]+)`', _save_inline_code, text)

        # Escape remaining (non-code) text
        text = html_module.escape(text)

        # Restore code blocks and inline code
        for i, block in enumerate(code_blocks):
            text = text.replace(f"\x00CODEBLOCK{i}\x00", block)
        for i, code in enumerate(inline_codes):
            text = text.replace(f"\x00INLINECODE{i}\x00", code)
        # Headers (longest prefix first)
        text = re.sub(r'^####\s+(.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
        text = re.sub(r'^###\s+(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^#\s+(.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        # Bullet lists
        text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
        text = re.sub(r'(<li>.*?</li>\n?)+', r'<ul>\g<0></ul>', text)
        # Paragraphs
        paragraphs = text.split('\n\n')
        result_parts = []
        for p in paragraphs:
            p = p.strip()
            if p and not p.startswith('<'):
                p = f'<p>{p}</p>'
            result_parts.append(p)
        text = '\n'.join(result_parts)
        # Line breaks
        text = text.replace('\n', '<br/>\n')
        # Clean up double <br/> in pre blocks
        text = re.sub(r'(<pre>.*?</pre>)', lambda m: m.group(0).replace('<br/>\n', '\n'), text, flags=re.DOTALL)
        return text
