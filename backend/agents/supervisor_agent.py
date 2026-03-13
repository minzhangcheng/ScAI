"""
Supervisor Agent - Lightweight real-time monitoring of bioinformatics analysis.

Uses a cheap vision LLM to evaluate every analysis step, detect technical issues,
identify biological checkpoints, and escalate to PI when needed. Also generates
the detailed report from its accumulated supervision log.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Awaitable
import json
import logging
import re
import hashlib
import uuid
import io
import base64
from pathlib import Path

logger = logging.getLogger(__name__)

from ..models.schemas import AgentType, AnalysisState, AnalysisStep
from ..utils.llm_client import OpenAICompatibleClient


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
]


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
  "is_biological_checkpoint": false,
  "pi_question": "question to ask PI if is_biological_checkpoint=true, empty string otherwise"
}

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
    ):
        self.llm_client = llm_client
        self.model = model
        self.pi_callback = pi_callback
        self.message_callback = message_callback
        self.session_id = session_id
        self.temperature = temperature
        # Use custom system_prompt if provided, otherwise fall back to built-in default
        self.system_prompt = system_prompt.strip() if system_prompt else ""

        # Current round state
        self.supervision_log: List[SupervisorRecord] = []
        self._step_count: int = 0
        self._recent_errors: List[str] = []
        self._pi_question_hashes: set = set()  # De-duplicate PI escalations

        # All rounds archive
        self.all_rounds_log: List[List[SupervisorRecord]] = []

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
            agent_reasoning=step.agent_reasoning[:500] if step.agent_reasoning else None,
            literature_topic=step.literature_topic,
            literature_result=step.literature_result[:500] if step.literature_result else None,
        )

        feedback_parts: List[str] = []

        # Phase 1: Rule-based checks (no LLM, instant)
        rule_issues = self._rule_check(step)
        if rule_issues:
            record.technical_issues.extend(rule_issues)
            feedback_parts.extend(rule_issues)

        # Phase 2: LLM evaluation (sampled to reduce cost)
        # Only invoke LLM on steps that are likely to need it
        should_llm_evaluate = (
            bool(step.code_error) or              # Has execution error
            bool(figures) or                       # Has figure output
            step.step_type == "annotate_clusters" or  # Critical biological decision
            self._step_count % 5 == 0              # Sample every 5 steps
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

            # Critical → inject feedback + escalate to PI
            if severity == "critical":
                if message:
                    feedback_parts.append(f"CRITICAL: {message}")
                redirect = llm_result.get("redirect_instruction", "")
                if redirect:
                    feedback_parts.append(f"Suggested action: {redirect}")
                # Escalate to PI
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

        # Hard-coded: annotate_clusters always escalates to PI
        if step.step_type == "annotate_clusters" and not record.pi_escalated:
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
        checkpoints = [r for r in all_records if r.checkpoint_type]
        critical = sum(1 for r in all_records if r.llm_severity == "critical")

        return {
            "total_steps_monitored": total,
            "steps_with_technical_issues": with_issues,
            "pi_escalations": pi_escalations,
            "checkpoints_reached": [
                {"step": r.step_number, "type": r.checkpoint_type}
                for r in checkpoints
            ],
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

        language = "zh" if any('\u4e00' <= c <= '\u9fff' for c in analysis_state.initial_instruction) else "en"
        zh = language == "zh"

        async def _progress(pct: float, step: str):
            if progress_callback:
                await progress_callback({"progress": pct, "step": step, "status": "generating"})

        await _progress(5, "准备分析记录..." if zh else "Preparing analysis records...")

        # Collect all records
        all_records = self.get_supervision_log()
        stats = self.get_summary_stats()

        # Build structured analysis narrative for the LLM
        narrative = self._build_analysis_narrative(all_records, analysis_state, workspace_path)

        # Collect figure paths from workspace
        figure_map = self._collect_figure_paths(all_records, workspace_path)

        await _progress(15, "生成报告内容..." if zh else "Generating report content...")

        # Generate report sections using strong model
        report_prompt = self._build_report_prompt(
            narrative, stats, analysis_state, figure_map, language
        )

        try:
            response = await self.llm_client.chat(
                model=report_model,
                messages=[
                    {"role": "system", "content": self._get_report_system_prompt(language)},
                    {"role": "user", "content": report_prompt},
                ],
                temperature=0.3,
                max_tokens=32768,
            )
            report_content = response.content or ""
        except Exception as e:
            logger.error(f"[{self.session_id}] Report generation LLM call failed: {e}")
            report_content = self._fallback_report(narrative, stats, language)

        await _progress(60, "解析报告结构..." if zh else "Parsing report structure...")

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
            else:
                filtered_sections.append(sec)

        await _progress(70, "嵌入图表..." if zh else "Embedding figures...")

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

        await _progress(80, "构建报告..." if zh else "Building report...")

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
            "tables": [],
            "graphic_abstract_path": None,  # GraphicAbstract only has image_base64, no file path
            "generated_at": datetime.now().isoformat(),
            "status": "completed",
            "progress": 100,
            "current_step": "完成" if zh else "Complete",
            "appendix": appendix_data,
            "analysis_log": all_records,
        }

        # Sanitize NaN/Infinity
        report = self._sanitize(report)

        # Save JSON
        report_path = output_path / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        await _progress(90, "生成HTML报告..." if zh else "Generating HTML report...")

        # Generate HTML
        html_path = self._generate_html_report(
            report, filtered_sections, abstract, conclusions,
            report_figures, figure_map, stats, language, output_path, workspace_path,
        )
        report["html_path"] = str(html_path)

        await _progress(100, "完成" if zh else "Complete")
        return report

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

            # Track recent errors for stuck detection
            self._recent_errors.append(step.code_error[:200])
            if len(self._recent_errors) > 5:
                self._recent_errors = self._recent_errors[-5:]

            # Check for 3 consecutive identical errors
            if len(self._recent_errors) >= 3:
                last_3 = self._recent_errors[-3:]
                if all(e == last_3[0] for e in last_3):
                    issues.append(
                        "WARNING: Agent appears stuck — same error occurred 3 times in a row. "
                        "Consider a different approach."
                    )
        else:
            # Reset error streak on success
            if self._recent_errors:
                self._recent_errors = []

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

        return issues

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
            parts.append(f"Agent reasoning: {step.agent_reasoning[:500]}")
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
            parts.append(f"Annotation result: {step.annotation_result_summary[:300]}")
        if step.findings:
            parts.append(f"Findings: {'; '.join(step.findings[:3])}")

        text_context = "\n\n".join(parts)

        # Build effective system prompt: user-custom prompt prepended to built-in default
        effective_prompt = (
            f"{self.system_prompt}\n\n{_SUPERVISOR_SYSTEM_PROMPT}"
            if self.system_prompt
            else _SUPERVISOR_SYSTEM_PROMPT
        )

        # If we have figures, try vision evaluation
        if figures:
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

                response = await self.llm_client.chat_with_vision(
                    model=self.model,
                    text_prompt=f"{effective_prompt}\n\n{text_with_fig_note}",
                    images=compressed,
                    temperature=self.temperature,
                    max_tokens=512,
                )
                return self._parse_supervisor_response(response.content or "")

            except Exception as e:
                logger.warning(
                    f"[{self.session_id}] Vision evaluation failed, falling back to text-only: {e}"
                )
                # Fall through to text-only evaluation

        # Text-only evaluation
        response = await self.llm_client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": effective_prompt},
                {"role": "user", "content": text_context},
            ],
            temperature=self.temperature,
            max_tokens=512,
        )
        return self._parse_supervisor_response(response.content or "")

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
            parts.append(f"Output: {step.code_output[-500:]}")
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
            parts.append(f"\nAnnotation result: {step.annotation_result_summary[:500]}")
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
            pass
        # Try extracting from markdown code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, TypeError):
                pass
        # Try finding any JSON object in the text
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                pass
        # Default: no intervention
        return {
            "severity": "none", "issues": [], "action": "none", "message": "",
            "is_biological_checkpoint": False, "pi_question": "",
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
        lines.append(f"Total steps monitored: {len(records)}")
        lines.append("")

        for rec in records:
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

        # Also include publication figures
        if state.publication_figures:
            lines.append("=== Publication Figures ===")
            for pf in state.publication_figures:
                lines.append(f"- {pf.name}: {pf.description or ''}")
                if pf.png_path:
                    lines.append(f"  Path: {pf.png_path}")

        # Truncate to fit context
        full_text = "\n".join(lines)
        if len(full_text) > 60000:
            full_text = full_text[:60000] + "\n\n[... truncated ...]"
        return full_text

    def _collect_figure_paths(
        self, records: List[Dict], workspace_path: Optional[Path]
    ) -> Dict[str, Path]:
        """Collect all figure file paths from supervision records."""
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

        return figure_map

    def _build_report_prompt(
        self,
        narrative: str,
        stats: Dict,
        state: AnalysisState,
        figure_map: Dict[str, Path],
        language: str,
    ) -> str:
        """Build the prompt for the report-writing LLM."""
        zh = language == "zh"
        figure_list = "\n".join(
            f"- {name} ({self._categorize_figure(name)})" for name in figure_map.keys()
        ) or ("无" if zh else "None")

        # Collect round results
        round_findings = []
        for rr in state.round_results:
            all_findings = (
                rr.bioinformatics_findings + rr.pathology_findings +
                rr.biochemistry_findings + rr.literature_findings
            )
            if all_findings:
                round_findings.extend(all_findings[:5])

        findings_text = "\n".join(f"- {f}" for f in round_findings[:20]) or ("无" if zh else "None")

        prompt = f"""Based on the following complete analysis records, generate a detailed research report.

## Analysis Narrative
{narrative}

## Summary Statistics
- Total steps monitored: {stats['total_steps_monitored']}
- Steps with technical issues: {stats['steps_with_technical_issues']}
- PI escalations: {stats['pi_escalations']}
- Critical issues: {stats['critical_issues']}

## Key Findings
{findings_text}

## Available Figures
{figure_list}

## Instructions
{"请用中文写报告。" if zh else "Write the report in English."}
Generate the report with the following sections (use the exact section IDs in brackets):

[abstract] - Research summary (background, methods, key findings, conclusions)
[analysis_overview] - Analysis workflow overview (steps, checkpoints, issues)
[data_qc] - Data overview and quality control
[clustering_annotation] - Clustering and cell type annotation (include PI checkpoint guidance if available)
[differential_expression] - Differential expression analysis (include PI checkpoint guidance if available)
[key_findings] - Key biological findings
[supervisor_log] - Analysis monitoring log (technical issues, PI intervention points)
[conclusions] - Conclusions and future directions
[figure_appendix] - Figure descriptions (reference each figure by filename)

Format each section as:
## [section_id] Section Title
Content...

Make the report thorough but concise. Reference specific step numbers and figures where relevant.
"""
        return prompt

    def _get_report_system_prompt(self, language: str) -> str:
        zh = language == "zh"
        if zh:
            return (
                "你是一位资深生物信息学研究员，负责撰写单细胞RNA-seq分析的详细报告。"
                "报告应当专业、严谨、全面，包含分析过程的关键决策和PI的指导意见。"
                "使用Markdown格式，引用具体的图表文件名和分析步骤编号。"
            )
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

            sections.append({
                "id": section_id,
                "title": title,
                "content": section_content,
                "analysis_type": "general",
                "description": "",
                "figures": [],
                "tables": [],
            })

        if not sections:
            # Fallback: treat entire content as one section
            sections.append({
                "id": "full_report",
                "title": "Analysis Report" if language == "en" else "分析报告",
                "content": content,
                "analysis_type": "general",
                "description": "",
                "figures": [],
                "tables": [],
            })

        return sections

    def _fallback_report(self, narrative: str, stats: Dict, language: str) -> str:
        """Generate a minimal fallback report when LLM call fails."""
        zh = language == "zh"
        if zh:
            return f"""## [abstract] 摘要
本报告由自动化监控系统生成。分析过程共监控 {stats['total_steps_monitored']} 个步骤，
发现 {stats['steps_with_technical_issues']} 个技术问题，
PI介入 {stats['pi_escalations']} 次。

## [analysis_overview] 分析流程总览
{narrative[:3000]}

## [conclusions] 结论
报告生成模型调用失败，以上为原始分析记录。请参考分析日志获取完整信息。
"""
        return f"""## [abstract] Abstract
This report was generated by the automated monitoring system. Analysis monitored {stats['total_steps_monitored']} steps,
found {stats['steps_with_technical_issues']} technical issues,
with {stats['pi_escalations']} PI escalations.

## [analysis_overview] Analysis Overview
{narrative[:3000]}

## [conclusions] Conclusions
Report generation model call failed. Above is the raw analysis record. Refer to the analysis log for complete information.
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
            "supervision_stats": stats,
            "round_results": [
                {"round": rr.round_number, "findings_count": (
                    len(rr.bioinformatics_findings) + len(rr.pathology_findings) +
                    len(rr.biochemistry_findings) + len(rr.literature_findings)
                )}
                for rr in state.round_results
            ],
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

        zh = language == "zh"

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
            toc_items.append(
                f'<li><a href="#{html_module.escape(sec["id"])}">'
                f'{html_module.escape(sec["title"])}</a></li>'
            )

        # Build sections HTML
        sections_html = []
        for sec in sections:
            content_html = self._markdown_to_html(sec.get("content", ""))

            # Embed figures referenced in this section
            fig_html_parts = []
            for fig_name, fig_b64 in figure_b64_cache.items():
                if fig_name.lower() in sec.get("content", "").lower():
                    pdf_link = ""
                    fig_path = figure_map.get(fig_name)
                    if fig_path:
                        pdf_p = fig_path.with_suffix(".pdf")
                        if pdf_p.exists():
                            pdf_link = (
                                f'<a href="figures/{html_module.escape(pdf_p.name)}" '
                                f'class="download-link" download>Download PDF</a>'
                            )
                    fig_html_parts.append(f'''
                        <div class="figure-container">
                            <img src="data:image/png;base64,{fig_b64}"
                                 alt="{html_module.escape(fig_name)}" class="report-figure"/>
                            <div class="figure-caption">
                                {html_module.escape(fig_name)} {pdf_link}
                            </div>
                        </div>
                    ''')

            sec_html = f'''
            <div class="section" id="{html_module.escape(sec["id"])}">
                <h2>{html_module.escape(sec["title"])}</h2>
                <div class="section-content">{content_html}</div>
                {"".join(fig_html_parts)}
            </div>
            '''
            sections_html.append(sec_html)

        # Stats bar
        stats_html = f'''
        <div class="stats-bar">
            <div class="stat-item">
                <span class="stat-value">{stats["total_steps_monitored"]}</span>
                <span class="stat-label">{"步骤监控" if zh else "Steps Monitored"}</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">{stats["steps_with_technical_issues"]}</span>
                <span class="stat-label">{"技术问题" if zh else "Issues Found"}</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">{stats["pi_escalations"]}</span>
                <span class="stat-label">{"PI 介入" if zh else "PI Escalations"}</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">{stats["critical_issues"]}</span>
                <span class="stat-label">{"严重问题" if zh else "Critical Issues"}</span>
            </div>
        </div>
        '''

        title = report.get("title", "Analysis Report")

        html_content = f'''<!DOCTYPE html>
<html lang="{"zh" if zh else "en"}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_module.escape(title)}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 1000px; margin: 0 auto; padding: 20px; color: #333; line-height: 1.6; }}
    h1 {{ color: #1a365d; border-bottom: 3px solid #2b6cb0; padding-bottom: 10px; }}
    h2 {{ color: #2c5282; border-bottom: 1px solid #bee3f8; padding-bottom: 5px; margin-top: 2em; }}
    .abstract {{ background: #ebf8ff; border-left: 4px solid #3182ce; padding: 15px 20px;
                 margin: 20px 0; border-radius: 0 8px 8px 0; }}
    .stats-bar {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
    .stat-item {{ background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px;
                  padding: 15px 20px; text-align: center; flex: 1; min-width: 120px; }}
    .stat-value {{ display: block; font-size: 1.8em; font-weight: bold; color: #2b6cb0; }}
    .stat-label {{ display: block; font-size: 0.85em; color: #718096; margin-top: 4px; }}
    .section {{ margin: 30px 0; }}
    .section-content {{ margin: 10px 0; }}
    .section-content p {{ margin: 8px 0; }}
    .figure-container {{ text-align: center; margin: 20px 0; }}
    .report-figure {{ max-width: 100%; height: auto; border: 1px solid #e2e8f0;
                      border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .figure-caption {{ font-size: 0.9em; color: #718096; margin-top: 8px; }}
    .download-link {{ color: #3182ce; text-decoration: none; margin-left: 10px;
                      font-weight: 500; }}
    .download-link:hover {{ text-decoration: underline; }}
    pre {{ background: #f7fafc; border: 1px solid #e2e8f0; padding: 12px;
           border-radius: 6px; overflow-x: auto; font-size: 0.9em; }}
    code {{ background: #edf2f7; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
    pre code {{ background: none; padding: 0; }}
    .toc {{ background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px;
             padding: 15px 20px 15px 40px; margin: 20px 0; }}
    .toc li {{ margin: 5px 0; }}
    .toc a {{ color: #2b6cb0; text-decoration: none; }}
    .toc a:hover {{ text-decoration: underline; }}
    .conclusions {{ background: #f0fff4; border-left: 4px solid #38a169; padding: 15px 20px;
                    margin: 20px 0; border-radius: 0 8px 8px 0; }}
    .footer {{ text-align: center; color: #a0aec0; font-size: 0.85em; margin-top: 40px;
               padding-top: 20px; border-top: 1px solid #e2e8f0; }}
</style>
</head>
<body>
<h1>{html_module.escape(title)}</h1>

<div class="abstract">
<h3>{"摘要" if zh else "Abstract"}</h3>
{self._markdown_to_html(abstract)}
</div>

{stats_html}

<div class="toc">
<h3>{"目录" if zh else "Table of Contents"}</h3>
<ol>{"".join(toc_items)}</ol>
</div>

{"".join(sections_html)}

<div class="conclusions">
<h3>{"结论" if zh else "Conclusions"}</h3>
{self._markdown_to_html(conclusions)}
</div>

{self._build_appendix_html_section(report, zh)}

<div class="footer">
    {"由 InteSc Supervisor Agent 自动生成" if zh else "Auto-generated by InteSc Supervisor Agent"}
    &bull; {datetime.now().strftime("%Y-%m-%d %H:%M")}
</div>
</body>
</html>'''

        html_file = output_path / "report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Copy PDF figures to output for download links
        figures_out = output_path / "figures"
        figures_out.mkdir(exist_ok=True)
        import shutil
        for fig_name, fig_path in figure_map.items():
            try:
                # Copy PNG
                shutil.copy2(fig_path, figures_out / fig_name)
                # Copy PDF if exists
                pdf_path = fig_path.with_suffix(".pdf")
                if pdf_path.exists():
                    shutil.copy2(pdf_path, figures_out / pdf_path.name)
            except Exception as e:
                logger.warning(f"Could not copy figure {fig_name}: {e}")

        logger.info(f"HTML report generated: {html_file}")
        return html_file

    @staticmethod
    def _build_appendix_html_section(report: Dict, zh: bool) -> str:
        """Build appendix HTML section from report data."""
        import html as html_module
        appendix = report.get("appendix")
        if not appendix:
            return ""

        esc = html_module.escape

        parts = []
        parts.append(f'<div class="section" id="appendix">')
        parts.append(f'<h2>{"附录：分析设置与记录" if zh else "Appendix: Analysis Settings & Records"}</h2>')

        # Project Information
        parts.append(f'<h3>{"项目信息" if zh else "Project Information"}</h3>')
        parts.append('<table style="width:100%;border-collapse:collapse;margin-bottom:1em;">')
        parts.append(f'<tr><td style="padding:4px 8px;"><strong>{"项目名称" if zh else "Project Name"}</strong></td>'
                     f'<td style="padding:4px 8px;">{esc(appendix.get("project_name", "-"))}</td></tr>')
        if appendix.get("created_at"):
            parts.append(f'<tr><td style="padding:4px 8px;"><strong>{"创建时间" if zh else "Created At"}</strong></td>'
                         f'<td style="padding:4px 8px;">{esc(appendix.get("created_at", "-"))}</td></tr>')
        if report.get("generated_at"):
            parts.append(f'<tr><td style="padding:4px 8px;"><strong>{"报告生成时间" if zh else "Report Generated"}</strong></td>'
                         f'<td style="padding:4px 8px;">{esc(report.get("generated_at", "-"))}</td></tr>')
        parts.append('</table>')

        # Agent Configuration
        models = appendix.get("agent_models", {})
        temps = appendix.get("agent_temperatures", {})
        if models:
            parts.append(f'<h3>{"Agent 配置" if zh else "Agent Configuration"}</h3>')
            parts.append('<table style="width:100%;border-collapse:collapse;margin-bottom:1em;">')
            parts.append(f'<thead><tr><th style="text-align:left;padding:4px 8px;border-bottom:1px solid #e2e8f0;">Agent</th>'
                         f'<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #e2e8f0;">{"模型" if zh else "Model"}</th>'
                         f'<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #e2e8f0;">Temperature</th></tr></thead><tbody>')
            for key in sorted(models.keys()):
                parts.append(f'<tr><td style="padding:4px 8px;">{esc(key.title())}</td>'
                             f'<td style="padding:4px 8px;font-family:monospace;font-size:0.9em;">{esc(str(models[key]))}</td>'
                             f'<td style="padding:4px 8px;">{temps.get(key, "-")}</td></tr>')
            parts.append('</tbody></table>')

        # Data Files
        data_files = appendix.get("data_files", [])
        if data_files:
            parts.append(f'<h3>{"数据文件" if zh else "Data Files"}</h3>')
            parts.append('<table style="width:100%;border-collapse:collapse;margin-bottom:1em;">')
            parts.append(f'<thead><tr><th style="text-align:left;padding:4px 8px;border-bottom:1px solid #e2e8f0;">{"文件名" if zh else "Filename"}</th>'
                         f'<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #e2e8f0;">{"类型" if zh else "Type"}</th>'
                         f'<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #e2e8f0;">{"类别" if zh else "Section"}</th></tr></thead><tbody>')
            for f in data_files:
                parts.append(f'<tr><td style="padding:4px 8px;font-family:monospace;font-size:0.9em;">{esc(f.get("name", "-"))}</td>'
                             f'<td style="padding:4px 8px;">{esc(f.get("type", "-"))}</td>'
                             f'<td style="padding:4px 8px;">{esc(f.get("section", "-"))}</td></tr>')
            parts.append('</tbody></table>')

        # Treatment Groups
        groups = appendix.get("treatment_groups", [])
        if groups:
            parts.append(f'<h3>{"实验分组" if zh else "Treatment Groups"}</h3>')
            parts.append('<table style="width:100%;border-collapse:collapse;margin-bottom:1em;">')
            parts.append(f'<thead><tr><th style="text-align:left;padding:4px 8px;border-bottom:1px solid #e2e8f0;">{"组名" if zh else "Group"}</th>'
                         f'<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #e2e8f0;">{"描述" if zh else "Description"}</th>'
                         f'<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #e2e8f0;">{"样本数" if zh else "Samples"}</th></tr></thead><tbody>')
            for g in groups:
                parts.append(f'<tr><td style="padding:4px 8px;">{esc(g.get("name", "-"))}</td>'
                             f'<td style="padding:4px 8px;">{esc(g.get("description", "-"))}</td>'
                             f'<td style="padding:4px 8px;">{g.get("sample_count", 0)}</td></tr>')
            parts.append('</tbody></table>')

        # Analysis Rounds
        round_results = appendix.get("round_results", [])
        if round_results:
            parts.append(f'<h3>{"分析轮次" if zh else "Analysis Rounds"}</h3>')
            parts.append('<table style="width:100%;border-collapse:collapse;margin-bottom:1em;">')
            parts.append(f'<thead><tr><th style="text-align:left;padding:4px 8px;border-bottom:1px solid #e2e8f0;">{"轮次" if zh else "Round"}</th>'
                         f'<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #e2e8f0;">{"发现数" if zh else "Findings"}</th></tr></thead><tbody>')
            for rr in round_results:
                parts.append(f'<tr><td style="padding:4px 8px;">{rr.get("round", "-")}</td>'
                             f'<td style="padding:4px 8px;">{rr.get("findings_count", 0)}</td></tr>')
            parts.append('</tbody></table>')

        parts.append('</div>')
        return "\n".join(parts)

    @staticmethod
    def _markdown_to_html(text: str) -> str:
        """Simple markdown to HTML conversion for report content."""
        if not text:
            return ""
        import html as html_module
        text = html_module.escape(text)
        # Code blocks
        text = re.sub(
            r'```(\w*)\n(.*?)```',
            lambda m: f'<pre><code>{m.group(2)}</code></pre>',
            text, flags=re.DOTALL,
        )
        # Inline code
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
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
