"""LLM-generated content for the 9-chapter detailed report.

Moved verbatim out of backend/agents/supervisor_agent.py; composed into SupervisorAgent.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Sequence
import asyncio
import json
import logging
import re
import uuid
from pathlib import Path

from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ...utils.error_parsing import compact_traceback

import sys
from . import report_presentation, report_prompt

logger = logging.getLogger(__name__)

from openai import APIConnectionError

from ...models.schemas import AnalysisState, AnalysisStep
from ...utils.llm_client import OpenAICompatibleClient, UpstreamExhaustedError
from ...services.report.interaction_records import write_interaction_record_files
from ...services.report.artifacts import (
    artifact_stems,
    eligible_artifacts,
    excluded_artifact_stems,
    publication_pdf_is_valid,
    resolve_archival_pdf,
    resolve_archival_render,
)
from ...services.report.figure_attribution import attribute_figures_to_rounds
from ...services.report.figure_placement import enforce_figure_placement
from ...services.report.figure_discovery import publication_figure_metadata
from ...services.report.report_outline import (
    GATE_SECTION_WORD_LIMIT,
    OutlineSection,
    align_sections_to_outline,
    build_report_outline,
    missing_outline_section_ids,
)
from ...services.report.output_cleanup import reset_report_asset_directories
from ...services.report.output_generation import (
    ReportGenerationError,
    ReportGenerationTransaction,
)
from ...utils.asyncio_tasks import run_in_owned_thread
from ...services.report.quality import (
    build_report_partial_fields,
    build_quality_disclosures,
)
from ..prompt_blocks import build_integration_block


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


class ReportContentMixin:
    """LLM-generated content for the 9-chapter detailed report."""

    async def generate_detailed_report(
        self,
        analysis_state: AnalysisState,
        report_model: str,
        output_dir: str,
        workspace_dir: str = None,
        progress_callback: Optional[Callable] = None,
        report_llm_client: Optional["OpenAICompatibleClient"] = None,
        chat_messages: Optional[List[Dict]] = None,
        force_fallback: bool = False,
    ) -> Dict[str, Any]:
        """Generate privately and atomically publish one complete report tree."""
        transaction = ReportGenerationTransaction(output_dir)
        try:
            report = await self._generate_detailed_report_staged(
                analysis_state=analysis_state,
                report_model=report_model,
                output_dir=str(transaction.staging_dir),
                workspace_dir=workspace_dir,
                progress_callback=progress_callback,
                report_llm_client=report_llm_client,
                chat_messages=chat_messages,
                force_fallback=force_fallback,
            )
            await run_in_owned_thread(transaction.commit_report, report)
            return report
        except BaseException:
            await run_in_owned_thread(transaction.abort)
            raise

    async def _generate_detailed_report_staged(
        self, analysis_state, report_model, output_dir, workspace_dir=None,
        progress_callback=None, report_llm_client=None, chat_messages=None,
        force_fallback=False,
    ) -> Dict[str, Any]:
        report_prompt.bind_facade(sys.modules[__name__])
        ctx = await report_prompt.prepare_report_generation(
            self, analysis_state, report_model, output_dir, workspace_dir,
            progress_callback, report_llm_client, force_fallback,
        )
        await report_prompt.parse_and_place_report(
            self, ctx, analysis_state, workspace_dir, chat_messages,
            progress_callback, report_model,
        )
        report_presentation.bind_facade(sys.modules[__name__])
        await report_presentation.build_report_figures(self, ctx, analysis_state)
        return await report_presentation.assemble_and_persist_report(
            self, ctx, analysis_state
        )

    @staticmethod
    def _build_figure_allocation_text(outline, figure_map, attribution) -> str:
        return report_prompt._build_figure_allocation_text(
            outline, figure_map, attribution
        )

    def _build_report_prompt(
        self, narrative, stats, state, figure_map, language,
        table_list=None, outline=None, attribution=None,
        figure_metadata_by_stem=None,
    ) -> str:
        report_prompt.bind_facade(sys.modules[__name__])
        return report_prompt._build_report_prompt(
            self, narrative, stats, state, figure_map, language,
            table_list=table_list, outline=outline, attribution=attribution,
            figure_metadata_by_stem=figure_metadata_by_stem,
        )

    def _get_report_system_prompt(self, language: str) -> str:
        return report_prompt._get_report_system_prompt(self, language)

    def _build_analysis_narrative(self, records, state, workspace_path) -> str:
        report_prompt.bind_facade(sys.modules[__name__])
        return report_prompt._build_analysis_narrative(
            self, records, state, workspace_path
        )

    def _build_step_context(self, step: AnalysisStep) -> str:
        """Build context string for a step (used in PI escalation for critical issues)."""
        parts = [f"Step {step.step_number} ({step.step_type})"]
        if step.code_output:
            parts.append(f"Output: {step.code_output}")
        if step.code_error:
            parts.append(
                f"Error: {compact_traceback(step.code_error, max_chars=300)}"
            )
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

    def _parse_report_sections(self, content, language, outline=None):
        report_prompt.bind_facade(sys.modules[__name__])
        return report_prompt._parse_report_sections(
            self, content, language, outline=outline
        )

    @staticmethod
    def _validate_report_content(report_content: str) -> bool:
        return report_prompt._validate_report_content(report_content)

    def _fallback_report(self, narrative: str, stats: Dict, language: str) -> str:
        return report_prompt._fallback_report(self, narrative, stats, language)
