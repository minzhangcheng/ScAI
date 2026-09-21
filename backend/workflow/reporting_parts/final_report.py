# Final report and detailed report generation nodes.
# Moved verbatim out of backend/workflow/reporting.py; composed into ReportingMixin.
"""Reporting mixin — final report, graphic abstract, and detailed report generation."""
import logging
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..state import WorkflowState
else:
    WorkflowState = dict

from ...models.schemas import (
    AnalysisState, AgentType, TaskStatus, GraphicAbstract,
    IntegratedReportPublication, PendingIntegratedReportGeneration,
)
from ...utils.llm_client import UpstreamExhaustedError
from ...constants import LlmFailureAction
from ..helpers import reset_unattended_gate_budget
from ..helpers import WorkflowTerminatedError
from ...config import get_session_reports, get_session_workspace
from ...services.report import DetailedReportGenerator, compress_image_to_base64
from ...services.report.figure_discovery import discover_workspace_figures
from ...services.report.artifacts import artifact_stems, eligible_artifacts
from ..direction_coverage import current_lineage_id, direction_coverage_status
from ...services.report.paths import resolve_report_output_dir
from ...services.report.lineage import build_lineage_report_state
from ...services.report.integrated_report import (
    integrated_report_catalog_fingerprint,
    integrated_report_source_fingerprint,
    sanitize_integrated_report,
)

from .shared import (  # noqa: F401
    logging,
    os,
    uuid,
    datetime,
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    AnalysisState,
    AgentType,
    TaskStatus,
    GraphicAbstract,
    UpstreamExhaustedError,
    LlmFailureAction,
    get_session_reports,
    get_session_workspace,
    DetailedReportGenerator,
    compress_image_to_base64,
    discover_workspace_figures,
    artifact_stems,
    eligible_artifacts,
    current_lineage_id,
    direction_coverage_status,
    resolve_report_output_dir,
    build_lineage_report_state,
    report_coverage_observation,
)


logger = logging.getLogger(__name__)

def detailed_report_event_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    """Build the terminal report event without rewriting a partial status."""
    return {
        "status": report.get("status", "completed"),
        "current_step": report.get("current_step", "Complete"),
        "report_id": report.get("id", ""),
        "title": report.get("title", ""),
        "sections_count": len(report.get("sections", [])),
        "figures_count": len(report.get("figures", [])),
        "tables_count": len(report.get("tables", [])),
        "partial_reason": report.get("partial_reason"),
        "partial_reasons": list(report.get("partial_reasons") or []),
        "missing_sections": list(report.get("missing_sections") or []),
        "figureless_sections": list(report.get("figureless_sections") or []),
        "unreported_figures": list(report.get("unreported_figures") or []),
        "quality_disclosures": report.get("quality_disclosures") or {},
    }


def analysis_completion_message(report: Optional[Dict[str, Any]]) -> str:
    """Describe terminal workflow completion without hiding a partial report."""
    if report and report.get("status") == "partial":
        detail = report.get("current_step") or report.get("partial_reason") or "partial"
        return f"Analysis completed; report is partial: {detail}"
    return "Analysis completed"


class FinalReportMixin:
    """Final report and detailed report generation nodes."""

    async def _generate_final_report(self, state: WorkflowState) -> WorkflowState:
        """Generate final analysis report including integrated report with figures"""
        analysis_state = state["analysis_state"]

        coverage_observation = report_coverage_observation(analysis_state)
        if coverage_observation:
            logger.warning(
                "[%s] Direction coverage observation recorded for PI review: %s",
                self.session_id,
                coverage_observation,
            )
            if analysis_state.round_results:
                note = "PI review observation: " + coverage_observation
                evidence_notes = analysis_state.round_results[-1].evidence_notes
                if note not in evidence_notes:
                    evidence_notes.append(note)
            await self._send_message(
                "status",
                "PI is reviewing incomplete direction coverage; report generation will proceed.",
                AgentType.PI,
            )

        if not analysis_state.analysis_log and not analysis_state.code_history:
            # current_round may be >0 even when no work happened (initialize_round
            # increments before the early-stop check), so use the log/history as
            # the real "did anything run" predicate.
            logger.info(f"[{self.session_id}] No analysis work recorded, short-circuiting report generation")
            self._set_status(analysis_state, TaskStatus.COMPLETED)
            await self._send_message("completed", {"reason": "stopped_before_execution"}, AgentType.PI)
            await self._send_state_sync(analysis_state, "analysis_complete")
            return {**state, "analysis_state": analysis_state}

        # Mark status so interrupted report generation can be detected on resume
        self._set_status(analysis_state, TaskStatus.GENERATING_REPORT)
        await self._send_state_sync(analysis_state, "report_generation_start")

        logger.info(f"[{self.session_id}] Generating final report after {analysis_state.current_round} rounds")

        # Step 1: Generate final summary (with LLM failure intervention)
        await self._send_message("status", "Generating final summary...", AgentType.PI)

        reset_unattended_gate_budget(self, "generate_final_report")
        while True:
            try:
                final_summary = await self.pi_agent.generate_final_summary(analysis_state)
                analysis_state.final_summary = final_summary
                await self._send_message(
                    "result",
                    {"final_summary": final_summary},
                    AgentType.PI
                )
                break
            except UpstreamExhaustedError as e:
                action = await self._wait_for_llm_intervention(
                    e, "generate_final_report", analysis_state
                )
                if action == LlmFailureAction.RETRY:
                    continue
                if action == LlmFailureAction.TERMINATE:
                    self._set_status(analysis_state, TaskStatus.STOPPED)
                    await self._send_state_sync(analysis_state, "final_report_terminated")
                    raise WorkflowTerminatedError("User terminated final summary generation") from e
                # Skip: use minimal fallback summary
                analysis_state.final_summary = (
                    "Analysis completed. Summary generation was skipped due to LLM failure."
                )
                await self._send_message(
                    "error",
                    "Final summary generation skipped due to LLM failure.",
                    AgentType.PI,
                )
                break
            except Exception as e:
                logger.exception(f"[{self.session_id}] Failed to generate final summary")
                analysis_state.final_summary = f"Summary generation failed: {str(e)}"
                await self._send_message(
                    "error",
                    f"Final summary generation failed: {str(e)}",
                    AgentType.PI,
                )
                break

        # Step 2: Generate integrated report with figure/table references
        await self._send_message("status", "Generating integrated report with figures...", AgentType.PI)

        integrated_report = None
        pending = analysis_state.integrated_report_state.pending
        report_lineage_id = current_lineage_id(analysis_state)
        source_fingerprint = integrated_report_source_fingerprint(
            analysis_state, report_lineage_id,
        )
        if (
            pending is None
            or pending.lineage_id != report_lineage_id
            or pending.source_fingerprint != source_fingerprint
        ):
            previous_pending = pending
            try:
                (
                    figure_refs,
                    reporting_limitations,
                    _selected_gallery_figures,
                ) = await self._prepare_brief_figures(analysis_state)
                previous_publication = analysis_state.integrated_report_state.published
                pending = PendingIntegratedReportGeneration(
                    base_publication_revision=(
                        previous_publication.publication_revision
                        if previous_publication else 0
                    ),
                    lineage_id=report_lineage_id,
                    figure_refs=figure_refs,
                    reporting_limitations=reporting_limitations,
                    catalog_fingerprint=integrated_report_catalog_fingerprint(
                        figure_refs
                    ),
                    source_fingerprint=integrated_report_source_fingerprint(
                        analysis_state, report_lineage_id,
                    ),
                )
                analysis_state.integrated_report_state.pending = pending
                await self._send_state_sync(
                    analysis_state,
                    "integrated_report_catalog_prepared",
                    raise_on_save_error=True,
                )
            except Exception as e:
                analysis_state.integrated_report_state.pending = previous_pending
                logger.exception(
                    f"[{self.session_id}] Failed to freeze integrated report figures"
                )
                await self._send_message(
                    "error",
                    f"Integrated report catalogue preparation failed: {str(e)}",
                    AgentType.PI,
                )
                pending = None

        reset_unattended_gate_budget(self, "generate_final_report")
        while pending is not None:
            try:
                integrated_report = await self.pi_agent.generate_integrated_report(
                    analysis_state,
                    pending.figure_refs,
                    pending.reporting_limitations,
                )
                break
            except UpstreamExhaustedError as e:
                action = await self._wait_for_llm_intervention(
                    e, "generate_final_report", analysis_state
                )
                if action == LlmFailureAction.RETRY:
                    continue
                if action == LlmFailureAction.TERMINATE:
                    self._set_status(analysis_state, TaskStatus.STOPPED)
                    await self._send_state_sync(analysis_state, "final_report_terminated")
                    raise WorkflowTerminatedError("User terminated integrated report generation") from e
                await self._send_message(
                    "error",
                    "Integrated report generation skipped due to LLM failure.",
                    AgentType.PI,
                )
                break
            except Exception as e:
                logger.exception(f"[{self.session_id}] Failed to generate integrated report")
                await self._send_message(
                    "error",
                    f"Integrated report generation failed: {str(e)}",
                    AgentType.PI,
                )
                break

        if integrated_report is not None:
            previous_report_state = analysis_state.integrated_report_state.model_copy(
                deep=True
            )
            previous_legacy_report = analysis_state.integrated_report
            try:
                assert pending is not None
                markdown = sanitize_integrated_report(
                    integrated_report,
                    pending.figure_refs,
                    pending.reporting_limitations,
                )
                publication = IntegratedReportPublication(
                    generation_id=pending.generation_id,
                    publication_revision=pending.base_publication_revision + 1,
                    lineage_id=pending.lineage_id,
                    markdown=markdown,
                    figure_refs=pending.figure_refs,
                    reporting_limitations=pending.reporting_limitations,
                    catalog_fingerprint=pending.catalog_fingerprint,
                )
                analysis_state.integrated_report_state.published = publication
                analysis_state.integrated_report_state.pending = None
                analysis_state.integrated_report = markdown
                # Persistence is the publication boundary. A websocket event is
                # emitted only after the body and mapping have committed together.
                await self._send_state_sync(
                    analysis_state,
                    "integrated_report_published",
                    raise_on_save_error=True,
                )
                await self._emit_integrated_report(analysis_state)
            except Exception as e:
                analysis_state.integrated_report_state = previous_report_state
                analysis_state.integrated_report = previous_legacy_report
                logger.exception(f"[{self.session_id}] Failed to assemble integrated report figures")
                await self._send_message(
                    "error",
                    f"Integrated report assembly failed: {str(e)}",
                    AgentType.PI,
                )

        logger.info(f"[{self.session_id}] Final report generated")

        return {
            **state,
            "analysis_state": analysis_state
        }

    async def _generate_detailed_report(self, state: WorkflowState) -> WorkflowState:
        """Generate detailed report with figures and tables"""
        analysis_state = state["analysis_state"]

        if not analysis_state.analysis_log and not analysis_state.code_history:
            logger.info(f"[{self.session_id}] No analysis work recorded, skipping detailed report")
            self._set_status(analysis_state, TaskStatus.COMPLETED)
            await self._send_message("completed", {"reason": "stopped_before_execution"}, AgentType.PI)
            await self._send_state_sync(analysis_state, "analysis_complete")
            return {**state, "analysis_state": analysis_state}

        logger.info(f"[{self.session_id}] Generating detailed report")
        await self._send_message(
            "status", "Generating detailed report with figures and tables...", AgentType.PI
        )

        # Reuse existing report_generation_id if present (R20 idempotency fix)
        report_generation_id = getattr(analysis_state, "report_generation_id", None)
        if not report_generation_id:
            report_generation_id = str(uuid.uuid4())
            analysis_state.report_generation_id = report_generation_id

        target_lineage_id = current_lineage_id(analysis_state)
        branch_id = None if target_lineage_id == "main" else target_lineage_id
        output_dir = str(resolve_report_output_dir(self.session_id, branch_id))
        os.makedirs(output_dir, exist_ok=True)

        workspace_dir = str(get_session_workspace(self.session_id))
        report_state = build_lineage_report_state(
            analysis_state,
            target_lineage_id,
            workspace_path=workspace_dir,
        )

        force_fallback = bool(
            analysis_state.detailed_report_fallback_pending
        )
        if force_fallback:
            # A prior process persisted the fallback handoff before it stopped.
            # Re-enter report generation without touching the failed upstream.
            analysis_state.wait_input_reason = None
            analysis_state.llm_failure_info = None
            self._set_status(analysis_state, TaskStatus.GENERATING_REPORT)
            await self._send_state_sync(
                analysis_state, "detailed_report_fallback_resume"
            )

        async def progress_callback(data: dict):
            await self._send_message(
                "progress",
                {
                    "step": data.get("step", ""),
                    "progress": data.get("progress", 0),
                    "status": data.get("status", "generating"),
                },
                AgentType.PI,
            )

        # === LLM call with retry + user intervention ===
        report: Optional[Dict[str, Any]] = None
        reset_unattended_gate_budget(self, "generate_detailed_report")
        while True:
            try:
                llm_client = self._settings_provider.get_llm_client('pi')
                pi_model = self._settings_provider.get_model("pi")
                bio_model = self._settings_provider.get_model("bioinformatics")
                self.ensure_agent_ready("supervisor")
                if self.supervisor_agent:
                    report = await self.supervisor_agent.generate_detailed_report(
                        analysis_state=report_state,
                        report_model=pi_model,
                        output_dir=output_dir,
                        workspace_dir=workspace_dir,
                        progress_callback=progress_callback,
                        report_llm_client=llm_client,
                        chat_messages=None,
                        force_fallback=force_fallback,
                    )
                else:
                    supervisor_model = self._settings_provider.get_model("supervisor")
                    generator = DetailedReportGenerator(
                        llm_client=llm_client,
                        pi_model=pi_model,
                        bio_model=bio_model,
                        output_dir=output_dir,
                        workspace_dir=workspace_dir,
                        progress_callback=progress_callback,
                        session_id=self.session_id,
                        vision_settings=self._vision_settings,
                        supervisor_model=supervisor_model,
                        target_lineage_id=target_lineage_id,
                        context_budget_callback=(
                            self._settings_provider.budget_text_dispatch
                        ),
                        context_budget_ack_callback=(
                            self._settings_provider.acknowledge_text_dispatch
                        ),
                    )
                    report = await generator.generate_report(
                        report_state,
                        force_fallback=force_fallback,
                    )

                await self._send_message(
                    "detailed_report",
                    detailed_report_event_payload(report),
                    AgentType.PI,
                )

                logger.info(f"[{self.session_id}] Detailed report generated successfully")
                break  # success

            except UpstreamExhaustedError as e:
                # A fallback requires an explicit Skip; an unresolved failure
                # must not authorize fallback after a process restart.
                action = await self._wait_for_llm_intervention(
                    e,
                    "generate_detailed_report",
                    analysis_state,
                )
                if action == LlmFailureAction.RETRY:
                    analysis_state.detailed_report_fallback_pending = False
                    force_fallback = False
                    self._set_status(analysis_state, TaskStatus.GENERATING_REPORT)
                    await self._send_state_sync(
                        analysis_state, "detailed_report_retry"
                    )
                    continue
                if action == LlmFailureAction.TERMINATE:
                    analysis_state.detailed_report_fallback_pending = False
                    self._set_status(analysis_state, TaskStatus.STOPPED)
                    await self._send_state_sync(
                        analysis_state, "detailed_report_terminated"
                    )
                    from ..core import WorkflowTerminatedError
                    raise WorkflowTerminatedError(
                        "User terminated workflow during detailed report generation"
                    )
                force_fallback = True
                analysis_state.detailed_report_fallback_pending = True
                self._set_status(analysis_state, TaskStatus.GENERATING_REPORT)
                await self._send_message(
                    "status",
                    "Detailed report LLM generation was skipped by the user; "
                    "building a deterministic fallback report.",
                    AgentType.PI,
                )
                await self._send_state_sync(
                    analysis_state, "detailed_report_fallback"
                )
                continue

            except Exception as e:
                logger.exception(f"[{self.session_id}] Failed to generate detailed report")
                await self._send_message(
                    "error",
                    f"Detailed report generation failed: {str(e)}. "
                    "Completed analysis results were retained; resume to retry the report.",
                    AgentType.PI,
                )
                break

        # === POST-PROCESSING: ALWAYS RUNS (even on Error) ===
        if report is None:
            # Preserve the failed node across graph END and process restarts.
            # Fallback remains authorized only by an explicit user Skip.
            analysis_state.sandbox_state = {
                **(analysis_state.sandbox_state or {}),
                "failed_node": "generate_detailed_report",
            }
            fallback_pending = analysis_state.detailed_report_fallback_pending
            self._set_status(
                analysis_state,
                TaskStatus.PAUSED if fallback_pending else TaskStatus.FAILED,
            )
            await self._send_state_sync(
                analysis_state,
                "detailed_report_fallback_failed" if fallback_pending else "detailed_report_failed",
            )
            return {
                **state, "analysis_state": analysis_state,
                "failed_node": "generate_detailed_report",
            }

        if (analysis_state.sandbox_state or {}).get("failed_node") == "generate_detailed_report":
            analysis_state.sandbox_state.pop("failed_node")
        analysis_state.detailed_report_fallback_pending = False
        self._set_status(analysis_state, TaskStatus.COMPLETED)
        await self._send_message(
            "completed", analysis_completion_message(report), AgentType.PI
        )

        await self._send_state_sync(analysis_state, "analysis_complete")

        logger.info(f"[{self.session_id}] Analysis workflow completed")

        return {
            **state,
            "analysis_state": analysis_state,
            "failed_node": None,
        }
