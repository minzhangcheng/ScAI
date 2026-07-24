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
)
from ...utils.llm_client import UpstreamExhaustedError
from ...constants import LlmFailureAction
from ...config import get_session_reports, get_session_workspace
from ...services.report import DetailedReportGenerator, compress_image_to_base64
from ...services.report.figure_discovery import discover_workspace_figures
from ...services.report.artifacts import artifact_stems, eligible_artifacts
from ..direction_coverage import current_lineage_id, direction_coverage_status
from ...services.report.paths import resolve_report_output_dir
from ...services.report.lineage import build_lineage_report_state

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
    validate_complete_report_allowed,
)


logger = logging.getLogger(__name__)


class FinalReportMixin:
    """Final report and detailed report generation nodes."""

    def _after_final_report(self, state: WorkflowState) -> str:
        return "blocked" if state.get("report_blocked") else "continue"

    async def _generate_final_report(self, state: WorkflowState) -> WorkflowState:
        """Generate final analysis report including integrated report with figures"""
        analysis_state = state["analysis_state"]

        # Validate direction coverage before proceeding
        allowed, reason = validate_complete_report_allowed(analysis_state)
        if not allowed and not getattr(analysis_state, "allow_partial_report", False):
            self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
            await self._send_message("report_blocked", {"reason": reason, "partial_allowed": True}, AgentType.PI)
            await self._send_state_sync(analysis_state, "report_blocked")
            return {
                **state,
                "analysis_state": analysis_state,
                "should_continue": False,
                "report_blocked": True,
            }

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
        while True:
            try:
                integrated_report = await self.pi_agent.generate_integrated_report(analysis_state)
                break
            except UpstreamExhaustedError as e:
                action = await self._wait_for_llm_intervention(
                    e, "generate_final_report", analysis_state
                )
                if action == LlmFailureAction.RETRY:
                    continue
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
            try:
                analysis_state.integrated_report = integrated_report
                await self._assemble_brief_figures(analysis_state, integrated_report)
            except Exception as e:
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
        if state.get("report_blocked"):
            return state

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

        report_state = build_lineage_report_state(analysis_state, target_lineage_id)

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

        workspace_dir = str(get_session_workspace(self.session_id))

        # === LLM call with retry + user intervention ===
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
                    )
                    report = await generator.generate_report(report_state)

                await self._send_message(
                    "detailed_report",
                    {
                        "status": "completed",
                        "report_id": report.get("id", ""),
                        "title": report.get("title", ""),
                        "sections_count": len(report.get("sections", [])),
                        "figures_count": len(report.get("figures", [])),
                        "tables_count": len(report.get("tables", [])),
                    },
                    AgentType.PI,
                )

                logger.info(f"[{self.session_id}] Detailed report generated successfully")
                break  # success

            except UpstreamExhaustedError as e:
                action = await self._wait_for_llm_intervention(
                    e, "generate_detailed_report", analysis_state
                )
                if action == LlmFailureAction.RETRY:
                    continue
                if action == LlmFailureAction.TERMINATE:
                    from ..core import WorkflowTerminatedError
                    raise WorkflowTerminatedError(
                        "User terminated workflow during detailed report generation"
                    )
                await self._send_message(
                    "error",
                    "Detailed report generation skipped due to LLM failure. "
                    "Analysis is still complete.",
                    AgentType.PI,
                )
                break

            except Exception as e:
                logger.exception(f"[{self.session_id}] Failed to generate detailed report")
                await self._send_message(
                    "error",
                    f"Detailed report generation failed: {str(e)}. "
                    "Analysis is still complete.",
                    AgentType.PI,
                )
                break

        # === POST-PROCESSING: ALWAYS RUNS (even on Error) ===
        await self._capture_post_report_if_enabled(
            analysis_state,
            report_generation_id,
        )

        # Only set COMPLETED if snapshot capture succeeded or Snapshot V2 is
        # intentionally disabled. If capture failed, RequiredSnapshotBlocked is
        # raised before this point and the run remains in its failure gate.
        if analysis_state.pending_required_snapshot is None:
            self._set_status(analysis_state, TaskStatus.COMPLETED)
            await self._send_message("completed", "Analysis completed", AgentType.PI)

            await self._send_state_sync(analysis_state, "analysis_complete")

            logger.info(f"[{self.session_id}] Analysis workflow completed")
        else:
            logger.warning(
                "[%s] Post-report snapshot capture failed; "
                "keeping status GENERATING_REPORT for snapshot-failure HITL gate",
                self.session_id,
            )

        return {
            **state,
            "analysis_state": analysis_state,
        }
