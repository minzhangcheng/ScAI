"""Reporting mixin — final report, graphic abstract, and detailed report generation."""
import logging
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .state import WorkflowState
else:
    WorkflowState = dict

from ..models.schemas import (
    AnalysisState, AgentType, TaskStatus, GraphicAbstract,
)
from ..utils.llm_client import UpstreamExhaustedError
from ..constants import LlmFailureAction
from ..config import get_session_reports, get_session_workspace
from ..services.report import DetailedReportGenerator, compress_image_to_base64
from ..services.report.figure_discovery import discover_workspace_figures
from ..services.report.artifacts import artifact_stems, eligible_artifacts
from .direction_coverage import current_lineage_id, direction_coverage_status
from ..services.report.paths import resolve_report_output_dir
from ..services.report.lineage import build_lineage_report_state

# Re-exported so `backend.workflow.reporting.<name>` keeps resolving
# for existing importers and for tests that patch these by name.
from .reporting_parts.shared import (  # noqa: F401
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
from .reporting_parts import (  # noqa: F401
    FinalReportMixin,
    GraphicAbstractMixin,
)


logger = logging.getLogger(__name__)


class ReportingMixin(
    FinalReportMixin,
    GraphicAbstractMixin,
):
    """Mixin providing report generation node methods.

    Composed into SingleCellAnalysisWorkflow. Relies on self.session_id,
    self.pi_agent, self.painter_agent, self.bioinformatics_agent,
    self.supervisor_agent, self._settings_provider, self._vision_settings,
    self._send_message, self._send_state_sync, self._set_status,
    self._wait_for_llm_intervention, self._create_auto_snapshot,
    self.sandbox, and self.llm_manager.
    """

    # ------------------------------------------------------------------
    # Text-mode graphic abstract (Painter unavailable fallback)
    # ------------------------------------------------------------------

    _DEFAULT_TEXT_DIAGRAM_PROMPT = (
        "Draw an ASCII/Unicode flow diagram summarizing the study's key signaling "
        "pathway and cellular processes. Reason from the final summary provided and "
        "highlight ligand -> receptor -> downstream effector -> transcription factor "
        "-> cellular outcome relationships. Include at least 5 nodes and use "
        "directional arrows. English labels only.\n\n"
        "Study summary:\n{summary}"
    )

    async def _capture_post_report_if_enabled(
        self,
        analysis_state: AnalysisState,
        report_generation_id: str,
    ):
        """Run the required post-report capture unless Snapshot V2 is disabled."""
        from ..config import is_snapshot_v2_enabled
        from .helpers import WorkflowSupersededError

        if not is_snapshot_v2_enabled():
            analysis_state.pending_required_snapshot = None
            return None

        # Idempotent post-report snapshot with pending_required_snapshot state
        # machine. Failure UX is producer-aware; capture may be skipped without
        # regenerating reports, forfeiting only this restore point.
        if analysis_state.pending_required_snapshot is not None:
            logger.info(
                "[%s] Found pending_required_snapshot from previous attempt; retrying capture",
                self.session_id,
            )
            pending = analysis_state.pending_required_snapshot
            try:
                record = await self._capture_required_snapshot(
                    analysis_state,
                    pending.get("report_generation_id", report_generation_id),
                )
                analysis_state.pending_required_snapshot = None
            except WorkflowSupersededError:
                analysis_state.pending_required_snapshot = None
                raise
            except Exception:
                pending["attempts"] = pending.get("attempts", 0) + 1
                analysis_state.pending_required_snapshot = pending
                raise
        else:
            # Set pending state before capture, clear on success
            analysis_state.pending_required_snapshot = {
                "report_generation_id": report_generation_id,
                "phase": "post_report",
                "attempts": 0,
            }
            try:
                record = await self._capture_post_report_snapshot(
                    analysis_state, report_generation_id,
                )
                analysis_state.pending_required_snapshot = None
            except WorkflowSupersededError:
                # The capture handed the safe point to a restore, and that
                # restore committed. This run is over and `analysis_state` is
                # the pre-restore object the restore just replaced: leaving the
                # pending marker on it would arm a post-report retry gate for a
                # capture that never failed, on a state nobody should resume.
                analysis_state.pending_required_snapshot = None
                raise
            except Exception:
                # Keep pending so retry re-enters capture only
                raise
        return record

    async def _capture_post_report_snapshot(
        self,
        analysis_state: AnalysisState,
        report_generation_id: str,
    ):
        """Capture a required post-report snapshot (fail closed).

        Returns the SnapshotRecordV2. Raises RequiredSnapshotBlocked on failure.

        When no live sandbox/kernel is present (e.g. post-teardown rebuild),
        pre-acquire a trusted no-kernel safe-point descriptor so empty-complete
        runtime is authorized. Capture-side ``_create_safe_point`` alone is
        untrusted metadata and cannot authorize empty-complete.
        """
        from ..config import get_session_workspace
        from ..services.snapshot_v2.errors import RequiredSnapshotBlocked
        from ..services.snapshot_v2.service import SnapshotService
        from ..services.snapshot_v2.continuation import ContinuationRegistry
        from ..services.snapshot_v2.models import SnapshotCaptureRequest
        from ..services.snapshot_v2.lineage import capture_lineage_from_state
        from ..services.snapshot_v2.paths import SnapshotV2Paths

        service = getattr(self, "snapshot_service", None)
        if service is None:
            paths = SnapshotV2Paths.for_session(self.session_id)
            service = SnapshotService(self.session_id, paths)
            self.snapshot_service = service

        # A restore job reserves the safe point and then waits for a live
        # participant to hand it over at a boundary. The report nodes contain no
        # other acknowledgement site — grep `await_restore_safe_point` over the
        # report path — so this capture is the last one the reservation can
        # reach, and it used to be the one producer that refuses instead of
        # yielding. Both refusal routes end the same way: with no kernel the
        # guard below raises LIVE_SANDBOX_REQUIRED outright, and with a live
        # kernel the guard is skipped entirely and the acquire inside
        # `capture_required_snapshot` refuses with SAFE_POINT_BUSY (a *pending*
        # request and an acquired hold are indistinguishable to
        # `_acquire_unlocked`). Either becomes RequiredSnapshotBlocked, which
        # arms the fail-closed snapshot-failure gate on a run that had no
        # snapshot problem — and the restore that caused it, having lost its
        # only acknowledger, waits out its full timeout and fails too.
        #
        # Yield only for a *pending* request, exactly as the workflow-boundary
        # capture does. An already-acquired foreign hold must keep failing
        # closed so a leaked hold stays a visible gate rather than an unbounded
        # wait here.
        if service.safe_point.has_pending_request(self.session_id):
            import inspect

            # getattr/isawaitable is this file family's idiom for boundary hooks
            # and is required: tests bind this capture onto bare harnesses that
            # do not implement the hook.
            await_restore = getattr(self, "await_restore_safe_point", None)
            if callable(await_restore):
                result = await_restore()
                if inspect.isawaitable(result):
                    await result

        # Read the kernel only after the handover: the restore may have torn the
        # sandbox down while this capture was frozen.
        sandbox = getattr(self, "sandbox", None)
        kernel_present = (
            sandbox is not None
            and getattr(sandbox, "km", None) is not None
        )
        capture_sandbox = sandbox if kernel_present else None
        workflow_state = {
            "analysis_state": analysis_state,
            "current_round": analysis_state.current_round,
            "kernel_present": kernel_present,
        }

        # Idle / post-teardown: authorize empty-complete with a producer-owned
        # trusted descriptor (mirrors chat idle path). Foreign holds fail closed
        # so capture's release cannot steal another owner's pause.
        owned_sp = False
        sp = service.safe_point
        owner_token = f"report:{self.session_id}:{id(self)}"
        if not kernel_present:
            if sp.is_paused(self.session_id) or sp.has_pending_request(self.session_id):
                raise RequiredSnapshotBlocked(
                    "LIVE_SANDBOX_REQUIRED",
                    "Post-report capture cannot steal an existing safe-point hold",
                )
            if not sp.is_eligibility_trusted(self.session_id):
                from ..models.schemas import TaskStatus

                status = getattr(analysis_state, "status", None)
                terminal = status in (
                    TaskStatus.COMPLETED,
                    getattr(TaskStatus, "FAILED", None),
                ) or str(status) in ("completed", "failed")
                descriptor = {
                    "kernel_present": False,
                    "current_round": analysis_state.current_round or 0,
                    "inactive": True,
                }
                if terminal or status == TaskStatus.GENERATING_REPORT:
                    descriptor["status"] = (
                        "completed" if terminal else str(status or "generating_report")
                    )
                    if terminal:
                        descriptor["continuation_kind"] = "terminal"
                if not await sp.acquire(
                    self.session_id, descriptor, owner_token=owner_token,
                ):
                    raise RequiredSnapshotBlocked(
                        "LIVE_SANDBOX_REQUIRED",
                        "Post-report capture failed to acquire idle safe point",
                    )
                owned_sp = True

        try:
            capsule = ContinuationRegistry(service.object_store).freeze(
                "post_report", analysis_state, workflow_state,
            )
            request = SnapshotCaptureRequest(
                session_id=self.session_id,
                reason="post_report",
                phase="post_report",
                capsule=capsule,
                round_number=analysis_state.current_round or 0,
                idempotency_key=(
                    str(report_generation_id) if report_generation_id else None
                ),
                **capture_lineage_from_state(analysis_state),
            )
            workspace_root = str(get_session_workspace(self.session_id))
            record = await service.capture_required_snapshot(
                analysis_state,
                request,
                # A retained sandbox object with km=None has no capturable
                # runtime. Route it through the trusted terminal/no-kernel
                # path instead of re-declaring historical dependencies as
                # values that must be read from a nonexistent kernel.
                sandbox=capture_sandbox,
                workflow_state=workflow_state,
                workspace_root=workspace_root,
                safe_point_owner_token=owner_token if owned_sp else None,
            )
            logger.info(
                "[%s] Post-report snapshot created: %s (gen_id=%s)",
                self.session_id, record.snapshot_id, report_generation_id,
            )
            return record
        finally:
            if owned_sp:
                # Match chat idle cleanup: cancel/release only holds we own.
                await sp.cancel_safe_point_request(self.session_id, owner_token)

    async def _capture_required_snapshot(self, analysis_state, report_generation_id):
        """Retry a pending required snapshot using stored state (capture only)."""
        return await self._capture_post_report_snapshot(
            analysis_state, report_generation_id,
        )

    async def retry_pending_required_snapshot(self):
        """Re-enter capture only for a pending post-report snapshot.

        Does not re-run report generators. Clears pending on success.
        Raises RequiredSnapshotBlocked on failure.
        """
        analysis_state = (
            getattr(self, "state", None)
            or getattr(self, "_current_analysis_state", None)
            or getattr(self, "_state", None)
        )
        if analysis_state is None:
            raise RuntimeError("No analysis state available for snapshot retry")

        pending = getattr(analysis_state, "pending_required_snapshot", None)
        if not pending:
            return None

        report_generation_id = pending.get("report_generation_id") or getattr(
            analysis_state, "report_generation_id", None
        )
        record = await self._capture_required_snapshot(
            analysis_state, report_generation_id,
        )
        analysis_state.pending_required_snapshot = None
        return record
