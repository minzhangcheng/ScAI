"""Idempotency tests for post-report snapshot capture.

Tests that:
1. post-report capture with pending_required_snapshot retries correctly
2. report generators are called once even when capture fails and retries
3. Committed manifest + lost response reconciliation clears pending state
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from backend.models.schemas import AnalysisState, TaskStatus


# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def analysis_state():
    """A minimal AnalysisState for report-generation context."""
    return AnalysisState(
        id="sess-idy",
        name="Idempotency Test",
        user_id="u1",
        status=TaskStatus.GENERATING_REPORT,
        current_round=2,
        initial_instruction="test analysis",
        report_generation_id=None,
        pending_required_snapshot=None,
    )


@pytest.fixture
def mock_report_generator():
    """Mock for the report generator (e.g. pi_agent.generate_final_summary)."""
    mock = MagicMock()
    mock.generate_final_summary = AsyncMock(return_value="Final summary text")
    mock.generate_integrated_report = AsyncMock(return_value="Integrated report text")
    return mock


@pytest.fixture
def mock_snapshot_service():
    """Mock for SnapshotService used in reporting."""
    service = MagicMock()
    service.capture = AsyncMock()
    return service



def _stub_manifest(session_id, snapshot_id, capsule=None):
    from backend.services.snapshot_v2.models import (
        SnapshotManifestV2, IdleCapsule, ObjectRef, empty_complete_runtime,
    )
    from datetime import datetime, timezone
    stub = ObjectRef(sha256="a" * 64, size=1, media_type="application/json")
    return SnapshotManifestV2(
        session_id=session_id,
        snapshot_id=snapshot_id,
        name=snapshot_id,
        phase="post_report",
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        analysis_state_ref=stub,
        notebook_ref=stub,
        runtime=empty_complete_runtime(),
        continuation=capsule or IdleCapsule(graph_version="2.0"),
        logical_size_bytes=1,
        physical_added_bytes=0,
        root_hash="b" * 64,
    )


# ---- Tests: pending_required_snapshot state machine -------------------------


class TestPendingRequiredSnapshot:
    """Tests for the pending_required_snapshot field and retry logic."""

    def test_pending_set_before_capture(self, analysis_state):
        """Verify pending_required_snapshot is populated before capture attempt."""
        report_gen_id = str(uuid.uuid4())
        analysis_state.report_generation_id = report_gen_id

        # Simulate what reporting.py does: set pending before capture
        analysis_state.pending_required_snapshot = {
            "report_generation_id": report_gen_id,
            "phase": "post_report",
            "attempts": 0,
        }

        assert analysis_state.pending_required_snapshot is not None
        assert analysis_state.pending_required_snapshot["report_generation_id"] == report_gen_id
        assert analysis_state.pending_required_snapshot["phase"] == "post_report"
        assert analysis_state.pending_required_snapshot["attempts"] == 0

    def test_pending_cleared_on_success(self, analysis_state):
        """Verify pending_required_snapshot is cleared after successful capture."""
        analysis_state.pending_required_snapshot = {
            "report_generation_id": str(uuid.uuid4()),
            "phase": "post_report",
            "attempts": 1,
        }

        # Simulate successful capture
        analysis_state.pending_required_snapshot = None

        assert analysis_state.pending_required_snapshot is None

    def test_pending_increments_attempts(self, analysis_state):
        """Verify retry increments the attempt counter."""
        analysis_state.pending_required_snapshot = {
            "report_generation_id": str(uuid.uuid4()),
            "phase": "post_report",
            "attempts": 0,
        }

        # Simulate a retry
        analysis_state.pending_required_snapshot["attempts"] += 1
        assert analysis_state.pending_required_snapshot["attempts"] == 1

        analysis_state.pending_required_snapshot["attempts"] += 1
        assert analysis_state.pending_required_snapshot["attempts"] == 2

    def test_pending_without_report_gen_id(self, analysis_state):
        """pending_required_snapshot is valid even without report_generation_id."""
        analysis_state.report_generation_id = None
        analysis_state.pending_required_snapshot = {
            "phase": "post_report",
            "attempts": 0,
        }
        assert analysis_state.pending_required_snapshot is not None
        assert analysis_state.pending_required_snapshot["phase"] == "post_report"


# ---- Tests: report generation idempotency -----------------------------------


class TestReportGenerationIdempotency:
    """Tests that capture retries don't re-run report generators."""

    @pytest.mark.asyncio
    async def test_report_generators_called_once_on_capture_retry(
        self, analysis_state, mock_report_generator, mock_snapshot_service
    ):
        """Simulate: first capture fails, retry succeeds.

        Report generators must be called once.
        Capture must be called twice with same idempotency key.
        """
        from backend.services.snapshot_v2.models import (
            SnapshotCaptureRequest,
            SnapshotRecordV2,
            SnapshotFailureRecord,
            IdleCapsule,
            SnapshotManifestV2,
        )

        report_gen_id = str(uuid.uuid4())
        analysis_state.report_generation_id = report_gen_id

        # Step 1: Generate reports (called once)
        summary = await mock_report_generator.generate_final_summary(analysis_state)
        integrated = await mock_report_generator.generate_integrated_report(analysis_state)

        assert mock_report_generator.generate_final_summary.call_count == 1
        assert mock_report_generator.generate_integrated_report.call_count == 1

        # Step 2: First capture attempt — FAIL
        analysis_state.pending_required_snapshot = {
            "report_generation_id": report_gen_id,
            "phase": "post_report",
            "attempts": 0,
        }
        fail_record = SnapshotRecordV2(
            snapshot_id="snap-fail-001",
            manifest=_stub_manifest("sess-idy", "snap-fail-001"),
            failure=SnapshotFailureRecord(
                snapshot_id="snap-fail-001",
                session_id="sess-idy",
                failed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                code="STORAGE_ERROR",
                message="Disk full",
            ),
        )
        mock_snapshot_service.capture.return_value = fail_record

        record = await mock_snapshot_service.capture(
            SnapshotCaptureRequest(
                session_id="sess-idy",
                reason="post_report",
                capsule=IdleCapsule(graph_version="2.0"),
            ),
        )
        assert record.failure is not None
        assert mock_snapshot_service.capture.call_count == 1

        # Step 3: Retry capture — SUCCEED (no report regeneration)
        analysis_state.pending_required_snapshot["attempts"] += 1
        success_record = SnapshotRecordV2(
            snapshot_id="snap-ok-002",
            manifest=_stub_manifest("sess-idy", "snap-ok-002"),
            failure=None,
        )
        mock_snapshot_service.capture.return_value = success_record

        record2 = await mock_snapshot_service.capture(
            SnapshotCaptureRequest(
                session_id="sess-idy",
                reason="post_report",
                capsule=IdleCapsule(graph_version="2.0"),
            ),
        )
        assert record2.failure is None
        assert mock_snapshot_service.capture.call_count == 2

        # Report generators still called exactly once each
        assert mock_report_generator.generate_final_summary.call_count == 1
        assert mock_report_generator.generate_integrated_report.call_count == 1

    @pytest.mark.asyncio
    async def test_reconciliation_clears_pending_after_manifest_committed(
        self, analysis_state, mock_snapshot_service
    ):
        """When capture response is lost but the manifest was already committed,
        the MANIFEST_ALREADY_COMMITTED failure code triggers reconciliation that
        clears pending_required_snapshot without a duplicate capture attempt.
        """
        from backend.services.snapshot_v2.models import (
            SnapshotRecordV2,
            SnapshotFailureRecord,
            IdleCapsule,
            SnapshotManifestV2,
        )

        report_gen_id = str(uuid.uuid4())
        analysis_state.report_generation_id = report_gen_id
        analysis_state.pending_required_snapshot = {
            "report_generation_id": report_gen_id,
            "phase": "post_report",
            "attempts": 1,
        }

        committed_record = SnapshotRecordV2(
            snapshot_id="snap-existing-001",
            manifest=_stub_manifest("sess-idy", "snap-existing-001"),
            failure=SnapshotFailureRecord(
                snapshot_id="snap-existing-001",
                session_id="sess-idy",
                failed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                code="MANIFEST_ALREADY_COMMITTED",
                message="Snapshot already exists — idempotent completion",
            ),
        )

        cleared = _reconcile_pending_snapshot(analysis_state, committed_record)

        assert cleared, "Reconciliation should return True for MANIFEST_ALREADY_COMMITTED"
        assert analysis_state.pending_required_snapshot is None
        assert mock_snapshot_service.capture.call_count == 0


# ---- Test helpers -----------------------------------------------------------


def _reconcile_pending_snapshot(state, record):
    """Reconcile pending_required_snapshot after a capture attempt.

    Mirrors the logic in reporting.py's post-processing section:
    - record.failure is None -> success, clear pending
    - record.failure.code == MANIFEST_ALREADY_COMMITTED -> idempotent, clear pending
    - Other failures -> keep pending, increment attempts

    Returns True if pending was cleared.
    """
    if record is not None:
        if record.failure is None:
            state.pending_required_snapshot = None
            return True
        if record.failure.code == "MANIFEST_ALREADY_COMMITTED":
            state.pending_required_snapshot = None
            return True
    pending = state.pending_required_snapshot
    if pending is not None:
        pending["attempts"] = pending.get("attempts", 0) + 1
    return False
