"""Snapshot-failure "Skip snapshot & continue" gate handlers.

Skip clears the failure gate WITHOUT capturing a snapshot and lets the analysis
continue with reduced restorability at that boundary. It is producer-aware:
workflow_boundary / post_report / chat are skippable; pre_restore is not (skipping
would proceed with a restore that overwrites live state without a safety point).
"""
import pytest

from backend.models.schemas import AnalysisState, TaskStatus


def _state(**kw) -> AnalysisState:
    return AnalysisState(id="sess-skip", name="T", user_id="u1",
                         initial_instruction="analyze", **kw)


def _gated(**kw) -> AnalysisState:
    st = _state(**kw)
    st.status = TaskStatus.WAITING_INPUT
    st.wait_input_reason = "snapshot_failure"
    st.snapshot_failure = {"code": "required_runtime_unpersistable", "message": "x"}
    return st


class TestSkipHandlers:
    def test_skip_workflow_boundary_arms_one_shot_and_resumes_running(self):
        from backend.services.snapshot_v2.failure_gate import skip_workflow_boundary

        st = _gated()
        skip_workflow_boundary(st, phase="pre_annotation")
        assert st.snapshot_failure is None
        assert st.wait_input_reason is None
        assert st.status == TaskStatus.RUNNING
        assert st.skip_required_snapshot_phase == "pre_annotation"

    def test_skip_post_report_completes_and_clears_pending(self):
        from backend.services.snapshot_v2.failure_gate import skip_post_report

        st = _gated()
        st.pending_required_snapshot = {"phase": "post_report"}
        skip_post_report(st)
        assert st.snapshot_failure is None
        assert st.wait_input_reason is None
        assert st.status == TaskStatus.COMPLETED
        assert st.pending_required_snapshot is None

    def test_skip_chat_restores_previous_status(self):
        from backend.services.snapshot_v2.failure_gate import skip_chat

        st = _gated()
        skip_chat(st, "completed")
        assert st.snapshot_failure is None
        assert st.wait_input_reason is None
        assert st.status == TaskStatus.COMPLETED


class TestConsumeSnapshotSkip:
    def test_matching_phase_is_consumed_once(self):
        from backend.services.snapshot_v2.failure_gate import consume_snapshot_skip

        st = _state()
        st.skip_required_snapshot_phase = "pre_annotation"
        assert consume_snapshot_skip(st, "pre_annotation") is True
        # One-shot: marker cleared, second boundary is not skipped.
        assert st.skip_required_snapshot_phase is None
        assert consume_snapshot_skip(st, "pre_annotation") is False

    def test_non_matching_phase_not_consumed(self):
        from backend.services.snapshot_v2.failure_gate import consume_snapshot_skip

        st = _state()
        st.skip_required_snapshot_phase = "pre_annotation"
        assert consume_snapshot_skip(st, "pre_direction") is False
        assert st.skip_required_snapshot_phase == "pre_annotation"

    def test_wildcard_matches_any_phase(self):
        from backend.services.snapshot_v2.failure_gate import consume_snapshot_skip

        st = _state()
        st.skip_required_snapshot_phase = "*"
        assert consume_snapshot_skip(st, "anything") is True
        assert st.skip_required_snapshot_phase is None

    def test_no_marker_returns_false(self):
        from backend.services.snapshot_v2.failure_gate import consume_snapshot_skip

        st = _state()
        assert consume_snapshot_skip(st, "pre_annotation") is False


class TestAllowedActionsOfferSkip:
    def test_workflow_boundary_gate_offers_skip(self):
        from backend.services.snapshot_v2.failure_gate import (
            ensure_workflow_boundary_retry_context,
        )
        from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked

        st = _state()
        err = RequiredSnapshotBlocked(
            "required_runtime_unpersistable", "x", phase="pre_annotation"
        )
        ensure_workflow_boundary_retry_context(st, err, previous_status="running")
        assert "skip" in (st.snapshot_failure or {}).get("allowed_actions", [])

    def test_post_report_gate_offers_skip(self):
        from backend.services.snapshot_v2.failure_gate import (
            ensure_post_report_retry_context,
        )

        st = _state()
        ensure_post_report_retry_context(st, previous_status="generating_report")
        assert "skip" in (st.snapshot_failure or {}).get("allowed_actions", [])

    def test_chat_gate_offers_skip(self):
        from backend.services.snapshot_v2.failure_gate import (
            enter_chat_snapshot_failure_gate,
        )
        from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked

        st = _state()
        err = RequiredSnapshotBlocked("required_runtime_unpersistable", "x")
        enter_chat_snapshot_failure_gate(st, err, "completed")
        assert "skip" in (st.snapshot_failure or {}).get("allowed_actions", [])

    def test_pre_restore_gate_does_not_offer_skip(self):
        """pre_restore uses the base (no-skip) gate — skipping is unsafe there."""
        from backend.services.snapshot_v2.failure_gate import enter_snapshot_failure_gate

        st = _state()
        enter_snapshot_failure_gate(st, {"code": "X", "message": "y"})
        assert "skip" not in (st.snapshot_failure or {}).get("allowed_actions", [])


class TestSkipDispatch:
    def test_dispatch_workflow_boundary(self):
        from backend.services.snapshot_v2.failure_gate import skip_snapshot_producer
        from backend.services.snapshot_v2.models import SnapshotRetryContext

        st = _gated()
        ctx = SnapshotRetryContext(
            producer="workflow_boundary",
            resume_policy="graph_checkpoint",
            snapshot_request={"phase": "pre_annotation"},
            previous_status="running",
        )
        outcome = skip_snapshot_producer(st, ctx)
        assert outcome.success is True
        assert outcome.still_blocked is False
        assert st.status == TaskStatus.RUNNING
        assert st.skip_required_snapshot_phase == "pre_annotation"

    def test_dispatch_pre_restore_is_unsupported(self):
        from backend.services.snapshot_v2.failure_gate import skip_snapshot_producer
        from backend.services.snapshot_v2.models import SnapshotRetryContext

        st = _gated()
        ctx = SnapshotRetryContext(
            producer="pre_restore",
            resume_policy="restore_transaction",
            snapshot_request={"phase": "pre_restore"},
            previous_status="running",
        )
        outcome = skip_snapshot_producer(st, ctx)
        assert outcome.success is False
        # Gate stays intact for an unsupported skip.
        assert st.snapshot_failure is not None
