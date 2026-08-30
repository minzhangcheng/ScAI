"""Successful snapshots omit obsolete Snapshot V2 failure machinery."""

import json

from backend.models.schemas import AnalysisState
from backend.services.snapshot_v2.service_parts.capture_steps import CaptureStepsMixin


def _payload(state: AnalysisState) -> dict:
    return json.loads(CaptureStepsMixin._serialize_analysis_state(state))


def test_successful_capture_clears_old_snapshot_failure_from_copy_only():
    state = AnalysisState(
        id="cleanup",
        user_id="u",
        snapshot_failure={"code": "capture_failed", "message": "old failure"},
        snapshot_transaction={"transaction_id": "old", "status": "failed"},
        wait_input_reason="snapshot_failure",
    )

    payload = _payload(state)

    assert payload["snapshot_failure"] is None
    assert payload["snapshot_transaction"] is None
    assert payload["wait_input_reason"] is None
    assert state.snapshot_failure["code"] == "capture_failed"
    assert state.snapshot_transaction["status"] == "failed"
    assert state.wait_input_reason == "snapshot_failure"


def test_runtime_persistence_failure_remains_durable():
    state = AnalysisState(
        id="runtime-failure",
        user_id="u",
        snapshot_failure={
            "code": "runtime_persistence_kernel_state_unavailable",
            "message": "namespace was not persisted",
        },
        wait_input_reason="snapshot_failure",
    )

    payload = _payload(state)

    assert payload["snapshot_failure"]["code"].startswith("runtime_persistence_")
    assert payload["wait_input_reason"] == "snapshot_failure"


def test_nonfailed_transaction_and_unrelated_wait_reason_are_preserved():
    state = AnalysisState(
        id="active-transaction",
        user_id="u",
        snapshot_failure={"code": "capture_failed", "message": "old failure"},
        snapshot_transaction={"transaction_id": "active", "status": "capturing"},
        wait_input_reason="round_wait",
    )

    payload = _payload(state)

    assert payload["snapshot_failure"] is None
    assert payload["snapshot_transaction"]["status"] == "capturing"
    assert payload["wait_input_reason"] == "round_wait"
