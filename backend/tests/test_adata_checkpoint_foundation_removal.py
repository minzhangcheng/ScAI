"""Retired Stage 1 foundation gate must not remain reachable."""
from __future__ import annotations

import pathlib


from backend.models.schemas import AnalysisState
from backend.services.legacy_state_sanitizer import sanitize_legacy_state


REPO = pathlib.Path(__file__).resolve().parents[2]
FOUNDATION_NAMES = (
    "foundation_integrity",
    "foundation_write_guard",
    "seal_foundation",
    "accept_foundation_extensions",
    "mark_foundation_drift",
    "foundation_is_sealed",
    "foundation_manifest",
    "classify_foundation_signature",
    "registered_derived_names",
)


def test_legacy_sanitizer_removes_all_foundation_and_closure_latches():
    state = AnalysisState(user_id="u")
    state.sandbox_state = {
        "runtime_revision": 1,
        "saved_runtime_revision": 1,
        "stage1_foundation_v1": {"status": "sealed"},
        "stage1_foundation_bindings_v1": {"derived": ["work"]},
        "stage2_foundation_integrity_failure_v1": {"error": "old"},
        "stage2_foundation_integrity_baseline_v1": {"records": []},
        "stage2_foundation_bindings_v1": {"derived": []},
        "analysis_closure_integrity_failure_v1": {"error": "old"},
    }

    assert sanitize_legacy_state(state, workflow_active=False) is True
    for key in (
        "stage1_foundation_v1",
        "stage1_foundation_bindings_v1",
        "stage2_foundation_integrity_failure_v1",
        "stage2_foundation_integrity_baseline_v1",
        "stage2_foundation_bindings_v1",
        "analysis_closure_integrity_failure_v1",
    ):
        assert key not in state.sandbox_state
    # Unrelated runtime observations remain untouched.
    assert state.sandbox_state["runtime_revision"] == 1
