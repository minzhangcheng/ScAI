"""Retired Stage 1 foundation gate must not remain reachable."""
from __future__ import annotations

import importlib
import pathlib

import pytest

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


def test_retired_foundation_module_is_absent():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.services.foundation_integrity")


def test_executor_no_longer_references_foundation_symbols():
    executor = (REPO / "backend/agents/bioinformatics/executor_execute.py").read_text()
    for name in FOUNDATION_NAMES:
        assert name not in executor
    supplementary = (
        REPO / "backend/agents/bioinformatics/supplementary_support.py"
    ).read_text()
    for name in (
        "foundation_write_guard",
        "foundation_is_sealed",
        "foundation_manifest",
        "classify_foundation_signature",
        "accept_foundation_extensions",
        "clear_foundation_bindings",
        "commit_foundation_bindings",
        "mark_foundation_drift",
    ):
        assert name not in supplementary


def test_supplementary_pass_has_no_signature_abort():
    supplementary = (
        REPO / "backend/agents/bioinformatics/supplementary.py"
    ).read_text()
    assert "capture_main_object_signature" not in supplementary
    assert "_supplementary_fatal_signature_changes" not in supplementary
    support = (
        REPO / "backend/agents/bioinformatics/supplementary_support.py"
    ).read_text()
    assert "signatures_match" not in support
    assert "capture_main_object_signature" not in support


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


def test_stage2_instruction_does_not_claim_a_sealed_canonical_object():
    from backend.workflow.agents import build_phase_instruction

    text = build_phase_instruction(
        planning_stage="stage2",
        analysis_phase="exploration",
        has_bulk=False,
        has_scrna=True,
        direction="",
    )
    assert "live, sealed canonical" not in text
    assert "Do NOT reload or rebind `adata`" not in text
    assert "Host-resolved source binding" in text
