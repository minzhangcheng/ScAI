"""Portability truthfulness for AnnData checkpoint blobs."""
from __future__ import annotations

import hashlib

from backend.models.schemas import AnalysisState, AdataCheckpointRecord
from backend.services.adata_checkpoints import (
    reconcile_catalog_availability,
    scrub_runtime_dependent_state,
)
from backend.routes.export_import_parts.import_archive_policy import (
    _resolve_import_dest,
)
from backend.config import get_session_dir


def _state_with_record(*, materialize: bool) -> tuple[AnalysisState, str]:
    session_id = f"portable-{hashlib.md5(str(materialize).encode()).hexdigest()[:10]}"
    data = b"portable-checkpoint-bytes"
    digest = hashlib.sha256(data).hexdigest()
    state = AnalysisState(id=session_id, user_id="u")
    state.adata_checkpoints = [
        AdataCheckpointRecord(
            checkpoint_id="cp-portable",
            status="available",
            boundary="round_complete",
            boundary_event_id="ev-portable",
            round_number=1,
            session_lineage_id="main",
            adata_lineage_id="lin-portable",
            blob_sha256=digest,
            blob_relative_path=f"adata-checkpoints/blobs/{digest}.h5ad",
            size_bytes=len(data),
            n_obs=2,
            n_vars=3,
        )
    ]
    if materialize:
        path = get_session_dir(session_id) / "adata-checkpoints" / "blobs"
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{digest}.h5ad").write_bytes(data)
    return state, digest


def test_missing_checkpoint_blob_is_marked_unavailable_not_materialized():
    state, _digest = _state_with_record(materialize=False)
    assert reconcile_catalog_availability(state, verify_digests=True) is True
    assert state.adata_checkpoints[0].status == "unavailable"
    assert state.adata_checkpoints[0].unavailable_reason == "not_materialized"


def test_materialized_checkpoint_blob_stays_available():
    state, _digest = _state_with_record(materialize=True)
    assert reconcile_catalog_availability(state, verify_digests=True) is False
    assert state.adata_checkpoints[0].status == "available"


def test_import_destination_maps_checkpoint_layout_under_session_root():
    destination = _resolve_import_dest(
        "adata-checkpoints/blobs/" + "a" * 64 + ".h5ad",
        "destination-session",
        "source-session",
    )
    assert destination == (
        get_session_dir("destination-session")
        / "adata-checkpoints"
        / "blobs"
        / ("a" * 64 + ".h5ad")
    )


def test_results_only_scrub_clears_continuation_not_history():
    payload = {
        "current_plan": {
            "id": "plan-1",
            "bioinformatics_run_scope_id": "run-old",
            "bioinformatics_settlement_scope_id": "settle-old",
        },
        "bioinformatics_settlement_attempt_ids": {"settle-old": ["e1"]},
        "adata_source_preparations": [
            {
                "preparation_id": "prep-1",
                "status": "durable",
                "registry_saved_name_ok": True,
                "binding_saved_names_ok": True,
            }
        ],
        "adata_live_bindings": [
            {"binding_id": "lb-1", "status": "registered"},
        ],
        "code_history": [
            {
                "id": "exec-1",
                "executor_run_kind": "ordinary",
                "executor_run_scope_id": "run-old",
                "executor_settlement_scope_id": "settle-old",
            }
        ],
        "sandbox_state": {
            "resolved_adata_source_context_v1": {"preparation_id": "prep-1"},
            "source_preparation_ack_v1": {"prep-1": "durable"},
        },
    }
    scrubbed = scrub_runtime_dependent_state(payload)
    assert scrubbed["current_plan"]["bioinformatics_run_scope_id"] is None
    assert scrubbed["current_plan"]["bioinformatics_settlement_scope_id"] is None
    assert scrubbed["bioinformatics_settlement_attempt_ids"] == {}
    assert scrubbed["adata_source_preparations"][0]["status"] == "unresolved"
    assert scrubbed["adata_live_bindings"][0]["status"] == "unresolved"
    assert "resolved_adata_source_context_v1" not in scrubbed["sandbox_state"]
    assert scrubbed["code_history"][0]["executor_run_scope_id"] == "run-old"
