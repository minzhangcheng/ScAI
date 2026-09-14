"""Model-level backward compatibility and public-payload guarantees."""
from __future__ import annotations

import json

import pytest

from backend.models.schemas import (
    AnalysisPlan,
    AnalysisState,
    AdataCheckpointRecord,
    AdataLiveBindingRecord,
    AdataSourcePreparation,
    AdataSourceSelection,
    canonical_id_set_digest,
    normalize_id_set,
)


def _available_record() -> AdataCheckpointRecord:
    digest = "a" * 64
    return AdataCheckpointRecord(
        status="available",
        boundary="round_complete",
        boundary_event_id="event-1",
        round_number=1,
        session_lineage_id="main",
        adata_lineage_id="lineage-1",
        blob_sha256=digest,
        blob_relative_path=f"adata-checkpoints/blobs/{digest}.h5ad",
        size_bytes=123,
        n_obs=10,
        n_vars=5,
        source_preparation_ids_at_capture=["prep-1"],
        source_preparation_id_count_at_capture=1,
        source_preparation_id_set_digest_at_capture=(
            "ee76f145edfb0b7a47f760bb871ecf7f"
            "92c48b924b12091ac4079dd7093abaa6"
        ),
    )


def test_checkpoint_record_requires_coherent_available_blob():
    record = _available_record()
    assert record.status == "available"
    with pytest.raises(ValueError):
        AdataCheckpointRecord(
            status="available",
            boundary="round_complete",
            boundary_event_id="event-2",
        )


def test_public_session_payload_excludes_catalog_paths_and_tokens():
    digest = "b" * 64
    record = _available_record().model_copy(
        update={
            "checkpoint_id": "cp_test",
            "blob_sha256": digest,
            "blob_relative_path": f"adata-checkpoints/blobs/{digest}.h5ad",
        }
    )
    state = AnalysisState(user_id="u")
    state.adata_checkpoints = [record]
    state.adata_source_preparations = [
        AdataSourcePreparation(
            owner_kind="plan",
            owner_id="plan-1",
            application_token="token-must-not-leak",
            selection=AdataSourceSelection(mode="checkpoint_copy", checkpoint_id="cp_test"),
        )
    ]
    state.adata_live_bindings = [
        AdataLiveBindingRecord(
            name="scai_work_7",
            anchor_name="scai_adata_anchor_7",
            mode_created="checkpoint_copy",
            adata_lineage_id="lineage-2",
            preparation_id="prep-1",
        )
    ]
    state.current_plan = AnalysisPlan(phase="exploration")
    state.current_plan.adata_source = AdataSourceSelection(
        mode="checkpoint_copy", checkpoint_id="cp_test"
    )

    payload = state.to_api_dict()
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "adata_checkpoints" not in payload
    assert "adata_source_preparations" not in payload
    assert "adata_live_bindings" not in payload
    assert "blob_relative_path" not in encoded
    assert "adata-checkpoints/blobs" not in encoded
    assert "token-must-not-leak" not in encoded
    # The approved plan itself remains reviewable and carries only the source
    # selection, never a Host path or application token.
    assert payload["current_plan"]["adata_source"]["mode"] == "checkpoint_copy"
    assert payload["current_plan"]["adata_source"]["checkpoint_id"] == "cp_test"


def test_bounded_execution_sample_accepts_more_than_200_ids():
    digest = "c" * 64
    ids = [f"exec-{index:04d}" for index in range(250)]
    normalized = normalize_id_set(ids)
    record = AdataCheckpointRecord(
        status="available",
        boundary="round_complete",
        boundary_event_id="event-many",
        round_number=1,
        session_lineage_id="main",
        adata_lineage_id="lineage-many",
        blob_sha256=digest,
        blob_relative_path=f"adata-checkpoints/blobs/{digest}.h5ad",
        size_bytes=1,
        n_obs=1,
        n_vars=1,
        execution_ids_at_capture=normalized[:200],
        execution_id_count_at_capture=len(normalized),
        execution_id_set_digest_at_capture=canonical_id_set_digest(normalized),
        attempt_execution_ids_at_capture=normalized[:200],
        attempt_execution_id_count_at_capture=len(normalized),
        attempt_execution_id_set_digest_at_capture=canonical_id_set_digest(normalized),
    )
    assert record.execution_id_count_at_capture == 250
    assert len(record.execution_ids_at_capture) == 200
    assert record.execution_id_set_digest_at_capture == canonical_id_set_digest(ids)


def test_complete_bounded_sample_rejects_noncanonical_or_wrong_digest():
    base = dict(
        status="available",
        boundary="round_complete",
        boundary_event_id="event-sample-validation",
        adata_lineage_id="lineage-sample-validation",
        blob_sha256="d" * 64,
        blob_relative_path=f"adata-checkpoints/blobs/{'d' * 64}.h5ad",
        size_bytes=1,
        n_obs=1,
        n_vars=1,
        execution_id_count_at_capture=2,
        execution_id_set_digest_at_capture=canonical_id_set_digest(["a", "b"]),
    )
    with pytest.raises(ValueError, match="sorted and unique"):
        AdataCheckpointRecord(
            **base, execution_ids_at_capture=["b", "a"],
        )
    with pytest.raises(ValueError, match="complete sample"):
        AdataCheckpointRecord(
            **{
                **base,
                "execution_id_set_digest_at_capture": "e" * 64,
            },
            execution_ids_at_capture=["a", "b"],
        )
