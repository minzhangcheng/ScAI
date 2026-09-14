"""Capture service contracts: boundary record, replay and immutable blob reuse."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.models.schemas import AnalysisState
from backend.services.adata_checkpoints.capture import capture_boundary_locked
from backend.services.adata_checkpoints.layout import blob_dir, verify_blob


class _FakeSandbox:
    def __init__(
        self, *, present: bool = True, with_km: bool = True,
        registration_ok: bool = True,
    ):
        self.km = object() if with_km else None
        self.runtime_revision = 0
        self.runtime_instance_id = "runtime-1"
        self._session_tmp = Path(tempfile.mkdtemp(prefix="scai-capture-"))
        self.adata_binding_registry_name = None
        self._present = present
        self._serialize_count = 0
        self.namespace_changes = 0
        self._registration_ok = registration_ok

    def _checkpoint_candidate_path(self) -> Path:
        return self._session_tmp / "candidate.h5ad"

    async def query_adata_registry_locked(self, *, registry_name=None):
        return {
            "registry_name": registry_name or "",
            "registry_present": False,
            "canonical": {
                "adata_present": self._present,
                "same_as_anchor": False,
                "binding_identity_status": "unknown",
                "anchor_name": None,
            },
        }

    async def serialize_adata_candidate_locked(self, candidate_path: Path):
        self._serialize_count += 1
        candidate_path.write_bytes(b"checkpoint-bytes")
        return {
            "ok": True,
            "reason": "",
            "n_obs": 3,
            "n_vars": 4,
            "size_bytes": candidate_path.stat().st_size,
        }

    async def register_canonical_anchor_locked(
        self, *, registry_name, anchor_name, adata_lineage_id, ancestor_checkpoint_id,
    ):
        return {
            "ok": self._registration_ok,
            "reason": None if self._registration_ok else "injected_failure",
            "registry_name": registry_name,
            "anchor_name": anchor_name,
        }

    def _note_execution_effect(self, *, mutates_namespace, reached_kernel):
        self.namespace_changes += 1
        self.runtime_revision += 1
        return True


def _state() -> AnalysisState:
    state = AnalysisState(user_id="u")
    state.id = "capture-test-session"
    return state


def _capture_kwargs(**overrides):
    kwargs = dict(
        boundary="round_complete",
        event_scope="round-1",
        round_number=1,
        planning_stage="full",
        phase="innovation",
        direction="direction-a",
        research_label="label",
        summary_at_capture="summary",
        outcome_at_capture="completed",
        execution_ids_at_capture=["exec-1"],
        attempt_execution_ids_at_capture=["exec-1", "attempt-2"],
    )
    kwargs.update(overrides)
    return kwargs


def test_capture_publishes_record_and_replays_same_event():
    async def _run():
        state = _state()
        sandbox = _FakeSandbox()
        first = await capture_boundary_locked(sandbox, state, **_capture_kwargs())
        assert first.available
        record = first.record
        assert record.status == "available"
        assert record.n_obs == 3 and record.n_vars == 4
        assert record.attempt_execution_id_count_at_capture == 2
        assert state.active_adata_checkpoint_id == record.checkpoint_id
        assert state.active_adata_lineage_id == record.adata_lineage_id
        assert state.adata_checkpoint_sequence == 1
        assert sandbox._serialize_count == 1
        blob = blob_dir(state.id) / f"{record.blob_sha256}.h5ad"
        verify_blob(
            blob,
            expected_sha256=record.blob_sha256,
            expected_size=record.size_bytes,
        )

        replay = await capture_boundary_locked(sandbox, state, **_capture_kwargs())
        assert replay.record.checkpoint_id == record.checkpoint_id
        assert sandbox._serialize_count == 1
        assert len(state.adata_checkpoints) == 1

    asyncio.run(_run())


def test_capture_discloses_without_canonical_adata():
    async def _run():
        state = _state()
        sandbox = _FakeSandbox(present=False)
        outcome = await capture_boundary_locked(sandbox, state, **_capture_kwargs())
        assert outcome.record is None
        assert outcome.disclosure == "no_canonical_adata"
        assert not state.adata_checkpoints
        assert sandbox._serialize_count == 0

    asyncio.run(_run())


def test_capture_discloses_without_live_kernel():
    async def _run():
        state = _state()
        sandbox = _FakeSandbox(with_km=False)
        outcome = await capture_boundary_locked(sandbox, state, **_capture_kwargs())
        assert outcome.record is None
        assert outcome.disclosure == "no_live_kernel"

    asyncio.run(_run())


def test_anchor_registration_failure_never_publishes_available_record():
    async def _run():
        state = _state()
        sandbox = _FakeSandbox(registration_ok=False)
        outcome = await capture_boundary_locked(
            sandbox, state, **_capture_kwargs()
        )
        assert outcome.disclosure == "failed_capture"
        assert outcome.record.status == "failed"
        assert outcome.record.blob_sha256 is None
        assert state.active_adata_checkpoint_id is None
        assert state.active_adata_lineage_id is None
        assert state.adata_checkpoint_sequence == 0
        assert [record.status for record in state.adata_checkpoints] == ["failed"]

    asyncio.run(_run())
