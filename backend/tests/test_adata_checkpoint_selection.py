"""Plan source selection: current, copy, inheritance and stale handles."""
from __future__ import annotations

import asyncio
import hashlib


from backend.models.schemas import (
    AdataCheckpointRecord,
    AdataSourceSelection,
    AnalysisPlan,
    AnalysisState,
)
from backend.config import get_session_dir
from backend.services.adata_checkpoints.selection import (
    ensure_adata_source_prepared,
)


class _FakeSandbox:
    def __init__(self):
        self.km = object()
        self._exec_lock = None
        self.runtime_revision = 0
        self.adata_binding_registry_name = None
        self.apply_calls = []
        self.registry = {
            "registry_present": True,
            "canonical": {"adata_present": True, "same_as_anchor": True},
            "preparations": {},
            "bindings": {},
        }

    async def query_adata_registry_locked(self, *, registry_name=None):
        return {
            "registry_name": registry_name or "",
            "registry_version": 1,
            **self.registry,
        }

    async def apply_adata_source_locked(self, **kwargs):
        self.apply_calls.append(kwargs)
        return {
            "ok": True,
            "mode": kwargs.get("mode"),
            "binding_name": kwargs.get("binding_name"),
            "anchor_name": kwargs.get("anchor_name"),
            "adata_lineage_id": kwargs.get("adata_lineage_id"),
            "n_obs": 2,
            "n_vars": 3,
        }

    def _note_execution_effect(self, **_kwargs):
        self.runtime_revision += 1

    def remember_host_managed_names(self, names):
        self.registry.setdefault("managed", []).extend(names)

    async def _save_session_locked(self):
        return True

    def sync_runtime_persistence_state(self, state):
        return None


def _state_with_checkpoint() -> tuple[AnalysisState, AdataCheckpointRecord]:
    session_id = "selection-unit-session"
    state = AnalysisState(id=session_id, user_id="u")
    data = b"selection-unit-checkpoint"
    digest = hashlib.sha256(data).hexdigest()
    blobs = get_session_dir(session_id) / "adata-checkpoints" / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / f"{digest}.h5ad").write_bytes(data)
    record = AdataCheckpointRecord(
        checkpoint_id="cp-source",
        status="available",
        boundary="round_complete",
        boundary_event_id="ev-source",
        round_number=1,
        session_lineage_id="main",
        adata_lineage_id="lin-source",
        blob_sha256=digest,
        blob_relative_path=f"adata-checkpoints/blobs/{digest}.h5ad",
        size_bytes=len(data),
        n_obs=2,
        n_vars=3,
    )
    state.adata_checkpoints = [record]
    return state, record


def test_checkpoint_copy_registers_working_binding_without_rebinding_adata():
    async def _run():
        state, record = _state_with_checkpoint()
        sandbox = _FakeSandbox()
        plan = AnalysisPlan(phase="exploration")
        plan.id = "plan-copy"
        preparation = await ensure_adata_source_prepared(
            sandbox,
            state,
            AdataSourceSelection(mode="checkpoint_copy", checkpoint_id=record.checkpoint_id),
            plan=plan,
        )
        assert preparation.status == "applied_warm"
        assert sandbox.apply_calls[0]["mode"] == "checkpoint_copy"
        assert sandbox.apply_calls[0]["checkpoint_path"].endswith(".h5ad")
        assert state.adata_live_bindings
        assert plan.adata_source_preparation_id == preparation.preparation_id
        assert state.active_adata_checkpoint_id is None

    asyncio.run(_run())
