"""Plan source selection: current, copy, inheritance and stale handles."""
from __future__ import annotations

import asyncio
import hashlib

import pytest

from backend.models.schemas import (
    AdataCheckpointRecord,
    AdataSourceSelection,
    AnalysisPlan,
    AnalysisState,
)
from backend.config import get_session_dir
from backend.services.adata_checkpoints.errors import AdataSourcePreparationError
from backend.services.adata_checkpoints.selection import (
    ensure_adata_source_prepared,
    selection_signature,
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


def test_current_source_is_a_noop():
    async def _run():
        state = AnalysisState(id="current-noop", user_id="u")
        sandbox = _FakeSandbox()
        assert await ensure_adata_source_prepared(
            sandbox, state, AdataSourceSelection(mode="current")
        ) is None
        assert sandbox.apply_calls == []

    asyncio.run(_run())


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


def test_same_source_preparation_is_reused_without_second_apply():
    async def _run():
        state, record = _state_with_checkpoint()
        sandbox = _FakeSandbox()
        selection = AdataSourceSelection(
            mode="checkpoint_copy", checkpoint_id=record.checkpoint_id
        )
        plan = AnalysisPlan(phase="exploration")
        plan.id = "plan-copy-1"
        first = await ensure_adata_source_prepared(
            sandbox, state, selection, plan=plan
        )
        # Simulate the registry facts the first application would have written.
        sandbox.registry["preparations"][first.preparation_id] = {
            "token": first.application_token,
            "scope_id": first.scope_id,
            "selection_signature": first.selection_signature,
            "session_lineage_id": first.session_lineage_id,
            "binding_identity_status": "exact",
        }
        plan.id = "plan-copy-2"
        second = await ensure_adata_source_prepared(
            sandbox, state, selection, plan=plan
        )
        assert second.preparation_id == first.preparation_id
        assert len(sandbox.apply_calls) == 1

    asyncio.run(_run())


def test_stale_live_binding_is_reported_without_reload():
    async def _run():
        state, _record = _state_with_checkpoint()
        state.adata_live_bindings = []
        sandbox = _FakeSandbox()
        with pytest.raises(AdataSourcePreparationError) as exc_info:
            await ensure_adata_source_prepared(
                sandbox,
                state,
                AdataSourceSelection(mode="live_binding", live_binding_id="lb-missing"),
            )
        assert exc_info.value.reason == "binding_stale"
        assert sandbox.apply_calls == []

    asyncio.run(_run())
