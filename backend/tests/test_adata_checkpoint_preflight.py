"""Canonical alias repair before capture, using real AnnData serialization."""
import asyncio
from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import anndata as ad
import numpy as np
import pytest

from backend.agents.bioinformatics.checkpoint_preflight import select_checkpoint_adata
from backend.models.schemas import AnalysisState
from backend.services.adata_checkpoints.layout import blob_dir
from backend.services.adata_checkpoints.projection import (
    build_checkpoint_projection, build_prompt_catalog_text,
)
from backend.utils.sandbox_parts.adata_checkpoints import AdataCheckpointMixin
from backend.utils.sandbox_parts.shared import CodeExecutionResult
from backend.workflow.adata_checkpoint_hooks import capture_and_persist_boundary


class NamespaceSandbox(AdataCheckpointMixin):
    """Run the trusted bridge code in a namespace without a Jupyter process."""

    def __init__(self, tmp_path, **bindings):
        self.namespace = bindings
        self.km = object()
        self._exec_lock = asyncio.Lock()
        self.runtime_revision = 0
        self.runtime_instance_id = "preflight-test"
        self.adata_binding_registry_name = None
        self._session_tmp = tmp_path
        self.saved_alias = None

    async def _execute_unlocked(self, code, **kwargs):
        assert self._exec_lock.locked()
        output = io.StringIO()
        with redirect_stdout(output):
            exec(code, self.namespace)
        return CodeExecutionResult(success=True, output=output.getvalue())

    def _note_execution_effect(self, **kwargs):
        self.runtime_revision += 1

    async def _save_session_locked(self):
        assert self._exec_lock.locked()
        self.saved_alias = self.namespace.get("adata")

    def sync_runtime_persistence_state(self, state):
        pass


def agent_for(sandbox, name="adata_merged", during_selection=None):
    async def chat(prompt):
        assert not sandbox._exec_lock.locked(), "LLM must not hold the kernel lock"
        assert '"canonical_ready": false' in prompt
        if during_selection:
            during_selection()
        return json.dumps({"binding_name": name, "reason": "Primary completed object"})

    return SimpleNamespace(
        refresh_runtime_config_if_needed=Mock(),
        chat_without_history=AsyncMock(side_effect=chat),
        _safe_parse_json=lambda value, **kwargs: json.loads(value),
    )


@pytest.mark.asyncio
async def test_alias_is_captured_and_passed_to_runtime_save(tmp_path, monkeypatch):
    primary = ad.AnnData(np.arange(12).reshape(3, 4))
    primary.obs["annotation"] = ["a", "b", "c"]
    subset = primary[:1].copy()
    sandbox = NamespaceSandbox(tmp_path, adata_merged=primary, adata_subset=subset)
    agent = agent_for(sandbox)
    state = AnalysisState(user_id="u")
    save = AsyncMock()
    monkeypatch.setattr("backend.services.session_store.async_save_session", save)

    outcome = await capture_and_persist_boundary(
        SimpleNamespace(sandbox=sandbox, bioinformatics_agent=agent), state,
        boundary="stage1_complete", round_number=1, planning_stage="stage1",
        summary_at_capture="Completed primary annotations in adata_merged",
    )

    assert outcome.available
    assert sandbox.namespace["adata"] is primary
    assert sandbox.namespace["adata_merged"] is primary
    assert sandbox.namespace["adata_subset"] is subset
    assert sandbox.saved_alias is primary
    assert sandbox.runtime_revision >= 1
    save.assert_awaited_once()
    path = blob_dir(state.id) / f"{outcome.record.blob_sha256}.h5ad"
    restored = ad.read_h5ad(path)
    np.testing.assert_array_equal(restored.X, primary.X)
    assert restored.obs["annotation"].tolist() == ["a", "b", "c"]
    assert restored.shape == (3, 4)


@pytest.mark.asyncio
async def test_existing_canonical_is_preserved_without_model_selection(tmp_path):
    canonical = ad.AnnData(np.zeros((2, 3)))
    sandbox = NamespaceSandbox(tmp_path, adata=canonical, adata_merged=canonical.copy())
    agent = agent_for(sandbox)
    selected = await select_checkpoint_adata(
        agent, sandbox, AnalysisState(user_id="u"), boundary="round_complete", summary="",
    )
    assert selected is None
    agent.chat_without_history.assert_not_awaited()
    assert sandbox.namespace["adata"] is canonical


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [None, "not_in_inventory"])
async def test_missing_or_unidentified_primary_does_not_guess(tmp_path, choice):
    sandbox = NamespaceSandbox(tmp_path, adata_subset=ad.AnnData(np.zeros((1, 3))))
    agent = agent_for(sandbox, choice)
    outcome = await capture_and_persist_boundary(
        SimpleNamespace(sandbox=sandbox, bioinformatics_agent=agent),
        AnalysisState(user_id="u"), boundary="round_complete",
    )
    assert outcome.disclosure == "no_canonical_adata"
    assert "adata" not in sandbox.namespace


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["replace", "new_canonical", "revision", "view", "bad_shape"])
async def test_revalidation_prevents_stale_or_unsupported_alias(tmp_path, change, monkeypatch):
    original = ad.AnnData(np.zeros((2, 3)))
    sandbox = NamespaceSandbox(tmp_path, adata_merged=original)

    def mutate():
        if change == "replace":
            sandbox.namespace["adata_merged"] = original.copy()
        elif change == "new_canonical":
            sandbox.namespace["adata"] = original.copy()
        elif change == "revision":
            sandbox.runtime_revision += 1

    if change == "view":
        sandbox.namespace["adata_merged"] = original[:1]
    elif change == "bad_shape":
        original._X = np.zeros((1, 3))
    agent = agent_for(sandbox, during_selection=mutate)
    monkeypatch.setattr("backend.services.session_store.async_save_session", AsyncMock())
    outcome = await capture_and_persist_boundary(
        SimpleNamespace(sandbox=sandbox, bioinformatics_agent=agent),
        AnalysisState(user_id="u"), boundary="round_complete",
    )
    if change == "new_canonical":
        assert outcome.available
        assert sandbox.namespace["adata"] is not original
    else:
        assert outcome.disclosure == "no_canonical_adata"
        assert "adata" not in sandbox.namespace


@pytest.mark.asyncio
async def test_agent_failure_remains_a_nonblocking_capture_disclosure(tmp_path):
    sandbox = NamespaceSandbox(tmp_path, adata_merged=ad.AnnData(np.zeros((2, 3))))
    agent = agent_for(sandbox)
    agent.chat_without_history.side_effect = RuntimeError("provider unavailable")
    outcome = await capture_and_persist_boundary(
        SimpleNamespace(sandbox=sandbox, bioinformatics_agent=agent),
        AnalysisState(user_id="u"), boundary="round_complete",
    )
    assert outcome.disclosure == "no_canonical_adata"
    assert "adata" not in sandbox.namespace


def test_empty_catalog_does_not_claim_a_live_canonical_object():
    state = AnalysisState(user_id="u")
    entry = build_checkpoint_projection(state)["entries"][0]
    assert entry["availability"] == "requires_runtime_verification"
    assert "unverified" in entry["summary_at_capture"]
    assert "requires runtime verification" in build_prompt_catalog_text(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_kind", ["wrong_type", "view", "bad_shape", "backed"])
async def test_invalid_canonical_can_be_repaired_without_losing_original(
    tmp_path, monkeypatch, invalid_kind,
):
    primary = ad.AnnData(np.zeros((3, 4)))
    if invalid_kind == "wrong_type":
        invalid = {"not": "AnnData"}
    elif invalid_kind == "view":
        invalid = primary[:1]
    elif invalid_kind == "backed":
        source = tmp_path / "backed.h5ad"
        primary.write_h5ad(source)
        invalid = ad.read_h5ad(source, backed="r")
    else:
        invalid = primary.copy()
        invalid._X = np.zeros((1, 4))
    sandbox = NamespaceSandbox(tmp_path, adata=invalid, adata_merged=primary)
    agent = agent_for(sandbox)
    monkeypatch.setattr("backend.services.session_store.async_save_session", AsyncMock())
    try:
        outcome = await capture_and_persist_boundary(
            SimpleNamespace(sandbox=sandbox, bioinformatics_agent=agent),
            AnalysisState(user_id="u"), boundary="round_complete",
        )
        assert outcome.available
        assert sandbox.namespace["adata"] is primary
        preserved = [value for key, value in sandbox.namespace.items()
                     if key.startswith("adata_before_binding_repair_")]
        assert len(preserved) == 1 and preserved[0] is invalid
        agent.chat_without_history.assert_awaited_once()
    finally:
        if invalid_kind == "backed":
            invalid.file.close()


@pytest.mark.asyncio
async def test_invalid_canonical_repaired_during_selection_is_not_overwritten(tmp_path):
    from backend.agents.bioinformatics.checkpoint_preflight import ensure_checkpoint_adata

    primary = ad.AnnData(np.zeros((3, 4)))
    concurrent = primary.copy()
    sandbox = NamespaceSandbox(tmp_path, adata="invalid", adata_merged=primary)
    agent = agent_for(sandbox, during_selection=lambda: sandbox.namespace.update(adata=concurrent))
    result = await ensure_checkpoint_adata(
        agent, sandbox, AnalysisState(user_id="u"), boundary="before_figure_followup", summary="",
    )
    assert result["reason"] == "canonical_changed"
    assert sandbox.namespace["adata"] is concurrent


@pytest.mark.asyncio
async def test_followup_sees_alias_before_disposable_runtime_is_started(tmp_path):
    from backend.tests.test_figure_followup_regressions import setup_task
    from backend.workflow.post_execution import PostExecutionReviewMixin

    state, contract, task = setup_task()
    primary = ad.AnnData(np.zeros((3, 4)))
    sandbox = NamespaceSandbox(tmp_path, adata_merged=primary)
    agent = agent_for(sandbox)

    async def render(_state, instruction, **kwargs):
        # The real executor pins the owner generation at this entry point.
        assert sandbox.namespace["adata"] is primary
        assert sandbox.runtime_revision > 0
        assert "does not establish an adata_annotated" in instruction
        return {"status": "completed", "analysis_log": [], "code_history": []}

    agent.run_autonomous_analysis = AsyncMock(side_effect=render)
    host = PostExecutionReviewMixin()
    host.session_id = state.id
    host.sandbox = sandbox
    host.bioinformatics_agent = agent
    host.state_callback = AsyncMock(return_value=True)
    host._figure_review_callback = AsyncMock()
    host.pi_agent = SimpleNamespace(
        decide_figure_followups=AsyncMock(return_value=[{
            "task_id": task.id, "disposition": "targeted_redraw",
            "instruction": "Plot completed results", "rationale": "Available results",
            "report_note": "",
        }]),
        select_fallback_figure=AsyncMock(return_value={}),
    )
    await host._run_figure_followup(state, contract, "full")
    agent.run_autonomous_analysis.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_canonical_without_replacement_cannot_publish_checkpoint(tmp_path):
    invalid = ad.AnnData(np.zeros((3, 4)))
    invalid._X = np.zeros((1, 4))
    sandbox = NamespaceSandbox(tmp_path, adata=invalid)
    agent = agent_for(sandbox, None)
    from backend.agents.bioinformatics.checkpoint_preflight import ensure_checkpoint_adata

    await ensure_checkpoint_adata(
        agent, sandbox, AnalysisState(user_id="u"), boundary="round_complete", summary="",
    )
    async with sandbox._exec_lock:
        result = await sandbox.serialize_adata_candidate_locked(tmp_path / "invalid.h5ad")
    assert result == {"ok": False, "reason": "inconsistent_matrix_shape"}
    assert not (tmp_path / "invalid.h5ad").exists()
    assert sandbox.namespace["adata"] is invalid
