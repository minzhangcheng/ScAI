"""Unregistered method notes retain image identity without granting publication."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.agents import bioinformatics_agent as facade
from backend.agents.bioinformatics import executor_execute
from backend.models.schemas import AnalysisState, CodeExecution
from backend.services.report.artifacts import ARTIFACT_REGISTRY_VERSION
from backend.services.report.figure_discovery import resolve_state_figure_candidates
from backend.services.report.figure_methods import (
    annotate_legacy_figure_methods, find_figure_method_annotation,
)
from backend.tests.figure_image_helpers import write_valid_image


@pytest.mark.asyncio
async def test_execute_notes_are_version_bound_and_do_not_register_an_image(tmp_path):
    path = tmp_path / "figures" / "contrast.jpg"
    path.parent.mkdir()
    write_valid_image(path, min_bytes=2200)
    execution = CodeExecution(code="plot()")
    agent = SimpleNamespace(code_history=[], sandbox=SimpleNamespace(working_dir=tmp_path), _notify=AsyncMock())
    context = SimpleNamespace(agent=agent, analysis_log=[], step_count=1, agent_reasoning=None, all_figures=[], step_figures=[])
    result = SimpleNamespace(output=f"[Auto-saved] JPG: {path}", error=None, figures=[])
    executor_execute.bind_facade(facade)
    await executor_execute.record_execution_result(
        context, execution, result, execution.code, "Plot", "technical", "planned", [],
        figure_analysis_metadata=[{"filename": path.name, "analysis_metadata": {"methods": "~condition"}}],
    )
    state = AnalysisState(user_id="u", code_history=agent.code_history, artifact_registry_version=ARTIFACT_REGISTRY_VERSION)
    state = AnalysisState.model_validate_json(state.model_dump_json())
    note = state.code_history[0].figure_method_annotations[0]
    assert note.basis == "execution_context" and note.evidence_execution_ids == [execution.id]
    assert resolve_state_figure_candidates(state, tmp_path, report_only=True) == []
    assert not state.publication_figures and not state.legacy_artifacts
    assert find_figure_method_annotation(state, note.relative_path, "different-bytes") is None
    state.restore_branch_id = "other-branch"
    assert find_figure_method_annotation(state, note.relative_path, note.render_sha256) is None


@pytest.mark.asyncio
async def test_historical_notes_are_inferred_persisted_and_reused(tmp_path):
    path = tmp_path / "figures" / "old.jpg"
    path.parent.mkdir()
    write_valid_image(path, min_bytes=2200)
    execution = CodeExecution(code="plt.savefig('old.jpg')", output="A plotted contrast")
    state = AnalysisState(user_id="u", code_history=[execution])

    async def annotate(candidates):
        assert candidates[0]["candidate_executions"][0]["execution_id"] == execution.id
        return [{"figure_id": candidates[0]["figure_id"], "basis": "inferred",
                 "analysis_metadata": {"methods": "Contrast plot; upstream model unknown"},
                 "evidence_execution_ids": [execution.id, "invented-id"],
                 "evidence_notes": "Filename match only; exact historical producer unconfirmed."}]

    callback = AsyncMock(side_effect=annotate)
    assert await annotate_legacy_figure_methods(state, tmp_path, callback) == 1
    state = AnalysisState.model_validate_json(state.model_dump_json())
    assert await annotate_legacy_figure_methods(state, tmp_path, callback) == 0
    callback.assert_awaited_once()
    candidate = resolve_state_figure_candidates(state, tmp_path, report_only=True)[0]
    assert candidate["analysis_metadata"]["annotation_basis"] == "inferred"
    assert candidate["analysis_metadata"]["annotation_evidence"] == execution.id
    assert not candidate["is_publication"] and not state.publication_figures
    write_valid_image(path, color=(200, 100, 0), min_bytes=2200)
    assert resolve_state_figure_candidates(state, tmp_path, report_only=True) == []


@pytest.mark.asyncio
async def test_annotation_failure_keeps_legacy_image_available_with_unknown_methods(tmp_path):
    path = tmp_path / "figures" / "old.jpg"
    path.parent.mkdir()
    write_valid_image(path, min_bytes=2200)
    state = AnalysisState(user_id="u")
    await annotate_legacy_figure_methods(state, tmp_path, AsyncMock(side_effect=RuntimeError("unavailable")))
    candidate = resolve_state_figure_candidates(state, tmp_path, report_only=True)[0]
    assert candidate["analysis_metadata"]["annotation_basis"] == "unknown"
    assert not candidate["is_publication"]
