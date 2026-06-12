"""Tests for bioinformatics agent species validation and organism override."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from backend.constants import WaitInputReason
from backend.models.schemas import AnalysisState, TaskStatus
from backend.workflow.callbacks import CallbacksMixin


class TestOrganismOverride:
    def test_session_species_overrides_llm_organism(self):
        state = AnalysisState(user_id="u1", species="Homo sapiens")
        llm_organism = "mouse"
        organism = state.species if state.species else llm_organism
        assert organism == "Homo sapiens"

    def test_no_override_when_species_empty(self):
        state = AnalysisState(user_id="u1", species="")
        llm_organism = "mouse"
        organism = state.species if state.species else llm_organism
        assert organism == "mouse"


class TestSpeciesValidatedFlag:
    def test_flag_default_false(self):
        state = AnalysisState(user_id="u1")
        assert not state.species_genes_validated

    def test_flag_survives_roundtrip(self):
        state = AnalysisState(user_id="u1", species="Homo sapiens", species_genes_validated=True)
        data = state.model_dump(mode="json")
        restored = AnalysisState.model_validate(data)
        assert restored.species_genes_validated is True


def _build_callbacks_mixin(state):
    inst = CallbacksMixin.__new__(CallbacksMixin)
    inst.session_id = "s1"
    inst._current_analysis_state = state
    inst._species_event = asyncio.Event()
    inst._species_user_response = None
    inst._send_state_sync = AsyncMock()
    inst._send_message = AsyncMock()
    inst._set_status = MagicMock(side_effect=lambda s, st: setattr(s, "status", st))
    inst.pi_agent = SimpleNamespace(decide_species_action=AsyncMock(return_value={"action": "keep"}))
    return inst


@pytest.mark.asyncio
async def test_gene_validation_uses_gene_mismatch_reason(monkeypatch):
    state = AnalysisState(user_id="u", species="Homo sapiens")
    state.status = TaskStatus.RUNNING
    inst = _build_callbacks_mixin(state)

    captured = {"reason": None}
    def _capture_set_status(s, st):
        s.status = st
        if st == TaskStatus.WAITING_SPECIES_INPUT:
            captured["reason"] = s.wait_input_reason
    inst._set_status.side_effect = _capture_set_status

    async def _signal_immediately(*a, **k):
        inst._species_event.set()
        return True
    monkeypatch.setattr("backend.workflow.callbacks.wait_with_skip_time", _signal_immediately)

    await inst._species_validation_callback(["TP53", "BRCA1"], "Homo sapiens",
                                            {"reasoning": "looks human", "likely_species": "Homo sapiens"})

    assert captured["reason"] == WaitInputReason.GENE_SPECIES_MISMATCH.value


@pytest.mark.asyncio
async def test_gene_validation_cleanup_on_cancel(monkeypatch):
    state = AnalysisState(user_id="u", species="Homo sapiens")
    state.status = TaskStatus.RUNNING
    inst = _build_callbacks_mixin(state)

    async def _raise(*a, **k):
        raise asyncio.CancelledError()
    monkeypatch.setattr("backend.workflow.callbacks.wait_with_skip_time", _raise)

    with pytest.raises(asyncio.CancelledError):
        await inst._species_validation_callback(["TP53"], "Homo sapiens",
                                                {"reasoning": "n/a", "likely_species": ""})
    assert state.species_input_info is None
    assert state.wait_input_reason is None


import json
from unittest.mock import AsyncMock, MagicMock
from backend.agents.bioinformatics_agent import BioinformaticsAgent


def _fresh_agent():
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent.sandbox = MagicMock()
    agent._notify = AsyncMock()
    agent.pi_callback = None
    agent.species_validation_callback = None
    agent.analysis_context = {}
    return agent


@pytest.mark.asyncio
async def test_auto_validate_handles_warning_prefix():
    """sandbox output containing a warning line before JSON should not break parsing."""
    agent = _fresh_agent()
    agent._current_analysis_state = AnalysisState(user_id="u", species="Homo sapiens")
    # First line is a warning; the JSON list is on the second line.
    fake_result = SimpleNamespace(
        output='UserWarning: harmless message\n["TP53", "BRCA1"]\n',
        error=None, success=True,
    )
    agent.sandbox.execute = AsyncMock(return_value=fake_result)
    agent._assess_gene_species_match = AsyncMock(return_value={"match": True})

    await agent._auto_validate_species_genes()
    agent._assess_gene_species_match.assert_awaited_once()


@pytest.mark.asyncio
async def test_assess_logs_warning_when_pi_callback_missing(caplog):
    agent = _fresh_agent()
    agent.pi_callback = None
    with caplog.at_level("WARNING"):
        result = await agent._assess_gene_species_match(["TP53"], "Homo sapiens")
    assert result == {"match": True}
    assert any("pi_callback" in rec.message for rec in caplog.records)
