"""Tests for PI Agent species inference methods."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock

from backend.models.schemas import AnalysisState
from backend.agents.pi_agent import PIAgent


def _make_pi_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client = MagicMock()
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


class TestInferSpecies:
    @pytest.mark.asyncio
    async def test_high_confidence_returns_species(self):
        agent = _make_pi_agent()
        state = AnalysisState(user_id="u1", name="Mouse kidney scRNA-seq",
                              initial_instruction="Analyze mouse kidney injury")
        response = json.dumps({
            "species": "Mus musculus", "confidence": "high",
            "suggestions": ["Mus musculus"], "reasoning": "Title says mouse"
        })
        with patch.object(agent, 'chat', new_callable=AsyncMock, return_value=response):
            result = await agent.infer_species(state)
        assert result["species"] == "Mus musculus"
        assert result["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_low_confidence_returns_suggestions(self):
        agent = _make_pi_agent()
        state = AnalysisState(user_id="u1", name="Kidney analysis",
                              initial_instruction="Analyze kidney scRNA-seq data")
        response = json.dumps({
            "species": "Mus musculus", "confidence": "low",
            "suggestions": ["Mus musculus", "Homo sapiens"],
            "reasoning": "Unclear from project info"
        })
        with patch.object(agent, 'chat', new_callable=AsyncMock, return_value=response):
            result = await agent.infer_species(state)
        assert result["confidence"] == "low"
        assert len(result["suggestions"]) == 2

    @pytest.mark.asyncio
    async def test_llm_error_returns_low_confidence(self):
        agent = _make_pi_agent()
        state = AnalysisState(user_id="u1", name="Test", initial_instruction="Test")
        with patch.object(agent, 'chat', new_callable=AsyncMock, side_effect=Exception("API error")):
            result = await agent.infer_species(state)
        assert result["confidence"] == "low"
        assert result["species"] == ""
        assert "error" in result.get("reasoning", "").lower() or "fail" in result.get("reasoning", "").lower()

    @pytest.mark.asyncio
    async def test_malformed_json_returns_low_confidence(self):
        agent = _make_pi_agent()
        state = AnalysisState(user_id="u1", name="Test", initial_instruction="Test")
        with patch.object(agent, 'chat', new_callable=AsyncMock, return_value="not json"):
            result = await agent.infer_species(state)
        assert result["confidence"] == "low"


class TestInferSpeciesFromHint:
    @pytest.mark.asyncio
    async def test_with_user_hint(self):
        agent = _make_pi_agent()
        state = AnalysisState(user_id="u1", name="Kidney analysis",
                              initial_instruction="Analyze kidney data")
        response = json.dumps({
            "species": "Rattus norvegicus", "confidence": "high",
            "suggestions": ["Rattus norvegicus"], "reasoning": "User specified rat"
        })
        with patch.object(agent, 'chat', new_callable=AsyncMock, return_value=response):
            result = await agent.infer_species_from_hint(state, "It's rat data")
        assert result["species"] == "Rattus norvegicus"
        assert result["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_llm_error_returns_low_confidence(self):
        agent = _make_pi_agent()
        state = AnalysisState(user_id="u1", name="Test", initial_instruction="Test")
        with patch.object(agent, 'chat', new_callable=AsyncMock, side_effect=Exception("API error")):
            result = await agent.infer_species_from_hint(state, "some hint")
        assert result["confidence"] == "low"
        assert result["species"] == ""

    @pytest.mark.asyncio
    async def test_malformed_json_returns_low_confidence(self):
        agent = _make_pi_agent()
        state = AnalysisState(user_id="u1", name="Test", initial_instruction="Test")
        with patch.object(agent, 'chat', new_callable=AsyncMock, return_value="not json"):
            result = await agent.infer_species_from_hint(state, "some hint")
        assert result["confidence"] == "low"
