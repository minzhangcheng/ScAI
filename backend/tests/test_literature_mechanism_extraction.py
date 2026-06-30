import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agents.literature_agent import LiteratureAgent
from backend.models.schemas import LiteratureMechanismSignal


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_llm_client.return_value = MagicMock()
    provider.get_temperature.return_value = 0.3
    provider.get_system_prompt.return_value = "test prompt"
    return provider


_LLM = json.dumps({
    "mechanism_signals": [{
        "mechanism": "ferroptosis",
        "supporting_pmids": ["38963329"],
        "evidence_type": "correlational",
        "relevance_to_current_study": "links Cu deficiency to iron-driven cell death",
        "observed_data_links": ["iron accumulation", "Hmox1 upregulation"],
        "limitations": ["no direct lipid peroxidation assay in cited work"],
    }]
})


@pytest.mark.asyncio
async def test_extracts_signals_generically(mock_provider):
    agent = LiteratureAgent(settings_provider=mock_provider)
    with patch.object(agent, "chat_without_history", new_callable=AsyncMock, return_value=_LLM):
        signals = await agent.extract_mechanism_signals(
            report_text="copper deficiency ... ferroptosis ... PMID 38963329",
            references=[{"title": "Cu and ferroptosis", "url": "u", "pmid": "38963329"}],
            observed_context="Iron 2.2x; Hmox1 up",
        )
    assert signals is not None
    assert len(signals) == 1
    assert isinstance(signals[0], LiteratureMechanismSignal)
    assert signals[0].mechanism == "ferroptosis"


@pytest.mark.asyncio
async def test_prompt_has_no_mechanism_hardcoding(mock_provider):
    agent = LiteratureAgent(settings_provider=mock_provider)
    with patch.object(agent, "chat_without_history", new_callable=AsyncMock, return_value=_LLM) as mock_chat:
        await agent.extract_mechanism_signals(
            report_text="body", references=[{"title": "t"}], observed_context="ctx",
        )
    prompt = mock_chat.call_args[0][0]
    low = prompt.lower()
    for banned in ("ferroptosis", "gpx4", "acsl4", "lipid peroxidation"):
        assert banned not in low, f"prompt must not hardcode {banned}"


@pytest.mark.asyncio
async def test_returns_none_on_llm_failure(mock_provider):
    agent = LiteratureAgent(settings_provider=mock_provider)
    with patch.object(agent, "chat_without_history", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
        signals = await agent.extract_mechanism_signals(
            report_text="body", references=[], observed_context="ctx",
        )
    assert signals is None


@pytest.mark.asyncio
async def test_returns_none_when_no_content(mock_provider):
    agent = LiteratureAgent(settings_provider=mock_provider)
    with patch.object(agent, "chat_without_history", new_callable=AsyncMock) as mock_chat:
        signals = await agent.extract_mechanism_signals(
            report_text="", references=[], observed_context="ctx",
        )
    assert signals is None
    mock_chat.assert_not_awaited()
