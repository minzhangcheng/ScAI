"""_evaluate_pubmed_results 应按年份降序采样更多文章并标注总数。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
from unittest.mock import MagicMock
from backend.agents.literature_agent import LiteratureAgent


def _agent():
    provider = MagicMock()
    provider.get_model.return_value = "m"
    provider.get_temperature.return_value = 0.3
    provider.get_system_prompt.return_value = "p"
    provider.get_llm_client.return_value = MagicMock()
    return LiteratureAgent(settings_provider=provider)


def test_eval_prompt_sorts_by_year_and_samples_30():
    agent = _agent()
    captured = {}

    async def fake_chat(prompt, temperature=0.3):
        captured["prompt"] = prompt
        return '{"sufficient": true}', False

    agent._chat_with_fallback = fake_chat  # type: ignore

    articles = [{"pmid": str(i), "title": f"T{i}", "abstract": "a", "year": str(2000 + i)}
                for i in range(40)]
    history = [{"round": 1, "query": "q"}]
    asyncio.run(agent._evaluate_pubmed_results("topic", articles, history))

    prompt = captured["prompt"]
    assert "T39" in prompt
    assert "T38" in prompt
    assert "40" in prompt
    assert "T0\n" not in prompt and "T0 " not in prompt
