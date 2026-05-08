import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_build_chat_sections_async_filters_delegation_active():
    from backend.services.report.chat_sections import build_chat_sections_async
    
    messages = [
        {"role": "user", "content": "What genes are enriched?", "timestamp": "2024-01-01T00:00:00"},
        {"role": "assistant", "content": "Working on it...", "timestamp": "2024-01-01T00:00:01", "delegation_active": True},
        {"role": "user", "content": "What genes are enriched?", "timestamp": "2024-01-01T00:00:00"},
        {"role": "assistant", "content": "Here are the enriched genes: ...", "timestamp": "2024-01-01T00:01:00"},
    ]
    sections = await build_chat_sections_async(messages, figures_dir="/tmp", tables_dir="/tmp", llm_client=None)
    assert len(sections) >= 1
    for s in sections:
        assert "delegation" not in s.get("title", "").lower()[:30] or "delegation" not in s.get("content", "").lower()[:30]

@pytest.mark.asyncio
async def test_build_chat_sections_async_respects_max_entries():
    from backend.services.report.chat_sections import build_chat_sections_async
    
    messages = []
    for i in range(30):
        messages.append({"role": "user", "content": f"Question {i}", "timestamp": f"2024-01-01T{i:02d}:00:00"})
        messages.append({"role": "assistant", "content": f"Answer {i}", "timestamp": f"2024-01-01T{i:02d}:01:00"})
    
    sections = await build_chat_sections_async(messages, figures_dir="/tmp", tables_dir="/tmp", llm_client=None, max_chat_entries=5)
    assert len(sections) <= 5

@pytest.mark.asyncio
async def test_build_chat_sections_async_empty_messages():
    from backend.services.report.chat_sections import build_chat_sections_async
    
    sections = await build_chat_sections_async([], figures_dir="/tmp", tables_dir="/tmp", llm_client=None)
    assert sections == []

@pytest.mark.asyncio
async def test_build_chat_sections_async_none_messages():
    from backend.services.report.chat_sections import build_chat_sections_async
    
    sections = await build_chat_sections_async(None, figures_dir="/tmp", tables_dir="/tmp", llm_client=None)
    assert sections == []

@pytest.mark.asyncio
async def test_build_chat_sections_async_uses_cached_title():
    from backend.services.report.chat_sections import build_chat_sections_async
    
    messages = [
        {"role": "user", "content": "What genes are enriched?", "timestamp": "2024-01-01T00:00:00", "_summary_title": "Enriched Gene Analysis"},
        {"role": "assistant", "content": "Here are the enriched genes: ...", "timestamp": "2024-01-01T00:01:00"},
    ]
    sections = await build_chat_sections_async(messages, figures_dir="/tmp", tables_dir="/tmp", llm_client=None)
    assert len(sections) == 1
    assert sections[0]["title"] == "Enriched Gene Analysis"


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_truncated_title():
    from backend.services.report.chat_sections import build_chat_sections_async

    messages = [
        {"role": "user", "content": "What genes are enriched?", "timestamp": "2024-01-01T00:00:00"},
        {"role": "assistant", "content": "Results here", "timestamp": "2024-01-01T00:01:00"},
    ]

    failing_client = AsyncMock()
    failing_client.chat = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    sections = await build_chat_sections_async(messages, figures_dir="/tmp", tables_dir="/tmp", llm_client=failing_client)
    assert len(sections) == 1
    assert sections[0]["title"] == "What genes are enriched?"


@pytest.mark.asyncio
async def test_llm_malformed_json_falls_back_to_truncated_title():
    from backend.services.report.chat_sections import build_chat_sections_async

    messages = [
        {"role": "user", "content": "Explain the pathway", "timestamp": "2024-01-01T00:00:00"},
        {"role": "assistant", "content": "The pathway involves...", "timestamp": "2024-01-01T00:01:00"},
    ]

    bad_client = AsyncMock()
    bad_client.chat = AsyncMock(return_value="This is not JSON at all")

    sections = await build_chat_sections_async(messages, figures_dir="/tmp", tables_dir="/tmp", llm_client=bad_client)
    assert len(sections) == 1
    assert sections[0]["title"] == "Explain the pathway"


@pytest.mark.asyncio
async def test_skips_figures_with_missing_files(tmp_path):
    from backend.services.report.chat_sections import build_chat_sections_async

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    (figures_dir / "exists.png").write_bytes(b"\x89PNG")

    messages = [
        {"role": "user", "content": "Show me", "timestamp": "2024-01-01T00:00:00"},
        {
            "role": "assistant",
            "content": "Here is the figure",
            "timestamp": "2024-01-01T00:01:00",
            "figures": [
                {"path": "exists.png"},
                {"path": "missing.png"},
                {"name": "also_missing.png"},
            ],
        },
    ]

    sections = await build_chat_sections_async(messages, figures_dir=str(figures_dir), tables_dir="/tmp", llm_client=None)
    assert len(sections) == 1
    assert len(sections[0]["figures"]) == 1
    assert sections[0]["figures"][0]["path"] == "exists.png"


@pytest.mark.asyncio
async def test_truncates_long_title_to_500_chars():
    from backend.services.report.chat_sections import build_chat_sections_async

    long_content = "x" * 600
    messages = [
        {"role": "user", "content": long_content, "timestamp": "2024-01-01T00:00:00"},
        {"role": "assistant", "content": "Answer", "timestamp": "2024-01-01T00:01:00"},
    ]

    sections = await build_chat_sections_async(messages, figures_dir="/tmp", tables_dir="/tmp", llm_client=None)
    assert len(sections) == 1
    assert len(sections[0]["title"]) <= 503
    assert sections[0]["title"].endswith("...")