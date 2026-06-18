"""Regression tests for GoogleDeepResearchClient._extract_research_result.

The fixture mirrors the real Interactions API response captured from
agent=deep-research-preview-04-2026, Api-Revision=2026-05-20: a `steps[]`
timeline where the final report lives in `model_output` steps as
`content[]` text blocks, planning lives in `thought` steps as a `summary[]`
array, and citations are `url_citation` annotations on the text blocks.

Guards against the regression where the parser read `output_text` /
`steps[].text` / `steps[].parts[]` (none of which exist in the live schema),
yielding an empty report.
"""
from backend.utils.llm_client import GoogleDeepResearchClient


def _client():
    # Bypass __init__ (no network / no httpx client needed for pure parsing).
    return GoogleDeepResearchClient.__new__(GoogleDeepResearchClient)


def _live_schema_response():
    return {
        "status": "completed",
        "object": "interaction",
        "agent": "deep-research-preview-04-2026",
        "steps": [
            {"type": "user_input", "content": [{"text": "my query", "type": "text"}]},
            {
                "type": "thought",
                "signature": "",
                "summary": [
                    {"text": "Generating research plan", "type": "text"},
                    {"text": "Defining scope", "type": "text"},
                ],
            },
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": "# Report\n\nFirst section. ",
                        "annotations": [
                            {"type": "url_citation", "url": "http://a", "start_index": 0, "end_index": 5},
                            {"type": "url_citation", "url": "http://a", "start_index": 6, "end_index": 9},
                            {"type": "url_citation", "url": "http://b"},
                        ],
                    }
                ],
            },
            {
                "type": "model_output",
                "content": [
                    {"type": "image", "mime_type": "image/png", "data": "iVBORw0=="},
                    {"type": "text", "text": "Second section."},
                ],
            },
        ],
    }


def test_extracts_report_from_model_output_content():
    result = _client()._extract_research_result(_live_schema_response())
    assert result["report"] == "# Report\n\nFirst section. Second section."


def test_excludes_user_input_text_from_report():
    result = _client()._extract_research_result(_live_schema_response())
    assert "my query" not in result["report"]


def test_collects_and_dedupes_url_citations():
    result = _client()._extract_research_result(_live_schema_response())
    assert [s["uri"] for s in result["sources"]] == ["http://a", "http://b"]


def test_captures_thought_summary_array():
    result = _client()._extract_research_result(_live_schema_response())
    assert [t["text"] for t in result["thinking"]] == [
        "Generating research plan",
        "Defining scope",
    ]


def test_ignores_inline_images():
    result = _client()._extract_research_result(_live_schema_response())
    assert "iVBORw0" not in result["report"]


def test_legacy_output_text_fallback():
    # Older revision: no steps, plain top-level output_text.
    data = {"status": "completed", "output_text": "legacy report body"}
    result = _client()._extract_research_result(data)
    assert result["report"] == "legacy report body"


def test_legacy_outputs_array_fallback():
    data = {
        "status": "completed",
        "outputs": [{"text": "from outputs[]"}],
    }
    result = _client()._extract_research_result(data)
    assert result["report"] == "from outputs[]"


def test_empty_response_yields_empty_report():
    result = _client()._extract_research_result({"status": "completed", "steps": []})
    assert result["report"] == ""
    assert result["sources"] == []
