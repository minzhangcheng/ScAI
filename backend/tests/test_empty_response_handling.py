"""completion_tokens==0 / empty content is a transient upstream truncation,
not a 'model only talks' behavior — it must not count toward the text-only
HITL escalation."""
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.utils.llm_client import LLMToolResponse, EMPTY_CONTENT_PLACEHOLDER


def _resp(content, completion_tokens, tool_calls=None):
    return LLMToolResponse(
        content=content,
        tool_calls=tool_calls or [],
        model="m",
        usage={"prompt_tokens": 10, "completion_tokens": completion_tokens, "total_tokens": 10},
        raw_response=None,
    )


def test_zero_token_empty_is_empty_response():
    """Empty content + zero completion tokens = transient truncation."""
    assert BioinformaticsAgent._is_empty_response(_resp("", 0)) is True


def test_error_placeholder_is_empty_response():
    """The extraction-error placeholder counts as empty."""
    assert BioinformaticsAgent._is_empty_response(_resp(EMPTY_CONTENT_PLACEHOLDER, 5)) is True


def test_whitespace_only_is_empty_response():
    """Whitespace-only content counts as empty (transient truncation)."""
    assert BioinformaticsAgent._is_empty_response(_resp("   ", 0)) is True


def test_real_text_response_is_not_empty():
    """Genuine text with real content is NOT empty, even with zero tokens
    in usage (the model chose to talk)."""
    assert BioinformaticsAgent._is_empty_response(_resp("Here is my analysis plan...", 0)) is False


def test_real_text_response_with_usage_is_not_empty():
    """Genuine text with real content and real token count is not empty."""
    assert BioinformaticsAgent._is_empty_response(_resp("Here is my analysis plan...", 50)) is False


def test_tool_call_response_is_not_empty():
    tc = object()  # any non-empty tool_calls list means it acted
    assert BioinformaticsAgent._is_empty_response(_resp("", 50, tool_calls=[tc])) is False


def test_missing_usage_with_real_text_is_not_empty():
    """If usage dict is empty/missing but content is real text, it's not empty.
    This mirrors the existing integration test scenario where usage={}."""
    resp = LLMToolResponse(
        content="I will now run marker gene analysis...",
        tool_calls=[],
        model="m",
        usage={},
        raw_response=None,
    )
    assert BioinformaticsAgent._is_empty_response(resp) is False