import pytest
from backend.utils.llm_client import _is_deep_research_model, _inject_web_search_if_needed


def test_is_dr_matches_hyphenated():
    assert _is_deep_research_model("o3-deep-research") is True


def test_is_dr_matches_non_hyphenated():
    # 用户点名的例子：deepresearch 无连字符
    assert _is_deep_research_model("o4-deepresearch-mini") is True


def test_is_dr_with_provider_prefix():
    assert _is_deep_research_model("openai/o3-deep-research") is True


def test_is_dr_false_for_normal_model():
    assert _is_deep_research_model("gpt-4o") is False


def test_inject_web_search_for_dr():
    tools = _inject_web_search_if_needed("o3-deep-research", [])
    assert {"type": "web_search_preview"} in tools


def test_inject_skips_non_dr():
    assert _inject_web_search_if_needed("gpt-4o", []) == []


def test_inject_no_duplicate_when_present():
    existing = [{"type": "web_search"}]
    out = _inject_web_search_if_needed("o3-deep-research", existing)
    assert out == existing  # 已含检索工具变体则不重复加


def test_inject_preserves_user_function_tools():
    user_tools = [{"type": "function", "function": {"name": "f"}}]
    out = _inject_web_search_if_needed("o3-deep-research", user_tools)
    assert user_tools[0] in out and {"type": "web_search_preview"} in out


from backend.utils.llm_client import is_endpoint_absent, is_responses_only_model


class _Err(Exception):
    pass


def test_endpoint_absent_routelevel_keywords():
    assert is_endpoint_absent(_Err("Cannot POST /v1/responses")) is True
    assert is_endpoint_absent(_Err("Unknown request URL")) is True
    assert is_endpoint_absent(_Err("404 page not found")) is True
    assert is_endpoint_absent(_Err("405 Method Not Allowed")) is True


def test_endpoint_absent_model_only_in_chat():
    assert is_endpoint_absent(_Err("This model is only supported in v1/chat/completions")) is True


def test_endpoint_absent_does_not_match_model_not_found():
    # 模型级 "model xxx not found" 不能被误判为端点缺失
    assert is_endpoint_absent(_Err("The model `foo` was not found")) is False


def test_responses_only_model_detect():
    assert is_responses_only_model(_Err("only supported in v1/responses, not v1/chat/completions")) is True
    assert is_responses_only_model(_Err("some other error")) is False


import backend.utils.llm_client as llm_mod
from backend.utils.llm_client import (
    _route_protocol,
    _is_unknown_web_search_tool_error,
    _swap_web_search_tool_name,
    _WEB_SEARCH_TOOL_NAMES,
)


def _clear_caches():
    llm_mod._NO_RESPONSES_ENDPOINT.clear()
    llm_mod._PROBE_CACHE.clear()


def test_route_explicit_suffix_responses():
    _clear_caches()
    assert _route_protocol("https://x/v1/responses", "gpt-4o") == "responses"


def test_route_explicit_suffix_messages():
    _clear_caches()
    assert _route_protocol("https://x/v1/messages", "claude-3") == "anthropic"
    assert _route_protocol("https://x/messages", "claude-3") == "anthropic"


def test_route_explicit_suffix_chat():
    _clear_caches()
    assert _route_protocol("https://x/v1/chat/completions", "gpt-4o") == "chat"


def test_route_no_v1_is_anthropic():
    _clear_caches()
    assert _route_protocol("https://api.anthropic.com", "claude-3") == "anthropic"


def test_route_dr_model_is_responses():
    _clear_caches()
    assert _route_protocol("https://x/v1", "o4-deepresearch-mini") == "responses"


def test_route_no_responses_endpoint_cached_chat():
    _clear_caches()
    llm_mod._NO_RESPONSES_ENDPOINT.add("https://x/v1")
    assert _route_protocol("https://x/v1", "gpt-4o") == "chat"


def test_route_probe_cache_hit():
    _clear_caches()
    llm_mod._PROBE_CACHE[("https://x/v1", "gpt-4o")] = "responses"
    assert _route_protocol("https://x/v1", "gpt-4o") == "responses"


def test_route_default_is_responses():
    _clear_caches()
    assert _route_protocol("https://x/v1", "gpt-4o") == "responses"


def test_route_dr_beats_no_responses_endpoint():
    # DR 模型必须走 responses，即便整站曾标记无 responses（边角，但顺序要确定）
    _clear_caches()
    llm_mod._NO_RESPONSES_ENDPOINT.add("https://x/v1")
    assert _route_protocol("https://x/v1", "o3-deep-research") == "responses"


def test_unknown_web_search_tool_error_detects():
    assert _is_unknown_web_search_tool_error(
        _Err("Unknown tool type: web_search_preview")
    ) is True
    assert _is_unknown_web_search_tool_error(
        _Err("tool not found: web_search")
    ) is True
    assert _is_unknown_web_search_tool_error(
        _Err("unsupported tool type 'web_search_preview'")
    ) is True
    assert _is_unknown_web_search_tool_error(
        _Err("not a valid tool: web_search")
    ) is True


def test_unknown_web_search_tool_error_rejects_normal():
    assert _is_unknown_web_search_tool_error(
        _Err("model not found")
    ) is False
    assert _is_unknown_web_search_tool_error(
        _Err("rate limit exceeded")
    ) is False
    assert _is_unknown_web_search_tool_error(
        _Err("unknown parameter: temperature")
    ) is False


def test_swap_web_search_tool_name():
    tools = [{"type": "web_search_preview"}, {"type": "function", "name": "f"}]
    result = _swap_web_search_tool_name(tools, "web_search_preview", "web_search")
    assert result == [{"type": "web_search"}, {"type": "function", "name": "f"}]
    assert tools[0]["type"] == "web_search_preview"


def test_swap_web_search_tool_name_no_match():
    tools = [{"type": "web_search"}, {"type": "function", "name": "f"}]
    result = _swap_web_search_tool_name(tools, "web_search_preview", "web_search")
    assert result == tools


from backend.utils.llm_client import is_responses_endpoint_absent, is_responses_chat_only_model


def test_responses_endpoint_absent_excludes_chat_only_model_error():
    err = _Err("This model is only supported in v1/chat/completions")
    assert is_responses_endpoint_absent(err) is False
    assert is_responses_chat_only_model(err) is True


def test_responses_endpoint_absent_keeps_route_level_markers():
    assert is_responses_endpoint_absent(_Err("Cannot POST /v1/responses")) is True
    assert is_responses_endpoint_absent(_Err("Unknown request URL")) is True
    assert is_responses_endpoint_absent(_Err("404 page not found")) is True
    assert is_responses_endpoint_absent(_Err("405 Method Not Allowed")) is True


@pytest.mark.asyncio
async def test_chat_only_model_fallback_does_not_mark_base_url_no_responses():
    from unittest.mock import AsyncMock, MagicMock
    from backend.utils.llm_client import OpenAICompatibleClient, LLMResponse
    import backend.utils.llm_client as llm_mod

    llm_mod._NO_RESPONSES_ENDPOINT.clear()
    llm_mod._PROBE_CACHE.clear()
    c = OpenAICompatibleClient(api_key="k", base_url="https://x/v1", agent_name="t")
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=_Err("This model is only supported in v1/chat/completions"))
    c._responses_adapter = fake

    inline_resp = MagicMock()
    inline_resp.choices = [MagicMock(message=MagicMock(content="CHAT", tool_calls=None))]
    inline_resp.model = "gpt-chat-only"
    inline_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                                  prompt_tokens_details=None)
    c.client.chat.completions.create = AsyncMock(return_value=inline_resp)

    r = await c.chat(model="gpt-chat-only", messages=[{"role": "user", "content": "hi"}])

    assert r.content == "CHAT"
    assert "https://x/v1" not in llm_mod._NO_RESPONSES_ENDPOINT
    assert llm_mod._PROBE_CACHE[("https://x/v1", "gpt-chat-only")] == "chat"


from backend.utils.llm_client import _route_protocol_order


def test_order_default_v1():
    llm_mod._NO_RESPONSES_ENDPOINT.clear(); llm_mod._PROBE_CACHE.clear()
    assert _route_protocol_order("https://x/v1", "gpt-4o") == ["responses", "chat", "anthropic"]


def test_order_no_v1_anthropic_first():
    llm_mod._NO_RESPONSES_ENDPOINT.clear(); llm_mod._PROBE_CACHE.clear()
    assert _route_protocol_order("https://api.anthropic.com", "claude-3") == [
        "anthropic", "responses", "chat"]


def test_order_anthropic_keyword_with_v1():
    llm_mod._NO_RESPONSES_ENDPOINT.clear(); llm_mod._PROBE_CACHE.clear()
    # /v1 present but 'anthropic' keyword floats anthropic to the front
    assert _route_protocol_order("https://api.anthropic.com/v1", "claude-3")[0] == "anthropic"


def test_route_anthropic_v1_resolves_anthropic():
    llm_mod._NO_RESPONSES_ENDPOINT.clear(); llm_mod._PROBE_CACHE.clear()
    assert _route_protocol("https://api.anthropic.com/v1", "claude-3") == "anthropic"


def test_order_messages_suffix_is_hard_pin():
    llm_mod._NO_RESPONSES_ENDPOINT.clear(); llm_mod._PROBE_CACHE.clear()
    assert _route_protocol_order("https://gw/v1/messages", "m") == ["anthropic"]


def test_order_explicit_responses_suffix_single():
    llm_mod._NO_RESPONSES_ENDPOINT.clear(); llm_mod._PROBE_CACHE.clear()
    assert _route_protocol_order("https://x/v1/responses", "m") == ["responses"]


def test_order_cached_decision_single():
    llm_mod._NO_RESPONSES_ENDPOINT.clear(); llm_mod._PROBE_CACHE.clear()
    llm_mod._PROBE_CACHE[("https://x/v1", "m")] = "chat"
    assert _route_protocol_order("https://x/v1", "m") == ["chat"]


def test_order_dr_model_responses_then_chat():
    llm_mod._NO_RESPONSES_ENDPOINT.clear(); llm_mod._PROBE_CACHE.clear()
    assert _route_protocol_order("https://x/v1", "o3-deep-research") == ["responses", "chat"]


def test_order_no_responses_endpoint_drops_responses():
    llm_mod._NO_RESPONSES_ENDPOINT.clear(); llm_mod._PROBE_CACHE.clear()
    llm_mod._NO_RESPONSES_ENDPOINT.add("https://x/v1")
    assert _route_protocol_order("https://x/v1", "m") == ["chat", "anthropic"]
