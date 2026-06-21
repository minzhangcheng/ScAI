from backend.utils.llm_client import _openai_sdk_base_url, _anthropic_sdk_base_url


def test_openai_strips_responses_suffix():
    assert _openai_sdk_base_url("https://x/v1/responses").rstrip("/") == "https://x/v1"


def test_openai_strips_chat_completions_suffix():
    assert _openai_sdk_base_url("https://x/v1/chat/completions").rstrip("/") == "https://x/v1"


def test_openai_leaves_plain_v1_untouched():
    assert _openai_sdk_base_url("https://x/v1").rstrip("/") == "https://x/v1"


def test_anthropic_strips_v1_messages():
    assert _anthropic_sdk_base_url("https://x/v1/messages").rstrip("/") == "https://x"


def test_anthropic_strips_bare_messages():
    assert _anthropic_sdk_base_url("https://x/messages").rstrip("/") == "https://x"


def test_anthropic_leaves_root_untouched():
    assert _anthropic_sdk_base_url("https://api.anthropic.com").rstrip("/") == "https://api.anthropic.com"


from backend.utils.llm_client import OpenAICompatibleClient, _route_protocol


def test_client_sdk_base_url_normalized_for_explicit_responses():
    c = OpenAICompatibleClient(api_key="k", base_url="https://x/v1/responses", agent_name="t")
    # SDK base_url must NOT contain the doubled /responses
    assert "responses" not in str(c.client.base_url)
    assert str(c.client.base_url).rstrip("/").endswith("/v1")
    # but routing still sees the explicit endpoint via the ORIGINAL base_url
    assert c.base_url == "https://x/v1/responses"
    assert _route_protocol(c.base_url, "gpt-4o") == "responses"
