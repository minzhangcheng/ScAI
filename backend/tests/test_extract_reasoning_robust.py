"""Provider reasoning shapes are tolerated but never exposed as answers."""
import pytest
from types import SimpleNamespace
from backend.utils.llm_client import OpenAICompatibleClient


def _client_proxy():
    client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    client.agent_name = "test"
    return client


def _msg(content=None, **kwargs):
    m = SimpleNamespace(content=content)
    for k, v in kwargs.items():
        setattr(m, k, v)
    return m


def test_reasoning_list_of_dicts_with_text_is_private():
    c = _client_proxy()
    msg = _msg(content=None, reasoning=[{"text": "step 1"}, {"text": "step 2"}])
    out = c._extract_content_from_message(msg)
    assert out == ""


def test_reasoning_list_of_strings_is_private_without_attribute_error():
    c = _client_proxy()
    msg = _msg(content=None, reasoning=["thought a", "thought b"])
    out = c._extract_content_from_message(msg)
    assert out == ""


def test_reasoning_string_is_private():
    c = _client_proxy()
    msg = _msg(content=None, reasoning="single reasoning")
    out = c._extract_content_from_message(msg)
    assert out == ""


def test_reasoning_none_returns_empty():
    c = _client_proxy()
    msg = _msg(content=None)
    out = c._extract_content_from_message(msg)
    assert out == ""


def test_reasoning_list_of_dicts_with_content_key_is_private():
    c = _client_proxy()
    msg = _msg(content=None, reasoning_content=[{"content": "alt key"}])
    out = c._extract_content_from_message(msg)
    assert out == ""


def test_reasoning_dict_value_is_private():
    c = _client_proxy()
    msg = _msg(content=None, thinking={"text": "dict thought"})
    out = c._extract_content_from_message(msg)
    assert out == ""
