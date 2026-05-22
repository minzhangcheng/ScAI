"""Tests for robust reasoning extraction in _extract_content_from_message (P2-3)."""
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


def test_reasoning_list_of_dicts_with_text():
    c = _client_proxy()
    msg = _msg(content=None, reasoning=[{"text": "step 1"}, {"text": "step 2"}])
    out = c._extract_content_from_message(msg)
    assert "step 1" in out and "step 2" in out


def test_reasoning_list_of_strings_no_attribute_error():
    """Provider returns list of plain strings — must not raise."""
    c = _client_proxy()
    msg = _msg(content=None, reasoning=["thought a", "thought b"])
    out = c._extract_content_from_message(msg)
    assert "thought a" in out and "thought b" in out


def test_reasoning_string():
    c = _client_proxy()
    msg = _msg(content=None, reasoning="single reasoning")
    out = c._extract_content_from_message(msg)
    assert "single reasoning" in out


def test_reasoning_none_returns_empty():
    c = _client_proxy()
    msg = _msg(content=None)
    out = c._extract_content_from_message(msg)
    assert out is not None


def test_reasoning_list_of_dicts_with_content_key():
    c = _client_proxy()
    msg = _msg(content=None, reasoning_content=[{"content": "alt key"}])
    out = c._extract_content_from_message(msg)
    assert "alt key" in out


def test_reasoning_dict_value():
    c = _client_proxy()
    msg = _msg(content=None, thinking={"text": "dict thought"})
    out = c._extract_content_from_message(msg)
    assert "dict thought" in out
