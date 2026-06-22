import asyncio
import json
import httpx
import pytest
from unittest.mock import MagicMock
from backend.utils.llm_client import (
    is_transient_probe_error,
    APITimeoutError, APIConnectionError, InternalServerError, RateLimitError,
    NotFoundError, BadRequestError,
)


def _err(cls, msg, status):
    return cls(message=msg, response=MagicMock(status_code=status), body=None)


def test_5xx_is_transient():
    assert is_transient_probe_error(_err(InternalServerError, "boom", 500)) is True


def test_429_is_transient():
    assert is_transient_probe_error(_err(RateLimitError, "slow down", 429)) is True


def test_timeout_is_transient():
    assert is_transient_probe_error(APITimeoutError(request=MagicMock())) is True
    assert is_transient_probe_error(asyncio.TimeoutError()) is True
    assert is_transient_probe_error(httpx.ReadTimeout("t")) is True


def test_connection_is_transient():
    assert is_transient_probe_error(APIConnectionError(request=MagicMock())) is True


def test_endpoint_absent_500_is_not_transient():
    # 'not implemented' + responses route marker -> deterministic, switch immediately
    e = _err(InternalServerError,
             "Error code: 500 - not implemented local:convert_request_failed", 500)
    assert is_transient_probe_error(e) is False


def test_404_endpoint_absent_is_not_transient():
    assert is_transient_probe_error(_err(NotFoundError, "Cannot POST /v1/responses", 404)) is False


def test_responses_only_model_is_not_transient():
    e = _err(BadRequestError, "This model is only supported in v1/responses", 400)
    assert is_transient_probe_error(e) is False


def test_chat_only_model_is_not_transient():
    e = _err(BadRequestError, "This model is only supported in v1/chat/completions", 400)
    assert is_transient_probe_error(e) is False


def test_json_decode_error_is_not_transient():
    assert is_transient_probe_error(json.JSONDecodeError("bad", "{", 0)) is False


def test_plain_bad_request_is_not_transient():
    assert is_transient_probe_error(_err(BadRequestError, "invalid param foo", 400)) is False
