from backend.tests.split_facade import split_source
import pathlib


def test_no_hardcoded_gemini_in_chat():
    src = split_source("backend.routes.chat")
    assert "gemini-3.1-pro-preview" not in src
