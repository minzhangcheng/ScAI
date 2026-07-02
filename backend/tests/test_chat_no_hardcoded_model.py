import pathlib


def test_no_hardcoded_gemini_in_chat():
    src = pathlib.Path("backend/routes/chat.py").read_text()
    assert "gemini-3.1-pro-preview" not in src
