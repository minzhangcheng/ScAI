import re
import backend.utils.llm_client as llm_mod


def test_temp_unsupported_cache_protected_by_lock():
    """P2-1: _temp_unsupported_models additions must be protected by an asyncio.Lock."""
    src = open(llm_mod.__file__).read()
    assert "asyncio.Lock" in src and "_temp_cache_lock" in src, \
        "P2-1: must declare an asyncio.Lock for the cache"
    for m in re.finditer(r"_temp_unsupported_models\.add\(", src):
        start = max(0, m.start() - 400)
        preceding = src[start:m.start()]
        assert "async with" in preceding and "_temp_cache_lock" in preceding, \
            f"P2-1: .add() at offset {m.start()} not inside `async with _temp_cache_lock`"


def test_toolchoice_unsupported_cache_protected():
    src = open(llm_mod.__file__).read()
    for m in re.finditer(r"_toolchoice_unsupported_models\.add\(", src):
        start = max(0, m.start() - 400)
        preceding = src[start:m.start()]
        assert "async with" in preceding and ("_toolchoice_cache_lock" in preceding or "_temp_cache_lock" in preceding), \
            f"P2-1: toolchoice .add() at offset {m.start()} not inside `async with` of a cache lock"
