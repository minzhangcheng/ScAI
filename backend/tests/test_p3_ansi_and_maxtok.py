import re
import backend.utils.error_parsing as ep_mod
import backend.utils.llm_client as llm_mod

def test_ansi_strips_osc():
    text = "before\x1b]0;title\x07after"
    out = ep_mod.strip_ansi(text)
    assert out == "beforeafter"

def test_ansi_strips_charset():
    text = "x\x1b(By"
    out = ep_mod.strip_ansi(text)
    assert out == "xy"

def test_ansi_still_strips_csi():
    text = "a\x1b[31mred\x1b[0mb"
    out = ep_mod.strip_ansi(text)
    assert out == "aredb"

def test_max_tokens_field_fallback_present():
    """P3-4: llm_client must have a fallback path for max_tokens/max_completion_tokens
    field naming, similar to the existing _temp_unsupported_models fallback."""
    src = open(llm_mod.__file__).read()
    assert "max_completion_tokens" in src, "must reference both field names"
    assert "max_tokens" in src, "must reference both field names"
    has_cache = "_max_tokens_unsupported_models" in src or "_max_tokens_field" in src
    has_swap = "max_completion_tokens" in src and re.search(
        r"BadRequestError|unsupported parameter|unrecognized.*parameter", src, re.IGNORECASE
    )
    assert has_cache or has_swap, \
        "P3-4: must have either a field cache or BadRequestError-driven swap for max_tokens fields"
