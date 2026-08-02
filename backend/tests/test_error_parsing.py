"""Tests for error_parsing utilities."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.utils.error_parsing import (
    TECHNICAL_ERROR_TYPES,
    API_ERROR_TYPES,
    CODE_GEN_ERROR_TYPES,
    compact_traceback,
    sanitize_error_payload,
    strip_ansi,
    extract_error_type,
    detect_swallowed_error,
)


def test_technical_error_types_is_public():
    """TECHNICAL_ERROR_TYPES should be importable without underscore prefix."""
    import backend.utils.error_parsing as ep
    assert hasattr(ep, "TECHNICAL_ERROR_TYPES")


def test_technical_error_types_not_empty():
    assert len(TECHNICAL_ERROR_TYPES) > 0


def test_technical_error_types_contains_common_errors():
    for err in ("KeyError", "TypeError", "ValueError", "RuntimeError"):
        assert err in TECHNICAL_ERROR_TYPES, f"{err} missing from TECHNICAL_ERROR_TYPES"


def test_naming_consistency_no_leading_underscore():
    """All three public error type constants should follow the same naming convention."""
    from backend.utils import error_parsing
    for name in ("TECHNICAL_ERROR_TYPES", "API_ERROR_TYPES", "CODE_GEN_ERROR_TYPES"):
        assert hasattr(error_parsing, name), f"{name} not found in error_parsing module"


def test_strip_ansi():
    assert strip_ansi("\x1b[31mError\x1b[0m") == "Error"
    assert strip_ansi("plain text") == "plain text"


def test_strip_ansi_removes_osc_with_bel_or_string_terminator():
    assert strip_ansi("\x1b]8;;https://invalid.example\x07label\x1b]8;;\x07") == "label"
    assert strip_ansi(
        "\x1b]8;;https://invalid.example\x1b\\label\x1b]8;;\x1b\\"
    ) == "label"


def test_error_payload_secret_keys_mask_arbitrary_value_formats():
    secret = "opaque value with spaces and no credential prefix"
    payload = {
        "api_key": secret,
        "nested": {
            "googleApiKey": secret,
            "access-token": secret,
            "db_password": 123456,
            "client.secret": [secret],
            "diagnostic": "ValueError: useful explanation",
        },
    }

    sanitized = sanitize_error_payload(payload)

    assert secret not in repr(sanitized)
    assert sanitized["api_key"] == "<redacted-secret>"
    assert sanitized["nested"]["db_password"] == "<redacted-secret>"
    assert sanitized["nested"]["diagnostic"] == "ValueError: useful explanation"


def test_error_payload_sanitizer_preserves_cycle_and_depth_diagnostics():
    cyclic = {"diagnostic": "RuntimeError: useful"}
    cyclic["nested"] = cyclic
    deep = current = {}
    for _index in range(20):
        current["nested"] = {}
        current = current["nested"]

    sanitized_cycle = sanitize_error_payload(cyclic)
    sanitized_depth = sanitize_error_payload(deep)

    assert sanitized_cycle["diagnostic"] == "RuntimeError: useful"
    assert sanitized_cycle["nested"] == "<recursive-error-payload>"
    cursor = sanitized_depth
    for _index in range(16):
        cursor = cursor["nested"]
    assert cursor == "<error-payload-depth-limit>"


def test_extract_error_type():
    assert extract_error_type("Traceback:\n  File ...\nKeyError: 'foo'") == "KeyError"
    assert extract_error_type("") == ""


def test_compact_traceback_preserves_final_exception_and_user_frame():
    noisy_frames = "\n".join(
        f'  File "/usr/lib/library_{index}.py", line {index}, in helper'
        for index in range(120)
    )
    traceback_text = (
        "Traceback (most recent call last):\n"
        "  Cell In[9], line 17, in <module>\n"
        "    analyze(dataset)\n"
        f"{noisy_frames}\n"
        "ValueError: intermediate failure\n\n"
        "During handling of the above exception, another exception occurred:\n\n"
        "Traceback (most recent call last):\n"
        "  File \"wrapper.py\", line 99, in _scai_cleanup_plt\n"
        "    _scai_cleanup_plt()\n"
        "  Cell In[9], line 18, in <module>\n"
        "    finalize_result()\n"
        "TypeError: final scientific failure"
    )

    compacted = compact_traceback(traceback_text, max_chars=900)

    assert len(compacted) <= 900
    assert "Cell In[9]" in compacted
    assert "TypeError: final scientific failure" in compacted
    assert "_scai_cleanup_plt" not in compacted
    assert extract_error_type(compacted) == "TypeError"


def test_compact_traceback_redacts_credentials_before_persisting():
    secret = "synthetic-fixture-credential-value"
    compacted = compact_traceback(
        f"Traceback:\napi_key='{secret}'\nValueError: request failed with Bearer {secret}",
        max_chars=600,
    )

    assert secret not in compacted
    assert "ValueError: request failed" in compacted


def test_compact_traceback_redacts_private_absolute_paths():
    traceback_text = (
        "Traceback (most recent call last):\n"
        '  File "/home/synthetic-user/private-project/analysis.py", line 4\n'
        '  File "/workspace/synthetic-project/backend/task.py", line 8\n'
        '  File "/data/sessions/synthetic-session-id/work/result.py", line 9\n'
        "RuntimeError: synthetic final failure"
    )

    compacted = compact_traceback(traceback_text, max_chars=1000)

    assert "synthetic-user" not in compacted
    assert "synthetic-project" not in compacted
    assert "synthetic-session-id" not in compacted
    assert "<private-path>" in compacted
    assert "<workspace-path>" in compacted
    assert "<session-path>" in compacted
    assert "RuntimeError: synthetic final failure" in compacted


def test_detect_swallowed_error_scvelo_case():
    """Caught AttributeError repr + skip decision → returns 'AttributeError'."""
    output = (
        "Innovation phase package status: {'scvelo': 'available'}\n"
        "{'condition': 'Con', 'status': 'failed', "
        "'error': \"AttributeError(\\\"module 'scvelo' has no attribute 'read'\\\")\", "
        "'decision': 'skip_velocity'}"
    )
    assert detect_swallowed_error(output) == "AttributeError"


def test_detect_swallowed_error_str_form():
    """Str-form 'ValueError: ...' with a 'skipping' keyword is detected."""
    output = "ValueError: not enough cells\nskipping trajectory analysis"
    assert detect_swallowed_error(output) == "ValueError"


def test_detect_swallowed_error_no_skip_keyword_returns_none():
    """A technical error with NO skip/abandon decision must NOT trigger (the
    agent may fix it on the next step — do not nag)."""
    output = "AttributeError(\"module 'scvelo' has no attribute 'read'\")\nRetrying with read_loom..."
    assert detect_swallowed_error(output) is None


def test_detect_swallowed_error_legit_data_skip_returns_none():
    """A data-driven skip with no technical error token must NOT trigger."""
    output = "No spliced/unspliced layers present; skip_velocity"
    assert detect_swallowed_error(output) is None


def test_detect_swallowed_error_empty_returns_none():
    assert detect_swallowed_error("") is None
    assert detect_swallowed_error(None) is None


def test_hyphenated_skip_decision_detected():
    output = "AttributeError: 'X' object has no attribute 'y'\nskip-velocity"
    assert detect_swallowed_error(output) == "AttributeError"


def test_underscore_skip_decision_still_detected():
    output = "ValueError: bad input\nskip_trajectory"
    assert detect_swallowed_error(output) == "ValueError"
