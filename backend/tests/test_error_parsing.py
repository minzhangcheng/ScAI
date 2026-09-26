"""Tests for error_parsing utilities."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.utils.error_parsing import (
    compact_traceback,
    sanitize_error_payload,
    extract_error_type,
)


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
