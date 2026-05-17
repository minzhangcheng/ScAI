"""Tests for error_parsing utilities."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.utils.error_parsing import (
    TECHNICAL_ERROR_TYPES,
    API_ERROR_TYPES,
    CODE_GEN_ERROR_TYPES,
    strip_ansi,
    extract_error_type,
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


def test_extract_error_type():
    assert extract_error_type("Traceback:\n  File ...\nKeyError: 'foo'") == "KeyError"
    assert extract_error_type("") == ""
