from backend.utils.sandbox import DANGEROUS_PATTERNS
import re

def _matches_any(code):
    for pattern, desc in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            return True
    return False

def test_detects_builtins_eval():
    code = "getattr(__builtins__, 'eval')('1+1')"
    assert _matches_any(code)

def test_detects_string_eval():
    code = """fn = "eval"\n"""
    assert _matches_any(code)
