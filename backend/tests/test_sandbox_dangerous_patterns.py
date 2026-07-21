from backend.utils.sandbox import DANGEROUS_PATTERNS, validate_code_safety
import re

def _matches_any(code):
    for pattern, desc in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            return True
    return False

def test_detects_builtins_eval():
    code = "getattr(__builtins__, 'eval')('1+1')"
    assert _matches_any(code)

def test_allows_dangerous_builtin_names_used_only_as_data():
    code = (
        "splits = {'eval': sample_ids, 'train': train_ids}\n"
        "scores = frame['eval']\n"
        "labels = ('train', 'eval')\n"
    )
    assert validate_code_safety(code) == (True, "")


def test_allows_string_alias_when_it_is_not_invoked():
    assert validate_code_safety('fn_name = "eval"\nprint(fn_name)') == (True, "")


def test_blocks_dynamic_builtin_retrieval_and_invocation():
    blocked = (
        "globals()['eval']('1 + 1')",
        "name = 'ev' + 'al'\nfn = globals()[name]\nfn('1 + 1')",
        "import builtins\ngetattr(builtins, 'eval')('1 + 1')",
        "fn = eval\nfn('1 + 1')",
    )
    for code in blocked:
        safe, reason = validate_code_safety(code)
        assert not safe, code
        assert "dangerous" in reason.lower() or "blocked" in reason.lower()


def test_allows_regex_compile():
    assert validate_code_safety("import re\npattern = re.compile(r'abc')") == (True, "")


def test_allows_safe_importlib_metadata_and_util():
    code = (
        "import importlib.util\n"
        "from importlib.metadata import version\n"
        "print(importlib.util.find_spec('numpy'), version('numpy'))"
    )
    assert validate_code_safety(code) == (True, "")


def test_blocks_unsafe_importlib_submodule():
    safe, reason = validate_code_safety("import importlib.machinery")
    assert not safe
    assert "importlib" in reason
