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


def test_blocks_posix_spawn_direct_alias_and_function_alias_calls():
    blocked = (
        "import os\nos.posix_spawn('/bin/true', ['true'], {})",
        "import os\nos.posix_spawnp('true', ['true'], {})",
        "import os as operating_system\noperating_system.posix_spawn('/bin/true', ['true'], {})",
        "from os import posix_spawn as launch\nlaunch('/bin/true', ['true'], {})",
        "import os\nlaunch = os.posix_spawn\nlaunch('/bin/true', ['true'], {})",
    )
    for code in blocked:
        safe, reason = validate_code_safety(code)
        assert not safe, code
        assert "posix_spawn" in reason


def test_blocks_import_only_process_aliases_before_they_persist_to_next_cell():
    blocked = (
        "from os import posix_spawn as launch",
        "from os import posix_spawnp as launch",
        "import subprocess",
        "from subprocess import run as launch",
    )
    for code in blocked:
        safe, reason = validate_code_safety(code)
        assert not safe, code
        assert "process" in reason.lower()


def test_blocks_reserved_kernel_namespace_direct_and_globals_mutation():
    blocked = (
        "_scai_pristine_json_dumps = lambda value: '[]'",
        "print(_scai_pristine_stdout)",
        "globals()['_scai_pristine_print'] = print",
        "globals()['_' + 'scai_pristine_print'] = print",
        "globals().update({'_scai_pristine_print': print})",
        "globals().setdefault('_scai_pristine_print', print)",
    )
    for code in blocked:
        safe, reason = validate_code_safety(code)
        assert not safe, code
        assert "reserved" in reason.lower()

    # A data column or string value is not namespace access.
    assert validate_code_safety(
        "frame['_scai_measurement'] = values"
    ) == (True, "")
    assert validate_code_safety(
        "namespace = globals()\nnamespace['user_result'] = 1"
    ) == (True, "")


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


def test_static_file_write_check_covers_update_and_attribute_open_modes():
    blocked = (
        "open('/etc/scai-test', 'r+')",
        "io.open('/etc/scai-test', 'x')",
        "Path('/etc/scai-test').open('a+')",
    )
    for code in blocked:
        safe, reason = validate_code_safety(code)
        assert not safe, code
        assert "file write" in reason.lower()


def test_static_file_write_check_preserves_reads_and_dynamic_runtime_checks():
    allowed = (
        "open('/etc/hosts', 'rb')",
        "Path('tables/result.txt').open('r+')",
        "open(output_path, mode)",
    )
    for code in allowed:
        assert validate_code_safety(code) == (True, ""), code
