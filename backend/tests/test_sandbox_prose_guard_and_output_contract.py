"""Tests for prose-aware safety matching and persistent-output names."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.utils.sandbox import (
    CodeExecutionResult,
    LocalJupyterSandbox,
    validate_code_safety,
)


def _sandbox(tmp_path, monkeypatch, name, *, before, after, result):
    sandbox = LocalJupyterSandbox(
        session_id=name,
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(
        sandbox,
        "_snapshot_user_globals",
        AsyncMock(side_effect=[before, after]),
    )
    monkeypatch.setattr(
        sandbox, "_execute_unlocked", AsyncMock(return_value=result)
    )
    return sandbox


def test_comment_and_string_mentions_are_not_calls():
    allowed = (
        "# do not use eval() or exec() here\nvalue = 1\n",
        "message = 'compile() and __import__() are not used'\n",
        'message = "eval(\'x\') is only documentation"\n',
        "text = '''\nDo not call exec(payload)\n'''\n",
    )
    for code in allowed:
        assert validate_code_safety(code) == (True, ""), code


def test_real_and_fstring_calls_remain_blocked():
    blocked = (
        "eval('1 + 1')",
        "exec('x = 1')",
        "__import__('os')",
        "compile('x = 1', '<s>', 'exec')",
        "f'{eval(\"1 + 1\")}'",
        "fn = eval\nfn('1 + 1')",
        "import builtins\ngetattr(builtins, 'eval')('1 + 1')",
    )
    for code in blocked:
        safe, reason = validate_code_safety(code)
        assert not safe, code
        assert "dangerous" in reason.lower() or "blocked" in reason.lower()


def test_unparseable_source_falls_back_to_raw_scan():
    from backend.utils.sandbox_parts.shared import _mask_prose

    # Tokenization succeeds, but the malformed control statement keeps the
    # source unparsable; the raw dangerous call must therefore still block.
    code = "# prose about exec()\nif True print('bad')\n"
    assert _mask_prose(code) == code
    safe, reason = validate_code_safety(code)
    assert not safe
    assert "exec" in reason.lower()


def test_string_sensitive_patterns_still_use_raw_source():
    blocked = (
        ("open('/etc/scai-test', 'r+')", "file write"),
        ("io.open('/etc/scai-test', 'x')", "file write"),
        ("setattr(obj, '__class__', Other)", "setattr"),
        ("delattr(obj, '__dict__')", "delattr"),
    )
    for code, expected in blocked:
        safe, reason = validate_code_safety(code)
        assert not safe, code
        assert expected in reason.lower(), (code, reason)


@pytest.mark.asyncio
async def test_nonidentifier_declaration_is_ignored_with_guidance(
    tmp_path, monkeypatch
):
    sandbox = _sandbox(
        tmp_path,
        monkeypatch,
        "filename-guidance",
        before=set(),
        after=set(),
        result=CodeExecutionResult(success=True, output="saved"),
    )
    result = await sandbox.execute(
        "frame.to_csv('sample_summary_table.csv')",
        declared_outputs=[" sample_summary_table.csv ", "./notes.txt"],
        track_globals=True,
    )
    assert result.success is True
    assert "sample_summary_table.csv" in result.output
    assert "not Python variable names" in result.output
    assert "declare the in-memory object" in result.output


@pytest.mark.asyncio
async def test_valid_identifier_remains_required(tmp_path, monkeypatch):
    sandbox = _sandbox(
        tmp_path,
        monkeypatch,
        "missing-variable",
        before=set(),
        after=set(),
        result=CodeExecutionResult(success=True, output=""),
    )
    result = await sandbox.execute(
        "pass",
        declared_outputs=["adata", "adata.h5ad"],
        track_globals=True,
    )
    assert result.success is False
    assert "adata" in result.error
    assert "adata.h5ad" not in result.error
