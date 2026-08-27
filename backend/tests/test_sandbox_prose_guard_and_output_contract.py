"""Tests for prose-aware safety matching and persistent-output names."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from backend.utils.sandbox import (
    CodeExecutionResult,
    DANGEROUS_PATTERNS,
    LocalJupyterSandbox,
    validate_code_safety,
)
from backend.tests.split_facade import source_for_path


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


def test_mask_preserves_character_offsets_and_newlines():
    from backend.utils.sandbox_parts.shared import _mask_prose

    code = (
        "# a comment about exec()\n"
        "message = 'eval() is prose'\n"
        "exec('x = 1')\n"
    )
    masked = _mask_prose(code)
    assert len(masked) == len(code)
    assert masked.count("\n") == code.count("\n")
    raw = list(re.finditer(r"\bexec\s*\(", code))
    matched = re.search(r"\bexec\s*\(", masked)
    assert raw and matched
    assert matched.start() == raw[-1].start()
    assert "exec(" not in masked.splitlines()[0]
    assert "eval(" not in masked.splitlines()[1]


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


def test_block_log_localizes_original_match(caplog):
    code = "# prose about exec()\nimport gc\nexec('x = 1')\n"
    with caplog.at_level(logging.WARNING):
        safe, _ = validate_code_safety(code)
    assert not safe
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "line 3" in logged, logged
    assert "exec('x = 1')" in logged, logged


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


def test_no_tracker_fallback_cannot_readd_nonidentifier():
    # bioinformatics_agent.py is a partial facade: part of its implementation
    # was split into backend.agents.bioinformatics, so reading the path alone
    # can scan a re-export header and make this assertion pass over nothing.
    source = source_for_path("backend/agents/bioinformatics_agent.py")
    marker = "No tracker (should not happen with track_globals)"
    fallback = source[source.index(marker):source.index("sandbox_state[", source.index(marker))]
    assert "isidentifier()" in fallback


def test_tool_schema_describes_variable_names():
    from backend.agents.bioinformatics.tool_schemas import TOOLS

    execute_code = next(
        tool["function"] for tool in TOOLS
        if tool["function"]["name"] == "execute_code"
    )
    description = execute_code["parameters"]["properties"]["persistent_outputs"][
        "description"
    ]
    assert "variable names only" in description.lower()
    assert "filename" in description.lower()
    assert "identifiers" in description.lower()


def test_s4_module_qualified_patterns_are_unchanged():
    expected = {
        (r"\bshutil\.rmtree\b", "shutil.rmtree"),
        (r"\bos\.remove\b", "os.remove"),
        (r"\bos\.unlink\b", "os.unlink"),
        (r"\bos\.rmdir\b", "os.rmdir"),
        (r"\bos\.rename\b", "os.rename"),
        (r"\bos\.replace\b", "os.replace"),
        (r"\bshutil\.move\b", "shutil.move"),
    }
    assert expected.issubset(set(DANGEROUS_PATTERNS))
    digest = hashlib.sha256(repr(DANGEROUS_PATTERNS).encode()).hexdigest()
    assert digest == "38b95c8313c2b5911319166c123be26446c0e29d0853c01061562f242cb48f71"
