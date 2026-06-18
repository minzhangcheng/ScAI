"""Regression guard for F1: `_escaped_notes` must be initialized in the COMMON
scope of the report_findings handler, not nested inside the EXPLORATION-only
`if` block. Otherwise the INNOVATION path and stage1 exploration crash with
NameError when the post-gate `for note in _escaped_notes` loop runs.

This is a source-level test (matching the repo's existing source-guard tests,
e.g. test_no_dead_modality_helpers.py) because the consuming code is a large
method inside an agent loop that cannot be exercised in isolation cheaply.
"""
import inspect
import re

from backend.agents import bioinformatics_agent


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def test_escaped_notes_init_is_in_common_scope():
    src = inspect.getsource(bioinformatics_agent)
    lines = src.splitlines()

    init_lines = [ln for ln in lines if ln.strip() == "_escaped_notes = []"]
    assert len(init_lines) == 1, "expected exactly one `_escaped_notes = []` init"

    consumer_lines = [ln for ln in lines if re.match(r"\s*for note in _escaped_notes:", ln)]
    assert len(consumer_lines) == 1, "expected exactly one consuming for-loop"

    # The init must be at the SAME indentation as the consuming loop (common
    # scope), never more-indented (nested inside the EXPLORATION-only branch).
    assert _indent(init_lines[0]) == _indent(consumer_lines[0]), (
        "_escaped_notes must be initialized in the same scope as its consumer; "
        f"init indent={_indent(init_lines[0])} consumer indent={_indent(consumer_lines[0])}"
    )
