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
from backend.tests.split_facade import split_source


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def test_escaped_notes_init_is_in_common_scope():
    src = split_source("backend.agents.bioinformatics_agent")
    lines = src.splitlines()

    init_lines = [ln for ln in lines if ln.strip() == "_escaped_notes = []"]
    assert len(init_lines) == 1, "expected exactly one `_escaped_notes = []` init"

    consumer_lines = [ln for ln in lines if "escaped_notes=_escaped_notes" in ln]
    assert len(consumer_lines) == 1, "expected exactly one handoff to the report attempt"

    init_idx = lines.index(init_lines[0])
    handoff_idx = lines.index(consumer_lines[0])
    assert init_idx < handoff_idx, (
        "_escaped_notes must be initialized before it is handed to ReportAttempt"
    )
