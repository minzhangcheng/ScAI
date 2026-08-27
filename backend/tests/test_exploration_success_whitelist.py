"""P1-12: whitelist semantics for _exploration_round_succeeded.

Only a clean settlement -- status "completed", or "partial" with findings --
counts as success. All other terminal/non-terminal statuses (error, failed,
timeout, max_steps_reached, completed_with_fallback, waiting_data, skipped,
no_data, empty, missing) must NOT advance the workflow into the innovation
phase.

Rationale (docs/problems.md P1-12): the previous blacklist treated unknown
or partial-failure statuses as success, causing premature EXPLORATION ->
INNOVATION transition with empty exploration reports.

"partial" is on the success side because it does not mean the loop stopped
early: it is what the executor returns when report_findings was accepted but
a bounded gate force-released, and what workflow.agents substitutes when the
required-figure contract is unmet. Both are quality disclosures carried
separately by evidence_gate_degraded / evidence_verified. This list did not
have an entry for it when "partial" was introduced in 0.34.9, and the
resulting refusal deadlocked a real session out of the innovation phase --
see test_exploration_partial_status.py.
"""
import re
import pytest

import backend.workflow.synthesis as synth_mod
from backend.workflow.synthesis import SynthesisMixin


class _Probe(SynthesisMixin):
    def __init__(self):
        pass


@pytest.mark.parametrize(
    "status,expected_success",
    [
        ("completed", True),
        # Findings are supplied by the fixture below, so this is the
        # "partial with substantive findings" case. The empty-findings case
        # lives in test_exploration_partial_status.py.
        ("partial", True),
        ("completed_with_fallback", False),
        ("error", False),
        ("failed", False),
        ("timeout", False),
        ("max_steps_reached", False),
        ("waiting_data", False),
        ("waiting_input", False),
        ("skipped", False),
        ("no_data", False),
        ("unavailable", False),
        ("", False),
        (None, False),
    ],
)
def test_exploration_round_succeeded_whitelist(status, expected_success):
    probe = _Probe()
    if status is None:
        # status key entirely missing
        state = {"bioinformatics_result": {}}
    else:
        state = {"bioinformatics_result": {"status": status, "findings": ["x"]}}
    result = probe._exploration_round_succeeded(state)
    assert result is expected_success, (
        f"status={status!r}: expected {expected_success}, got {result}"
    )


def test_no_bio_result_at_all_is_not_success():
    """When bioinformatics_result key is entirely absent, exploration did
    not produce confirmed findings — must not advance."""
    probe = _Probe()
    assert probe._exploration_round_succeeded({}) is False


def test_failed_node_still_blocks_even_with_completed_status():
    """@with_retry skip path sets failed_node — honor it over stale result."""
    probe = _Probe()
    state = {
        "failed_node": "bioinformatics_analysis",
        "bioinformatics_result": {"status": "completed", "findings": ["x"]},
    }
    assert probe._exploration_round_succeeded(state) is False


def test_uses_whitelist_source_check():
    """Static check: implementation must compare against 'completed' explicitly."""
    src = open(synth_mod.__file__).read()
    m = re.search(
        r"def _exploration_round_succeeded\b.*?(?=\n    (?:async )?def |\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert m, "could not locate _exploration_round_succeeded source"
    body = m.group(0)
    assert ('== "completed"' in body) or ("== 'completed'" in body), (
        "P1-12: must use whitelist (status == 'completed')"
    )
