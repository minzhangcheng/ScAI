"""Regression: an operationally 'partial' result with findings may settle.

An earlier scai-2 run exposed the original deadlock: a host statistical veto
turned an n=1 design into partial and a second host check then failed the round.
Scientific checks no longer create partial status. This test preserves support
for operational partial results and legacy persisted sessions without reviving
the removed scientific veto.

'partial' post-dates the whitelist. It was added in 7401f052 (0.34.9,
2026-07-21); the whitelist was written in b6c6cff2 (2026-05-22) and its
parametrized guard never listed it. Every other consumer on the same
pipeline already treats it as a clean settlement -- see
helpers.decide_bioinfo_post_execution and workflow.agents' `status_was_unclean`,
both of which spell the clean set as ("completed", "partial").

'partial' remains a disclosure of operationally degraded evidence, not a host
scientific verdict.
"""
import re

import pytest

import backend.workflow.synthesis as synth_mod
from backend.workflow.synthesis import SynthesisMixin


class _Probe(SynthesisMixin):
    def __init__(self):
        pass


def _degraded_operational_state(findings):
    """A supervisor force-stop is operational, not scientific acceptance."""
    return {
        "bioinformatics_result": {
            "status": "partial",
            "findings": findings,
            "evidence_verified": False,
            "evidence_gate_degraded": True,
            "evidence_gate_findings": [
                {
                    "gate": "supervisor-force-stop",
                    "rule_id": "scai-native/supervisor-force-stop",
                    "detail": "execution stopped after repeated runtime errors",
                    "severity": "block",
                    "forced_release": True,
                },
            ],
        },
    }


def test_operational_partial_with_findings_does_not_fail_exploration():
    probe = _Probe()
    state = _degraded_operational_state([{"finding": "state transition"}])
    assert probe._exploration_round_succeeded(state) is True


def test_legacy_figure_partial_does_not_fail_exploration():
    """Old persisted partial records remain readable. New figure observations
    no longer downgrade a completed result."""
    probe = _Probe()
    state = {
        "bioinformatics_result": {
            "status": "partial",
            "findings": [{"finding": "metallothionein upregulation"}],
            "evidence_gate_degraded": False,
        },
    }
    assert probe._exploration_round_succeeded(state) is True


@pytest.mark.parametrize("empty", [[], None])
def test_partial_without_findings_still_blocks(empty):
    """P1-12's actual concern was an exploration report built from nothing.
    A partial round that produced no findings still has nothing to report on."""
    probe = _Probe()
    state = _degraded_operational_state(empty)
    assert probe._exploration_round_succeeded(state) is False


def test_partial_still_honors_failed_node():
    """@with_retry's HITL Skip path outranks any status the stale result carries."""
    probe = _Probe()
    state = _degraded_operational_state([{"finding": "x"}])
    state["failed_node"] = "bioinformatics_analysis"
    assert probe._exploration_round_succeeded(state) is False


def test_partial_acceptance_is_conditioned_on_findings():
    """Static guard: the 'partial' branch must stay conditioned on findings, so a
    later simplification cannot widen it into an unconditional pass."""
    src = open(synth_mod.__file__).read()
    m = re.search(
        r"def _exploration_round_succeeded\b.*?(?=\n    (?:async )?def |\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert m, "could not locate _exploration_round_succeeded source"
    body = m.group(0)
    assert '"partial"' in body or "'partial'" in body, (
        "partial must be handled explicitly, not by falling through to False"
    )
    assert "findings" in body, (
        "the partial branch must require substantive findings (P1-12)"
    )
