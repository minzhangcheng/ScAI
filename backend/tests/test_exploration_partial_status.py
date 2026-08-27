"""Regression: a 'partial' bioinformatics status must not fail exploration.

Measured on session 3d675631-8c5f-4a6a-ad66-ef4c8996bb44 (scai-2, 0.48.1).
Stage 1 ran to completion -- 39 steps, 10 findings, 22 saved figures, an
accepted report_findings call -- but the statistical gate
'unidentifiable-design' fired on a one-sample-per-condition design (Con and
DKO, n=1 each). That gate is in the executor's `_unsatisfiable` set, so it
force-released at its 1-rejection ceiling and set evidence_gate_degraded,
which maps to status='partial' in bioinformatics_agent.

_exploration_round_succeeded then read 'partial' as a failure and emitted
exploration_failed(reason='did_not_converge'), whose dialog blames the
uploaded data and offers "retry Stage 1". The retry cannot help: n=1 per
group is a property of the experiment, so every retry re-triggers the same
gate. The user is deadlocked out of the innovation phase.

'partial' post-dates the whitelist. It was added in 7401f052 (0.34.9,
2026-07-21); the whitelist was written in b6c6cff2 (2026-05-22) and its
parametrized guard never listed it. Every other consumer on the same
pipeline already treats it as a clean settlement -- see
helpers.decide_bioinfo_post_execution and workflow.agents' `status_was_unclean`,
both of which spell the clean set as ("completed", "partial").

'partial' is a disclosure of degraded evidence, not a claim of incomplete
execution. The degradation still reaches the report through
evidence_gate_degraded / evidence_verified / evidence_notes, which the
Report A generation and the report hedging read directly.
"""
import re

import pytest

import backend.workflow.synthesis as synth_mod
from backend.workflow.synthesis import SynthesisMixin


class _Probe(SynthesisMixin):
    def __init__(self):
        pass


def _degraded_gate_state(findings):
    """The shape workflow.agents hands to synthesis for the measured session."""
    return {
        "bioinformatics_result": {
            "status": "partial",
            "findings": findings,
            "evidence_verified": False,
            "evidence_gate_degraded": True,
            "evidence_gate_findings": [
                {
                    "gate": "unidentifiable-design",
                    "rule_id": "one-sample-per-condition-is-unidentifiable",
                    "detail": "group(s) Con, DKO have fewer than 2 samples",
                    "severity": "block",
                    "rejection_count": 1,
                    "rejection_limit": 1,
                    "forced_release": True,
                },
            ],
        },
    }


def test_force_released_gate_does_not_fail_exploration():
    """The measured session: substantive findings behind a force-released gate."""
    probe = _Probe()
    state = _degraded_gate_state([{"finding": "hepatocyte state transition"}])
    assert probe._exploration_round_succeeded(state) is True


def test_partial_from_unmet_figure_contract_does_not_fail_exploration():
    """workflow.agents downgrades 'completed' -> 'partial' when the required-figure
    contract is unmet. The executor demoted that same check to a non-blocking
    observation on purpose ("it is not a gate -- it is a trap"), and
    decide_bioinfo_post_execution deliberately omits it as a clause. It must not
    become blocking again one frame up."""
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
    state = _degraded_gate_state(empty)
    assert probe._exploration_round_succeeded(state) is False


def test_partial_still_honors_failed_node():
    """@with_retry's HITL Skip path outranks any status the stale result carries."""
    probe = _Probe()
    state = _degraded_gate_state([{"finding": "x"}])
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
