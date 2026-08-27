"""No gate may trap the analysis loop.

Every report_findings gate in this agent carries a ceiling and an escape hatch,
on the principle that a check the agent cannot satisfy must not stall the run.
The register_score gates added alongside them initially did not, which mattered
most for the plan_id requirement: unlike a provenance error, it cannot be fixed
from inside register_score at all -- it needs a prior register_analysis_plan
call. A model that misses that connection would retry forever, and because a
rejected call never sets has_register_score, each attempt also burns a step
instead of being refunded.

These tests pin the ceiling, the escape, and the fact that an escape is
recorded rather than silently blessed.
"""
import inspect
import re

import pytest

import backend.agents.bioinformatics_agent as bio_mod


def _loop_source() -> str:
    return inspect.getsource(bio_mod)


def _register_score_block() -> str:
    src = _loop_source()
    start = src.index('elif tool_name == "register_score":')
    end = src.index('elif tool_name == "register_mechanism_programs":', start)
    return src[start:end]


def _joined(block: str) -> str:
    """Source with adjacent string literals joined and whitespace collapsed.

    Long messages are written as implicitly concatenated literals across lines,
    so a phrase that reads contiguously in the emitted text is split in the
    source. Matching the raw source would make these tests fail on reflow
    rather than on behaviour.
    """
    joined = re.sub(r'"\s*\n\s*(?:f?)"', "", block)
    return re.sub(r"\s+", " ", joined)


def test_score_gate_has_a_ceiling():
    src = _loop_source()
    assert "MAX_SCORE_GATE_REJECTIONS" in src
    assert "score_gate_reject_count" in src


def test_every_score_gate_rejection_is_counted():
    """An uncounted `continue` is an unbounded loop by another name."""
    block = _register_score_block()
    # Each rejection path must sit inside a counter comparison.
    assert block.count("score_gate_reject_count < MAX_SCORE_GATE_REJECTIONS") >= 2, (
        "both the provenance and the plan rejection paths must be bounded"
    )


def test_score_gate_escape_records_what_was_not_established():
    """Force-releasing a gate silently would be worse than not having it."""
    block = _register_score_block()
    assert "gate_note" in block
    assert "gate_escape" in block, "the escape must be recorded on the score"
    text = _joined(block)
    assert "unattributed" in text
    assert "not pre-specified" in text


def test_score_is_still_registered_after_the_ceiling():
    """Blocking registration forever would leave the methodology section empty."""
    block = _register_score_block()
    text = _joined(block)
    escape_idx = text.index("max rejections reached, recording score as unattributed")
    tail = text[escape_idx:]
    assert "has_register_score = True" in tail or "score_def = ScoreDefinition(" in tail


@pytest.mark.parametrize("ceiling", [
    "MAX_MECHANISM_REJECTIONS",
    "MAX_STATS_GATE_REJECTIONS",
    "MAX_INTEGRATION_REJECTIONS",
    "MAX_DECONV_REJECTIONS",
    "MAX_SCORE_GATE_REJECTIONS",
])
def test_all_gate_ceilings_are_defined(ceiling):
    """Keep the set of bounded gates visible in one place."""
    src = _loop_source()
    assert re.search(rf"{ceiling}\s*=\s*\d+", src), f"{ceiling} has no numeric ceiling"


def test_no_unbounded_rejection_added_to_register_analysis_plan():
    """Plan rejections are payload errors, fixable in one retry -- but check they
    cannot silently accumulate either."""
    src = _loop_source()
    start = src.index('elif tool_name == "register_analysis_plan":')
    end = src.index('elif tool_name == "register_score":', start)
    block = src[start:end]
    # These are structural payload errors the model fixes by correcting its own
    # arguments, so they follow the existing register_mechanism_programs pattern.
    # What must hold is that each one still emits a tool message before continuing.
    assert block.count('"role": "tool"') >= block.count("continue"), (
        "every rejection path must emit a tool message, or the tool_call is orphaned"
    )


# ---------------------------------------------------------------------------
# The metadata-only step refund is the same invariant seen from the other side.
# A gate that refuses costs the run a step; a refund that never stops costs the
# run its bound. `while step_count < max_steps` is the executor loop's only
# terminator, so an unconditional `step_count -= 1` on a turn the model can
# repeat at will means max_steps no longer bounds anything -- the run can spend
# an arbitrary number of LLM turns and then report that it executed 0 steps.
# ---------------------------------------------------------------------------

def _metadata_refund_block() -> str:
    """The `not has_substantive_work` branch, where free turns are refunded."""
    src = _loop_source()
    start = src.index("if not has_substantive_work and response.has_tool_calls:")
    end = src.index("if llm_error:", start)
    return src[start:end]


def test_every_metadata_refund_is_bounded_by_a_ceiling():
    """No refund branch may decrement the step budget unconditionally.

    Both known free-turn kinds -- a rendered figure and a registered score --
    are turns the model can emit indefinitely. Each refund must sit under a
    comparison against its own ceiling, so a third one added later without a
    bound fails here rather than in a run that never ends.
    """
    block = _metadata_refund_block()
    refunds = [
        line for line in block.splitlines() if line.strip() == "step_count -= 1"
    ]
    assert refunds, "the metadata refund branch was not found"

    joined = _joined(block)
    ceilings = re.findall(r"MAX_FREE_[A-Z_]+_ITERATIONS", joined)
    guards = re.findall(r"<=\s*MAX_FREE_[A-Z_]+_ITERATIONS", joined)
    assert len(guards) == len(refunds), (
        f"{len(refunds)} refund(s) but {len(guards)} ceiling guard(s): "
        "an unguarded refund removes the loop's only bound"
    )
    assert "MAX_FREE_REGISTER_SCORE_ITERATIONS" in ceilings
    assert "MAX_FREE_FIGURE_ITERATIONS" in ceilings


def test_an_exhausted_allowance_charges_a_normal_step():
    """Past the ceiling the turn must cost a step, not be silently dropped.

    A ceiling that merely stops refunding without an explicit else would be
    easy to write as a no-op; pin that both allowances say so.
    """
    joined = _joined(_metadata_refund_block())
    assert "Free figure allowance exhausted" in joined
    assert "Free register_score allowance exhausted" in joined


# ---------------------------------------------------------------------------
# The Stage-2 mandatory-analysis gates are one family. Each asks the same
# question -- did the agent's code mention this analysis -- and each must ask it
# through the bounded helper, so a model that cannot satisfy one is released
# after a ceiling with the escape on the record.
#
# Gate 1 was the family's only outlier: a hand-rolled keyword test ending in a
# bare `continue`, with no counter, no ceiling and no record. Its refusal also
# burns a step and is invisible to the supervisor, whose stuck-detection needs a
# code_error the refusal step does not carry -- so nothing else in the system
# could end it either.
# ---------------------------------------------------------------------------

_KEYWORD_GATES = [
    ("Gate 1 (bulk RNA-seq)",
     "# Gate 1: Bulk RNA-seq analysis", "# Gate 2:"),
    ("Gate 2 (bulk/scRNA cross-validation)",
     "# Gate 2: Bulk/scRNA-seq cross-validation", "# Gate 3:"),
    ("Gate 3 (deconvolution)",
     "# Gate 3: Cell-type deconvolution", "# Gate 4:"),
    ("Gate 4 (innovation bulk)",
     "# Gate 4: INNOVATION-phase bulk direction validation",
     "eligible_evidence_ids = "),
]


def _gate_block(start_marker: str, end_marker: str) -> str:
    src = _loop_source()
    start = src.index(start_marker)
    return src[start:src.index(end_marker, start)]


@pytest.mark.parametrize(("label", "start", "end"), _KEYWORD_GATES)
def test_every_stage2_keyword_gate_is_bounded(label, start, end):
    block = _gate_block(start, end)
    assert "_check_gate_keyword" in block, (
        f"{label} tests the agent's code for keywords without the bounded "
        "helper, so it can refuse report_findings for the whole step budget"
    )
    assert re.search(r"MAX_[A-Z_]+_REJECTIONS", block), (
        f"{label} routes through the helper but names no ceiling"
    )


@pytest.mark.parametrize(("label", "start", "end"), _KEYWORD_GATES)
def test_every_stage2_keyword_gate_records_its_escape(label, start, end):
    block = _gate_block(start, end)
    assert "if escaped:" in block, (
        f"{label} never handles its own release, so the ceiling is unobservable"
    )
    assert "_evidence_gate_record" in block, (
        f"{label} releases without recording it, and the round then reports as "
        "though the mandatory analysis had been performed"
    )


def test_the_hand_rolled_bulk_keyword_test_is_gone():
    """Pin the removal, so the outlier cannot return by copy-paste."""
    assert "code_mentions_bulk" not in _loop_source()


# ---------------------------------------------------------------------------
# The evidence-citation gate is deliberately NOT a member of the releasing
# family, and these tests pin that asymmetry rather than erase it. Releasing it
# would accept findings that cite no successful execution -- the one thing it
# exists to prevent -- and `main` does not release it either.
#
# What it lacked was any trace that it had fired. It refuses with a bare
# `continue`, and the refusal step carries no code_error, so the supervisor's
# stuck detection (which requires one) cannot see it. A run could spend its
# entire step budget on the same refusal and report nothing about why.
# ---------------------------------------------------------------------------

def _evidence_citation_gate() -> str:
    src = _loop_source()
    start = src.index("eligible_evidence_ids = {")
    return src[start:src.index("has_substantive_work = True", start)]


def test_the_evidence_citation_gate_counts_its_refusals():
    block = _evidence_citation_gate()
    assert re.search(r"evidence_citation_reject_count\s*\+=", block), (
        "the gate refuses without counting, so nothing downstream can tell one "
        "mis-cited call from a whole budget spent on the same refusal"
    )


def test_a_repeated_evidence_citation_refusal_reaches_the_pi():
    block = _evidence_citation_gate()
    assert "_evidence_gate_record" in block, (
        "a refusal that keeps repeating must be recorded; it is invisible to "
        "the supervisor, so this record is the only channel that carries it"
    )


def test_the_evidence_citation_gate_never_force_releases():
    """The asymmetry is the point: this one holds an integrity invariant.

    Every other gate in this file is required to release at a ceiling. This one
    is required NOT to: accepting findings that cite nothing would defeat the
    only thing it checks. If a future change gives it an escape hatch for
    symmetry's sake, this fails.
    """
    block = _evidence_citation_gate()
    assert "continue" in block, "the gate must still refuse"
    assert "forced_release=True" not in block, (
        "this gate must never force-release: it would accept findings citing "
        "no successful execution"
    )
