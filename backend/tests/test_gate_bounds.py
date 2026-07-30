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
