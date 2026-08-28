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
from backend.agents.bioinformatics import (
    executor_auxiliary,
    executor_dispatch,
    executor_loop,
    executor_registration,
    executor_report_gates,
)


def _loop_source() -> str:
    return "\n".join((
        inspect.getsource(bio_mod),
        inspect.getsource(executor_dispatch),
        inspect.getsource(executor_loop),
        inspect.getsource(executor_registration),
        inspect.getsource(executor_report_gates),
    ))


def _register_score_block() -> str:
    return inspect.getsource(executor_registration.handle_register_score)


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
    assert "max_score_gate_rejections" in src
    assert "score_gate_reject_count" in src


def test_every_score_gate_rejection_is_counted():
    """An uncounted `continue` is an unbounded loop by another name."""
    block = _register_score_block()
    # Each rejection path must sit inside a counter comparison.
    assert block.count(
        "score_gate_reject_count < context.max_score_gate_rejections"
    ) >= 2, (
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
    "max_mechanism_rejections",
    "max_stats_gate_rejections",
    "max_integration_rejections",
    "max_deconv_rejections",
    "max_score_gate_rejections",
])
def test_all_gate_ceilings_are_defined(ceiling):
    """Keep the set of bounded gates visible in one place."""
    src = _loop_source()
    assert re.search(rf"{ceiling}:\s*int\s*=\s*\d+", src), f"{ceiling} has no numeric ceiling"


def test_no_unbounded_rejection_added_to_register_analysis_plan():
    """Plan rejections are payload errors, fixable in one retry -- but check they
    cannot silently accumulate either."""
    block = inspect.getsource(
        executor_registration.handle_register_analysis_plan
    )
    # These are structural payload errors the model fixes by correcting its own
    # arguments, so they follow the existing register_mechanism_programs pattern.
    # What must hold is that each one still emits a tool message before continuing.
    assert block.count('"role": "tool"') >= (
        block.count("return ToolAction.CONTINUE") - 1
    ), (
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
    src = inspect.getsource(executor_loop.finish_turn)
    start = src.index("if not has_substantive_work and response.has_tool_calls:")
    end = src.index("context.text_response_count", start)
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
# Bulk reporting is gated by structured host facts and evidence lineage. Method
# choice belongs to PI/Bioinformatics; source-code vocabulary is not evidence
# that the selected analysis ran.
# ---------------------------------------------------------------------------


def _bulk_method_gate() -> str:
    return inspect.getsource(executor_report_gates.check_mandatory_report_gates)


def test_bulk_method_decision_gap_is_advisory_and_non_blocking():
    block = _bulk_method_gate()
    assert "state.method_decisions" in block
    assert "state.bulk_input_profiles" in block
    assert "attempt.evidence_executions" in block
    assert "_evidence_gate_record" in block
    assert 'gate="bulk-method-decision"' in block
    assert 'severity="advise"' in block
    bulk_branch = block.split("if has_bulk_data", 1)[1]
    assert "return False" not in bulk_branch


def test_bulk_gate_does_not_infer_execution_from_code_keywords():
    block = _bulk_method_gate()
    forbidden = (
        "_check_gate_keyword",
        "code_mentions_bulk",
        "bulk-rnaseq-analysis",
        "bulk-scrna-cross-validation",
        "cell-type-deconvolution",
        "innovation-bulk-validation",
    )
    assert not [token for token in forbidden if token in block]


# ---------------------------------------------------------------------------
# Producer citations are semantic candidates. Missing or mistaken IDs must not
# spend autonomous-agent turns; the Host retains the finding as unverified and
# the Supervisor decides whether any current execution actually supports it.
# ---------------------------------------------------------------------------

def _evidence_citation_gate() -> str:
    return inspect.getsource(executor_report_gates.check_report_evidence)


def test_evidence_candidates_do_not_count_or_trigger_refusals():
    block = _evidence_citation_gate()
    assert "evidence_citation_reject_count" not in block
    assert "return False" not in block


def test_evidence_candidates_remain_explicitly_unverified():
    block = _evidence_citation_gate()
    assert "candidate_evidence_execution_ids" in block
    assert 'record["evidence_execution_ids"] = []' in block
    assert 'record["semantic_status"] = "unverified"' in block


def test_evidence_candidate_normalization_always_returns_to_the_workflow():
    block = _evidence_citation_gate()
    assert block.rstrip().endswith("return True")
