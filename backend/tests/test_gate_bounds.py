"""Scientific observations may not trap or decide the analysis loop.

Missing score provenance or a frozen panel remains recorded, but registration
proceeds and the PI decides how the limitation affects acceptance. Structural
tool payload errors retain their normal correction path.
"""
import inspect
import re

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


def test_register_score_has_no_scientific_rejection_ceiling():
    src = _loop_source()
    assert "max_score_gate_rejections" not in src
    assert "score_gate_reject_count" not in src


def test_register_score_does_not_reject_missing_scientific_metadata():
    block = _register_score_block()
    before_registration = block[:block.index("context.has_register_score = True")]
    assert "return ToolAction.CONTINUE" not in before_registration
    assert "register_score rejected" not in block


def test_score_limitations_are_recorded_for_pi_review():
    block = _register_score_block()
    assert "review_notes" in block
    assert "pi_review_observations" in block
    assert 'severity="advise"' in block
    text = _joined(block)
    assert "unattributed" in text
    assert "not pre-specified" in text
    assert "PI must decide" in text


def test_score_is_registered_without_host_scientific_acceptance():
    block = _register_score_block()
    assert "context.has_register_score = True" in block
    assert "score_def = ScoreDefinition(" in block


def test_scientific_rejection_counters_are_retired():
    src = _loop_source()
    retired = (
        "max_score_gate_rejections",
        "score_gate_reject_count",
        "max_mechanism_rejections",
        "max_stats_gate_rejections",
        "max_integration_rejections",
        "max_bulk_rejections",
        "max_evidence_citation_refusals",
        "max_deconv_rejections",
        "max_innovation_bulk_rejections",
    )
    assert not [name for name in retired if name in src]


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
    return inspect.getsource(executor_report_gates.attach_report_observations)


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
    return inspect.getsource(
        executor_report_gates.record_report_evidence_candidates
    )


def test_evidence_candidates_do_not_count_or_trigger_refusals():
    block = _evidence_citation_gate()
    assert "evidence_citation_reject_count" not in block
    assert "return False" not in block


def test_evidence_candidates_remain_explicitly_unverified():
    block = _evidence_citation_gate()
    assert "candidate_evidence_execution_ids" in block
    assert 'record["evidence_execution_ids"] = []' in block
    assert 'record["semantic_status"] = "unverified"' in block


def test_evidence_candidate_normalization_has_no_acceptance_boolean():
    block = _evidence_citation_gate()
    assert "return True" not in block
    assert "return False" not in block
