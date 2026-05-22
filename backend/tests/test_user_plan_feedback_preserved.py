"""P2-6: user_plan_feedback must not be cleared before the revision call returns
successfully. Otherwise, if revise_plan_incrementally raises, the user's
feedback is permanently lost and they must re-type it on the next attempt."""
import re

import backend.workflow.planning as planning_mod


def test_user_plan_feedback_cleared_only_on_success():
    src = open(planning_mod.__file__).read()

    # For every assignment that clears user_plan_feedback to None, ensure that
    # the revise_plan_incrementally call appears in the preceding context.
    # Assignments inside the round-reset block (`_pi_planning`) are exempt —
    # they happen before any plan exists and have no feedback to lose.
    for m in re.finditer(r"user_plan_feedback\"?\]?\s*=\s*None", src):
        offset = m.start()
        preceding = src[max(0, offset - 1500):offset]

        # Exempt: the round-reset clearing in _pi_planning (no revise nearby).
        if "Reset plan-related state for new round" in preceding[-800:]:
            continue
        # Exempt: helper definition at module top.
        if "def _consume_user_plan_feedback" in preceding[-400:]:
            continue

        has_revise_before = bool(
            re.search(r"revise_plan_incrementally\s*\(", preceding)
        )
        assert has_revise_before, (
            f"P2-6 violation: user_plan_feedback cleared at offset {offset} "
            "before revise_plan_incrementally call. Feedback would be lost on "
            "revision failure."
        )


def test_consume_helper_not_called_before_revise():
    """The `_consume_user_plan_feedback` helper clears state. It must not be
    called inline immediately before revise_plan_incrementally — instead the
    code should read the value, call revise, and only clear on success."""
    src = open(planning_mod.__file__).read()

    # Find every call site of _consume_user_plan_feedback (excluding its def).
    for m in re.finditer(r"_consume_user_plan_feedback\s*\(", src):
        # Skip the definition line.
        line_start = src.rfind("\n", 0, m.start()) + 1
        line = src[line_start:src.find("\n", m.start())]
        if line.lstrip().startswith("def "):
            continue
        # Look ahead a short window for a revise_plan_incrementally call.
        following = src[m.start():m.start() + 600]
        if "revise_plan_incrementally" in following:
            raise AssertionError(
                "P2-6 violation: _consume_user_plan_feedback (which clears the "
                "field) is invoked immediately before revise_plan_incrementally. "
                "On revision failure, the feedback is lost."
            )
