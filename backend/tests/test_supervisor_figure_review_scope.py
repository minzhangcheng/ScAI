"""Repeated visual rejection is not an upstream-analysis defect.

`_supervisor_figure_review` used to keep a per-figure-name rejection counter on
the orchestrator instance -- alive for the whole session -- and use it to bound
a loop that is re-entered per render (`figures.py`: `revision_count = 0` then
`while revision_count <= 3`). Once a figure name had been rejected twice, every
later render of that name escalated to `requires_reanalysis` on its FIRST
review, with no revision attempted.

The escalation is also the wrong kind of claim: a low visual score becomes an
assertion that the upstream analysis is defective, which stamps the figure
REVISION_REQUESTED. That status is not report-eligible, so the figure can never
satisfy the plan's visual contract afterwards -- the abandoned-figure state that
the workflow figure gate then had to be taught not to refuse on.

The bounded producer review loop already owns retry exhaustion. This layer does
not need to, and its counter was in the wrong scope to do it correctly anyway.
"""
import inspect
import re

from backend.workflow.callbacks import CallbacksMixin


def _review_source() -> str:
    return inspect.getsource(CallbacksMixin._supervisor_figure_review)


def test_no_session_scoped_rejection_counter_bounds_a_per_render_loop():
    src = _review_source()
    assert "_sup_rejections" not in src, (
        "a counter living on the orchestrator instance cannot bound a loop that "
        "is re-entered per render; it only starves later renders of revisions"
    )


def test_a_low_visual_score_is_never_promoted_to_an_upstream_defect():
    src = _review_source()
    # The assignment, not the word: the docstring names the thing it refuses
    # to do, and a bare substring check would read that prose as the code.
    assert not re.search(r"""\[.?requires_reanalysis.?\]\s*=""", src), (
        "escalating here stamps the figure REVISION_REQUESTED, which is not "
        "report-eligible, so the figure can never satisfy the plan contract"
    )


def test_the_relaxed_sole_reviewer_threshold_is_kept():
    """The useful half of this branch stays: a small model must not block."""
    src = _review_source()
    assert "supervisor_is_sole_reviewer" in src
    assert "relaxed" in src.lower()
