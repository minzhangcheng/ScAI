"""The figure revision loop's exit conditions.

The loop is where a correctly-diagnosed figure gets shipped anyway. Two exits
did that: a "quality score stalled" rule that accepted the current version
whenever the score failed to rise (using ``<=``, so a score that *fell* also
counted as stalled), and reviewer approval, which nothing cross-checked. These
tests pin the repaired behaviour: a measured collision is not something either
exit may override, and a regression is rolled back rather than shipped.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.bioinformatics.figures import FiguresMixin
from backend.models.schemas import AnalysisState, FigureReviewStatus
from backend.services.figure_layout import LAYOUT_LINT_SENTINEL

BLOCKING = [{
    "kind": "text_over_text", "severity": "block", "text": "Con FXR/bile",
    "what": "legend entry", "collides_with": "Cyp7a1",
    "collides_with_what": "yticklabel", "overlap_px2": 465.8,
    "bbox": [531.5, 193.0, 649.8, 203.0],
}]
CLEAN: list = []


class _FakeSandbox:
    """Writes real files and returns a scripted lint verdict per execution.

    A step may be a plain violation list, or a dict describing an execution that
    did something less tidy: emitted no verdict, produced no figure at all, or
    rendered blank.
    """

    def __init__(self, working_dir, script):
        self.working_dir = str(working_dir)
        self._script = list(script)
        self.executions = 0

    async def execute(self, code, **kwargs):
        step = self._script[min(self.executions, len(self._script) - 1)]
        if not isinstance(step, dict):
            step = {"violations": step}
        self.executions += 1
        figures = Path(self.working_dir) / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        out = []
        if step.get("no_figure"):
            # the wrapper's own wording when the code left nothing to save;
            # the PREVIOUS file stays on disk untouched
            out.append("[Auto-save] No figures found — skipping save")
        else:
            size = int(step.get("size", 4000))
            # distinct bytes per execution so a rollback is detectable
            (figures / "f.jpg").write_bytes(b"J" * size + bytes([self.executions]) * 100)
            (figures / "f.pdf").write_bytes(b"P" * 3000 + bytes([self.executions]) * 100)
            out.append("Saved JPG")
        violations = step.get("violations", CLEAN)
        if violations is not None:
            out.append(LAYOUT_LINT_SENTINEL + json.dumps(violations))
        return SimpleNamespace(output="\n".join(out) + "\n", error=None)


class _Agent(FiguresMixin):
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.technique_agent = None
        self.revision_calls = []

    def _validate_code(self, code):
        return True, ""

    async def _notify(self, *args, **kwargs):
        return None

    def _compress_image_to_base64(self, image_path, max_width=800):
        return f"b64:{Path(image_path).stat().st_size}"

    async def _revise_figure_code(self, original_code, feedback, suggestions,
                                  must_fix, figure_type, state=None,
                                  image_base64=None, revision_history=None,
                                  layout_violations=None):
        self.revision_calls.append({
            "feedback": feedback,
            "must_fix": list(must_fix or []),
            "layout_violations": list(layout_violations or []),
        })
        return f"revised code {len(self.revision_calls)}"


def _reviews(*scripted):
    queue = list(scripted)

    async def callback(figure):
        return queue.pop(0) if queue else scripted[-1]

    return callback


def _review(approved=False, score=6, **extra):
    payload = {
        "approved": approved, "quality_score": score,
        "feedback": "needs work", "suggestions": [], "must_fix": ["fix the legend"],
        "requires_reanalysis": False,
    }
    payload.update(extra)
    return payload


async def _run(tmp_path, verdicts, reviews):
    agent = _Agent(_FakeSandbox(tmp_path, verdicts))
    state = AnalysisState(user_id="u")
    result = await agent._handle_generate_publication_figure(
        code="plot()", figure_name="f", description="d", figure_type="heatmap",
        state=state, review_callback=_reviews(*reviews),
    )
    return agent, state, result


@pytest.mark.asyncio
async def test_stalled_score_still_accepts_a_clean_figure(tmp_path):
    """The shortcut is about diminishing returns and stays available when the
    renderer has no objection — this is the pre-existing behaviour."""
    agent, state, result = await _run(
        tmp_path, [CLEAN], [_review(score=6), _review(score=6)],
    )
    figure = state.publication_figures[0]
    assert figure.review_status == FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS
    assert "did not improve" in result
    assert agent.sandbox.executions == 2  # initial + one revision


@pytest.mark.asyncio
async def test_stalled_score_cannot_ship_a_measured_collision(tmp_path):
    """A stalled score is a judgement call; covered text is not. The loop must
    keep revising instead of stamping accepted_with_reservations."""
    agent, state, _ = await _run(
        tmp_path, [BLOCKING],
        [_review(score=6), _review(score=6), _review(score=6), _review(score=6)],
    )
    figure = state.publication_figures[0]
    assert figure.review_status == FigureReviewStatus.REVISION_REQUESTED
    assert agent.sandbox.executions == 4  # initial + 3 revisions, budget spent
    assert figure.unresolved_must_fix, "unfixed defects must be recorded"


@pytest.mark.asyncio
async def test_reviewer_approval_does_not_override_the_renderer(tmp_path):
    """The reviewer judges a downscaled JPEG; the renderer reports the artists.
    While budget remains, the measurement wins."""
    agent, state, _ = await _run(
        tmp_path, [BLOCKING, BLOCKING, CLEAN],
        [_review(approved=True, score=9), _review(approved=True, score=9),
         _review(approved=True, score=9)],
    )
    figure = state.publication_figures[0]
    assert agent.revision_calls, "approval with a blocking defect must trigger a redraw"
    assert figure.review_status == FigureReviewStatus.APPROVED
    assert not figure.layout_violations


@pytest.mark.asyncio
async def test_measured_violations_reach_the_redraw_prompt(tmp_path):
    """The reason two revisions scored 6 -> 6 in production is that the redraw
    only ever received prose. It now receives coordinates."""
    agent, _, _ = await _run(
        tmp_path, [BLOCKING], [_review(score=5), _review(score=7), _review(approved=True, score=8)],
    )
    first = agent.revision_calls[0]
    assert first["layout_violations"] == BLOCKING
    assert any("Cyp7a1" in item for item in first["must_fix"]) or any(
        "Cyp7a1" in str(v) for v in first["layout_violations"]
    )


@pytest.mark.asyncio
async def test_a_worse_revision_is_rolled_back(tmp_path):
    """`curr_score <= prev_score` treated a regression as a stall and shipped
    the worse figure. A regression now restores the best rendered version."""
    agent, state, result = await _run(
        tmp_path, [CLEAN], [_review(score=6), _review(score=5)],
    )
    figure = state.publication_figures[0]
    assert figure.review_status == FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS
    assert "restored" in result
    # the first render's bytes are back on disk
    assert (tmp_path / "figures" / "f.jpg").read_bytes().endswith(b"\x01" * 100)


@pytest.mark.asyncio
async def test_a_held_back_figure_records_why_on_itself(tmp_path):
    """The figure is dropped from every report while its review status still
    reads "accepted with reservations" and the UI shows it as reviewed, so the
    reason has to travel with the figure or it exists nowhere."""
    _, state, _ = await _run(
        tmp_path, [BLOCKING], [_review(approved=True, score=9)] * 4,
    )
    figure = state.publication_figures[0]
    from backend.services.report.artifacts import is_report_eligible

    assert is_report_eligible(figure) is False
    assert "[Layout gate]" in (figure.pi_feedback or "")
    assert "held back" in figure.pi_feedback


@pytest.mark.asyncio
async def test_a_revision_with_no_verdict_does_not_clear_a_known_collision(tmp_path):
    """An unmeasured render must not read as a fixed one. Clearing the previous
    measurement would re-open both the stall shortcut and report eligibility --
    the exact path this change exists to close."""
    agent, state, _ = await _run(
        tmp_path,
        [BLOCKING, {"violations": None}, {"violations": None}, {"violations": None}],
        [_review(score=6)] * 4,
    )
    figure = state.publication_figures[0]
    assert figure.layout_violations == BLOCKING
    assert figure.review_status == FigureReviewStatus.REVISION_REQUESTED


@pytest.mark.asyncio
async def test_a_revision_that_produces_no_figure_is_not_credited(tmp_path):
    """`if len(sig): plot() else: print(...)` runs without raising and leaves the
    previous file on disk, so jpg_path.exists() and the size check both pass on
    stale bytes. The loop must not treat that as a new render."""
    agent, state, _ = await _run(
        tmp_path,
        [BLOCKING, {"no_figure": True, "violations": None}],
        [_review(score=6)] * 3,
    )
    figure = state.publication_figures[0]
    assert agent.sandbox.executions == 2, "the loop stops instead of looping on stale bytes"
    assert figure.layout_violations == BLOCKING


@pytest.mark.asyncio
async def test_a_blank_revision_restores_the_previous_render(tmp_path):
    """The wrapper saves in a finally block, so a blank revision has already
    replaced the good file before the loop can object."""
    agent, state, _ = await _run(
        tmp_path, [CLEAN, {"size": 100}], [_review(score=6), _review(score=6)],
    )
    restored = (tmp_path / "figures" / "f.jpg").read_bytes()
    assert len(restored) > 2000, "the good render is back on disk"
    assert restored.endswith(b"\x01" * 100)


@pytest.mark.asyncio
async def test_rollback_only_happens_to_a_better_version(tmp_path, monkeypatch):
    """The snapshot is skipped for a render over the size cap, so best_snapshot
    can hold a version older AND worse than the one about to be discarded.
    Restoring it would then replace a 5 with a 3 while reporting a rollback."""
    from backend.agents.bioinformatics import figures as figures_mod

    # cap chosen between the two render sizes below, so the high-scoring
    # middle render is the one that cannot be snapshotted
    monkeypatch.setattr(figures_mod, "_MAX_ROLLBACK_SNAPSHOT_BYTES", 20_000)
    agent, state, result = await _run(
        tmp_path,
        [{"size": 4_000}, {"size": 60_000}, {"size": 60_000}],
        [_review(score=3), _review(score=8), _review(score=5)],
    )
    assert "restored" not in result, "must not swap a 5 for the stale 3"

    # the ordinary case still rolls back
    agent2, state2, result2 = await _run(
        tmp_path, [CLEAN], [_review(score=6), _review(score=4)],
    )
    assert "restored" in result2


@pytest.mark.asyncio
async def test_data_level_objection_carries_layout_defects_forward(tmp_path):
    """requires_reanalysis exits before the revision loop, which used to
    discard the layout half of the same review along with it."""
    _, state, result = await _run(
        tmp_path, [BLOCKING],
        [_review(requires_reanalysis=True, feedback="bulk samples are ungrouped")],
    )
    figure = state.publication_figures[0]
    assert figure.review_status == FigureReviewStatus.REVISION_REQUESTED
    assert "Cyp7a1" in result, "the measured collision must survive the early exit"
    assert figure.unresolved_must_fix
