"""The figure revision loop's exit conditions.

The loop is where a correctly-diagnosed figure gets shipped anyway. Two exits
did that: a "quality score stalled" rule that accepted the current version
whenever the score failed to rise (using ``<=``, so a score that *fell* also
counted as stalled), and reviewer approval, which nothing cross-checked. These
tests pin the repaired behaviour: a measured collision is not something either
exit may override, and a regression is rolled back rather than shipped.
"""
import asyncio
import json
import base64
import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.bioinformatics.figures import FiguresMixin
from backend.models.schemas import AnalysisState, FigureReviewStatus
from backend.models.schemas import PublicationFigure
from backend.services.figure_layout import LAYOUT_LINT_SENTINEL, blocking_violations
from backend.services.report.artifacts import (
    eligible_artifacts,
    image_file_sha256,
    pdf_file_sha256,
    preview_render_sha256,
    publication_pdf_is_valid,
)
from backend.tests.figure_image_helpers import valid_image_bytes, valid_pdf_bytes

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
        jpg_match = re.search(r"^\s*_jpg_path = (.+)$", code, re.MULTILINE)
        pdf_match = re.search(r"^\s*_pdf_path = (.+)$", code, re.MULTILINE)
        jpg_path = Path(ast.literal_eval(jpg_match.group(1)))
        pdf_path = Path(ast.literal_eval(pdf_match.group(1)))
        jpg_path.parent.mkdir(parents=True, exist_ok=True)
        out = []
        if step.get("no_figure"):
            # the wrapper's own wording when the code left nothing to save;
            # the PREVIOUS file stays on disk untouched
            out.append("[Auto-save] No figures found — skipping save")
        else:
            size = int(step.get("size", 4000))
            if step.get("corrupt_image") or size < 2000:
                jpg = b"not-an-image"[:size].ljust(size, b"x")
            else:
                # Real, decodable JPEG plus a distinct trailer so a rollback is
                # observable without relying on fake magic bytes.
                jpg = valid_image_bytes(
                    "JPEG",
                    color=(self.executions, 34, 56),
                    min_bytes=size,
                    trailing=bytes([self.executions]) * 100,
                )
            jpg_path.write_bytes(jpg)
            if step.get("corrupt_pdf"):
                pdf_path.write_bytes(b"%PDF-corrupt" + b"x" * 1000)
            elif not step.get("missing_pdf"):
                pdf_path.write_bytes(
                    valid_pdf_bytes(color=(self.executions, 34, 56))
                )
            out.append("Saved JPG")
            if step.get("missing_pdf"):
                out.append("[Auto-save] Warning: could not save PDF: test failure")
            else:
                out.append("Saved PDF: f.pdf")
        violations = step.get("violations", CLEAN)
        if violations is not None:
            out.append(LAYOUT_LINT_SENTINEL + json.dumps(violations))
        return SimpleNamespace(output="\n".join(out) + "\n", error=None)


class _TransportFailureSandbox:
    def __init__(self, working_dir, failure):
        self.working_dir = str(working_dir)
        self.failure = failure
        self.executions = 0

    async def execute(self, _code, **_kwargs):
        self.executions += 1
        raise self.failure


class _SetupFailureSandbox:
    def __init__(self, working_dir, error):
        self.working_dir = str(working_dir)
        self.error = error
        self.executions = 0

    async def execute(self, _code, **_kwargs):
        self.executions += 1
        return SimpleNamespace(output="", error=self.error)


class _Agent(FiguresMixin):
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.technique_agent = None
        self.revision_calls = []
        self.notifications = []

    def _validate_code(self, code):
        return True, ""

    async def _notify(self, message_type, content):
        self.notifications.append((message_type, content))

    def _compress_image_to_base64(self, image_path, max_width=800):
        return base64.b64encode(Path(image_path).read_bytes()).decode()

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


def test_upsert_replaces_only_within_the_same_lineage(tmp_path):
    agent = _Agent(_FakeSandbox(tmp_path, [CLEAN]))
    state = AnalysisState(user_id="u")
    main = PublicationFigure(
        id="main-id", name="same", description="main", figure_type="other",
        png_path="", pdf_path="", lineage_id="main",
    )
    branch = PublicationFigure(
        id="branch-id", name="same", description="branch", figure_type="other",
        png_path="", pdf_path="", lineage_id="branch-a",
    )
    state.publication_figures = [main]

    agent._upsert_publication_figure(state, branch)

    assert [(item.lineage_id, item.id) for item in state.publication_figures] == [
        ("main", "main-id"),
        ("branch-a", "branch-id"),
    ]


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
async def test_pending_record_precedes_review_and_final_event_follows_layout_gate(tmp_path):
    agent = _Agent(_FakeSandbox(tmp_path, [BLOCKING, CLEAN]))
    state = AnalysisState(user_id="u")
    callback_observations = []
    reviews = [_review(approved=True, score=9), _review(approved=True, score=9)]

    async def callback(figure):
        stored = next((item for item in state.publication_figures if item.id == figure.id), None)
        callback_observations.append(stored.review_status if stored else None)
        return reviews.pop(0)

    await agent._handle_generate_publication_figure(
        code="plot()",
        figure_name="f",
        description="d",
        figure_type="heatmap",
        state=state,
        review_callback=callback,
    )

    assert callback_observations == [
        FigureReviewStatus.PENDING,
        FigureReviewStatus.PENDING,
    ]
    lifecycle = [
        (kind, payload["figure"]["review_status"])
        for kind, payload in agent.notifications
        if kind in {"figure_approved", "figure_revision"}
    ]
    # The first reviewer approval was overruled by a measured collision.  The
    # producer reports revision, then reports the clean replacement as approved.
    assert lifecycle == [
        ("figure_revision", "revision_requested"),
        ("figure_approved", "approved"),
    ]


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
async def test_an_unmeasured_first_render_is_not_recorded_as_clean(tmp_path):
    """The first render has no earlier measurement to keep, so it must record
    the absence of one.

    A platform diagnostic is produced with no review callback and never enters
    the loop, and ``is_report_eligible`` waives the review-status check for it --
    the layout gate is the only question it is asked. Storing ``[]``, which is
    exactly what a measured-clean figure stores, makes the gap invisible to every
    consumer that reads the record.
    """
    agent = _Agent(_FakeSandbox(tmp_path, [{"violations": None}]))
    state = AnalysisState(user_id="u")
    await agent._handle_generate_publication_figure(
        code="plot()", figure_name="f", description="d", figure_type="qc",
        state=state, review_callback=None, diagnostic=True,
    )
    figure = state.publication_figures[0]
    assert figure.layout_violations, "unverified must not be stored as measured-clean"
    assert [v["kind"] for v in figure.layout_violations] == ["lint_skipped"]
    # Advisory, not blocking: nothing was measured to be wrong, and a redraw
    # cannot fix a linter that did not run. A blocking record here would spend
    # the revision budget on an unfixable objection and drop from every report
    # the one figure the QC chapter has.
    assert blocking_violations(figure.layout_violations) == []

    # ... and a render the linter did measure still records nothing, so the two
    # remain distinguishable.
    clean_agent = _Agent(_FakeSandbox(tmp_path, [CLEAN]))
    clean_state = AnalysisState(user_id="u")
    await clean_agent._handle_generate_publication_figure(
        code="plot()", figure_name="f", description="d", figure_type="qc",
        state=clean_state, review_callback=None, diagnostic=True,
    )
    assert clean_state.publication_figures[0].layout_violations == []


@pytest.mark.asyncio
async def test_a_malformed_lint_verdict_is_recorded_as_unverified(tmp_path):
    agent = _Agent(_FakeSandbox(tmp_path, [{"violations": {"not": "a list"}}]))
    state = AnalysisState(user_id="u")

    await agent._handle_generate_publication_figure(
        code="plot()", figure_name="malformed", description="d",
        figure_type="qc", state=state, review_callback=None, diagnostic=True,
    )

    figure = state.publication_figures[0]
    assert [item["kind"] for item in figure.layout_violations] == ["lint_skipped"]
    assert eligible_artifacts([figure]) == []


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
async def test_an_initial_no_figure_execution_cannot_register_stale_same_name_bytes(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "f.jpg").write_bytes(b"old" * 2000)
    (figures / "f.pdf").write_bytes(b"old" * 1000)
    agent = _Agent(_FakeSandbox(tmp_path, [{"no_figure": True, "violations": None}]))
    state = AnalysisState(user_id="u")

    result = await agent._handle_generate_publication_figure(
        code="print('no result to draw')",
        figure_name="f",
        description="new round result",
        figure_type="heatmap",
        state=state,
        review_callback=_reviews(_review(approved=True, score=9)),
        plan_step_id=3,
    )

    assert "no new JPG render" in result
    assert state.publication_figures == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"corrupt_image": True, "size": 4000}, "cannot be decoded"),
        ({"missing_pdf": True}, "did not produce a new publication PDF"),
        ({"corrupt_pdf": True}, "Publication PDF is missing or invalid"),
    ],
)
async def test_initial_render_requires_decodable_jpg_and_valid_pdf(
    tmp_path, step, message
):
    agent = _Agent(_FakeSandbox(tmp_path, [step]))
    state = AnalysisState(user_id="u")

    result = await agent._handle_generate_publication_figure(
        code="plot()", figure_name="f", description="d", figure_type="heatmap",
        state=state, review_callback=None,
    )

    assert message in result
    assert state.publication_figures == []


@pytest.mark.asyncio
async def test_generated_record_carries_render_and_preview_hashes(tmp_path):
    agent = _Agent(_FakeSandbox(tmp_path, [CLEAN]))
    state = AnalysisState(user_id="u")

    await agent._handle_generate_publication_figure(
        code="plot()", figure_name="f", description="d", figure_type="heatmap",
        state=state, review_callback=None,
    )

    figure = state.publication_figures[0]
    assert len(figure.render_sha256 or "") == 64
    assert len(figure.preview_sha256 or "") == 64


@pytest.mark.asyncio
async def test_generated_record_inherits_current_plan_id(tmp_path):
    agent = _Agent(_FakeSandbox(tmp_path, [CLEAN]))
    state = AnalysisState(
        user_id="u",
        current_plan={"id": "approved-plan", "phase": "innovation"},
    )

    await agent._handle_generate_publication_figure(
        code="plot()", figure_name="f", description="d", figure_type="heatmap",
        state=state, review_callback=None, plan_step_id=7,
    )

    figure = state.publication_figures[0]
    assert figure.plan_id == "approved-plan"
    assert figure.plan_step_id == 7


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
async def test_a_revision_missing_its_pdf_restores_the_previous_pair(tmp_path):
    agent, state, _ = await _run(
        tmp_path, [CLEAN, {"missing_pdf": True}],
        [_review(score=6), _review(score=7)],
    )

    restored = (tmp_path / "figures" / "f.jpg").read_bytes()
    assert restored.endswith(b"\x01" * 100)
    assert publication_pdf_is_valid(tmp_path / "figures" / "f.pdf")
    assert state.publication_figures[0].render_sha256


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


@pytest.mark.asyncio
async def test_scalar_must_fix_stays_one_item_when_reanalysis_is_deferred(tmp_path):
    """A scalar review instruction must not be persisted one character at a time."""
    _, state, _ = await _run(
        tmp_path,
        [CLEAN],
        [_review(
            requires_reanalysis=True,
            must_fix="recheck the grouped comparison",
        )],
    )

    figure = state.publication_figures[0]
    assert figure.unresolved_must_fix == ["recheck the grouped comparison"]


def _approved_prior_figure(tmp_path: Path) -> tuple[PublicationFigure, bytes]:
    figures = tmp_path / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    jpg = figures / "f.jpg"
    pdf = figures / "f.pdf"
    old_bytes = valid_image_bytes("JPEG", color=(90, 20, 20), min_bytes=4000)
    jpg.write_bytes(old_bytes)
    pdf.write_bytes(valid_pdf_bytes(color=(90, 20, 20)))
    preview = base64.b64encode(old_bytes).decode()
    figure = PublicationFigure(
        id="prior-head",
        name="f",
        description="Prior accepted render",
        figure_type="heatmap",
        png_path=str(jpg),
        pdf_path=str(pdf),
        png_base64=preview,
        render_sha256=image_file_sha256(jpg),
        preview_sha256=preview_render_sha256(preview),
        pdf_sha256=pdf_file_sha256(pdf),
        review_status=FigureReviewStatus.APPROVED,
        logical_figure_id="logical-result",
        is_published_head=True,
        layout_violations=[],
    )
    return figure, old_bytes


@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.asyncio
async def test_dispatched_figure_transport_failure_invalidates_expression_artifacts(
    tmp_path, failure_type
):
    prior, _old_bytes = _approved_prior_figure(tmp_path)
    prior.round_number = 1
    sandbox = _TransportFailureSandbox(
        tmp_path, failure_type("synthetic figure transport failure")
    )
    agent = _Agent(sandbox)
    state = AnalysisState(
        user_id="u", current_round=1, publication_figures=[prior]
    )

    with pytest.raises(failure_type, match="synthetic figure transport failure"):
        await agent._handle_generate_publication_figure(
            code="sc.pp.log1p(adata)\nplot()",
            figure_name="candidate",
            description="mutating candidate",
            figure_type="heatmap",
            state=state,
        )

    assert sandbox.executions == 1
    assert prior.is_current is False
    assert state.sandbox_state["analysis_generation"] == 1
    assert not (tmp_path / "figures" / ".attempts").exists()


@pytest.mark.asyncio
async def test_figure_runtime_setup_failure_is_compacted_without_invalidation(tmp_path):
    prior, _old_bytes = _approved_prior_figure(tmp_path)
    prior.round_number = 1
    private_path = "/home/synthetic-user/private-project/analysis.py"
    synthetic_secret = "synthetic-fixture-credential-value"
    frames = "\n".join(
        f'  File "{private_path}", line {index}, in helper'
        for index in range(120)
    )
    runtime_error = (
        "\x1b[31mTraceback (most recent call last):\x1b[0m\n"
        f"api_key='{synthetic_secret}'\n{frames}\n"
        "TypeError: synthetic runtime setup failure"
    )
    sandbox = _SetupFailureSandbox(tmp_path, runtime_error)
    agent = _Agent(sandbox)
    state = AnalysisState(
        user_id="u", current_round=1, publication_figures=[prior]
    )

    result = await agent._handle_generate_publication_figure(
        code=(
            "sc.pp.log1p(adata)\n"
            "sc.tl.score_genes(adata, gene_list=['A'])\n"
            "plot()"
        ),
        figure_name="candidate",
        description="blocked candidate",
        figure_type="heatmap",
        state=state,
    )

    assert sandbox.executions == 1
    assert "TypeError: synthetic runtime setup failure" in result
    assert len(result) <= 3100
    assert "\x1b" not in result
    assert synthetic_secret not in result
    assert "synthetic-user" not in result
    assert prior.is_current is True
    assert "analysis_generation" not in state.sandbox_state


@pytest.mark.asyncio
async def test_failed_same_name_attempt_never_touches_the_published_pair(tmp_path):
    prior, old_bytes = _approved_prior_figure(tmp_path)
    agent = _Agent(_FakeSandbox(tmp_path, [{"no_figure": True, "violations": None}]))
    state = AnalysisState(user_id="u", publication_figures=[prior])

    result = await agent._handle_generate_publication_figure(
        code="plot()",
        figure_name="f",
        logical_figure_id="logical-result",
        description="failed redraw",
        figure_type="heatmap",
        state=state,
        review_callback=_reviews(_review(approved=True, score=9)),
    )

    assert "no new JPG render" in result
    assert (tmp_path / "figures" / "f.jpg").read_bytes() == old_bytes
    assert state.publication_figures == [prior]
    assert not (tmp_path / "figures" / ".attempts").exists()


@pytest.mark.asyncio
async def test_reviewer_crash_abandons_staging_and_preserves_prior_head(tmp_path):
    prior, old_bytes = _approved_prior_figure(tmp_path)
    agent = _Agent(_FakeSandbox(tmp_path, [CLEAN]))
    state = AnalysisState(user_id="u", publication_figures=[prior])

    async def crashing_reviewer(_figure):
        raise RuntimeError("review transport failed")

    with pytest.raises(RuntimeError, match="review transport failed"):
        await agent._handle_generate_publication_figure(
            code="plot()",
            figure_name="f",
            logical_figure_id="logical-result",
            description="candidate redraw",
            figure_type="heatmap",
            state=state,
            review_callback=crashing_reviewer,
        )

    assert (tmp_path / "figures" / "f.jpg").read_bytes() == old_bytes
    assert prior.is_published_head is True
    assert eligible_artifacts(state.publication_figures) == [prior]
    assert not (tmp_path / "figures" / ".attempts").exists()


@pytest.mark.asyncio
async def test_rejected_redraw_retains_the_prior_reportable_head(tmp_path):
    prior, old_bytes = _approved_prior_figure(tmp_path)
    agent = _Agent(_FakeSandbox(tmp_path, [CLEAN]))
    state = AnalysisState(user_id="u", publication_figures=[prior])

    await agent._handle_generate_publication_figure(
        code="plot()",
        figure_name="f",
        logical_figure_id="logical-result",
        description="candidate redraw",
        figure_type="heatmap",
        state=state,
        review_callback=_reviews(_review(requires_reanalysis=True)),
    )

    candidate = state.publication_figures[-1]
    assert candidate.revision_of_id == prior.id
    assert candidate.is_published_head is False
    assert prior.is_published_head is True
    assert eligible_artifacts(state.publication_figures) == [prior]
    assert Path(prior.png_path).read_bytes() == old_bytes
    assert ".revisions" in Path(prior.png_path).parts


@pytest.mark.asyncio
async def test_approved_redraw_advances_explicit_logical_head(tmp_path):
    prior, old_bytes = _approved_prior_figure(tmp_path)
    agent = _Agent(_FakeSandbox(tmp_path, [CLEAN]))
    state = AnalysisState(user_id="u", publication_figures=[prior])

    await agent._handle_generate_publication_figure(
        code="plot()",
        figure_name="f",
        logical_figure_id="logical-result",
        description="accepted redraw",
        figure_type="heatmap",
        state=state,
        review_callback=_reviews(_review(approved=True, score=9)),
    )

    head = state.publication_figures[-1]
    assert head.revision_of_id == prior.id
    assert head.supersedes_id == prior.id
    assert head.is_published_head is True
    assert prior.is_published_head is False
    assert eligible_artifacts(state.publication_figures) == [head]
    assert Path(prior.png_path).read_bytes() == old_bytes
