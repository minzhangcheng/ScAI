"""The figure revision loop's exit conditions.

The loop is where a correctly-diagnosed figure gets shipped anyway. Two exits
did that: a "quality score stalled" rule that accepted the current version
whenever the score failed to rise (using ``<=``, so a score that *fell* also
counted as stalled), and reviewer approval, which nothing cross-checked. These
tests pin the repaired behaviour: a measured collision is not something either
exit may override, and a regression is rolled back rather than shipped.
"""
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
from backend.services.figure_layout import LAYOUT_LINT_SENTINEL
from backend.services.report.artifacts import (
    eligible_artifacts,
    image_file_sha256,
    pdf_file_sha256,
    preview_render_sha256,
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
        "feedback": "needs work", "suggestions": [],
        "must_fix": [] if approved else ["fix the legend"],
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
async def test_a_blank_revision_restores_the_previous_render(tmp_path):
    """The wrapper saves in a finally block, so a blank revision has already
    replaced the good file before the loop can object."""
    agent, state, _ = await _run(
        tmp_path, [CLEAN, {"size": 100}], [_review(score=6), _review(score=6)],
    )
    restored = (tmp_path / "figures" / "f.jpg").read_bytes()
    assert len(restored) > 2000, "the good render is back on disk"
    assert restored.endswith(b"\x01" * 100)


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
