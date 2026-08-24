"""X1 tests for scalar-or-list review instruction boundaries."""

import importlib
import json

import pytest

from backend.agents.bioinformatics.diagrams import DiagramsMixin


def test_normalize_text_list_bounds_and_deduplicates_scalar_or_list():
    formatting = importlib.import_module("backend.utils.formatting")
    normalize = getattr(formatting, "normalize_text_list", None)
    assert callable(normalize), "shared review-instruction normalizer is missing"

    assert normalize("  fix the legend  ", limit=3) == ["fix the legend"]
    assert normalize([" A ", "A", " B ", "C"], limit=2) == ["A", "B"]
    assert normalize(["abcdefgh"], limit=1, max_chars=4) == ["abcd"]
    assert len(normalize("z" * 5000)[0]) == 1000


class _DiagramReviewer(DiagramsMixin):
    async def chat_without_history(self, *_args, **_kwargs):
        return json.dumps({
            "quality_score": 4,
            "feedback": "labels need correction",
            "strengths": [],
            "issues": [],
            "suggestions": [],
            "must_fix": "correct the missing label" + ("y" * 5000),
        })

    def _safe_parse_json(self, value, default=None, **_kwargs):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default


@pytest.mark.asyncio
async def test_current_figure_and_diagram_review_boundaries_normalize_scalar_must_fix_once(
    tmp_path,
):
    from backend.tests.test_figure_revision_gate import _run

    _, state, _ = await _run(
        tmp_path,
        [[]],
        [{
            "approved": False,
            "quality_score": 5,
            "feedback": "data needs reanalysis",
            "suggestions": [],
            "must_fix": "recheck the grouped comparison" + ("x" * 5000),
            "requires_reanalysis": True,
        }],
    )
    figure_must_fix = state.publication_figures[0].unresolved_must_fix
    assert len(figure_must_fix) == 1
    assert figure_must_fix[0] == (
        "recheck the grouped comparison" + ("x" * 5000)
    )[:1000]

    diagram_review = await _DiagramReviewer().review_text_diagram(
        state=None,
        text_diagram="A -> B",
        revision_history=[],
    )
    assert len(diagram_review["must_fix"]) == 1
    assert diagram_review["must_fix"][0] == (
        "correct the missing label" + ("y" * 5000)
    )[:1000]
