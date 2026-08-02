"""The PI integrated-report prompt sees only figures the report can render."""

import pytest
from PIL import Image

from backend.agents.pi.reports import ReportsMixin
from backend.models.schemas import (
    AnalysisState,
    FigureReviewStatus,
    PublicationFigure,
)


def _render(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.effect_noise((256, 256), 100).convert("RGB").save(path)
    assert path.stat().st_size > 2000
    return str(path)


def _figure(name, status, *, png_path="", figure_type="barplot",
            description="bar-plot metadata", layout_violations=None):
    return PublicationFigure(
        name=name,
        description=description,
        figure_type=figure_type,
        png_path=png_path or f"/tmp/{name}.jpg",
        pdf_path=f"/tmp/{name}.pdf",
        review_status=status,
        layout_violations=layout_violations or [],
    )


class _Harness(ReportsMixin):
    session_id = "session"

    def __init__(self):
        self.prompt = ""

    async def chat(self, prompt, temperature=None):
        self.prompt = prompt
        return "integrated report"

    def rollback_last_turn(self):  # pragma: no cover - only used on empty retries
        raise AssertionError("non-empty response should not roll back")


@pytest.mark.asyncio
async def test_integrated_report_uses_only_renderable_report_eligible_figure_metadata(tmp_path):
    state = AnalysisState(
        id="session",
        user_id="test-user",
        name="Report eligibility",
        initial_instruction="Analyze the supplied data.",
    )
    state.publication_figures = [
        _figure(
            "kept_barplot", FigureReviewStatus.APPROVED,
            png_path=_render(tmp_path / "kept_barplot.jpg"),
        ),
        _figure(
            "kept_reserved", FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS,
            png_path=_render(tmp_path / "kept_reserved.jpg"),
        ),
        _figure(
            "pending_panel", FigureReviewStatus.PENDING,
            png_path=_render(tmp_path / "pending_panel.jpg"),
        ),
        _figure(
            "rejected_panel", FigureReviewStatus.REVISION_REQUESTED,
            png_path=_render(tmp_path / "rejected_panel.jpg"),
        ),
        _figure(
            "covered_panel", FigureReviewStatus.APPROVED,
            png_path=_render(tmp_path / "covered_panel.jpg"),
            layout_violations=[{"kind": "text_over_text", "severity": "block"}],
        ),
        # Review approval alone is not enough: this restored record has neither
        # a live file nor a recoverable record preview.
        _figure(
            "approved_ghost", FigureReviewStatus.APPROVED,
            png_path=str(tmp_path / "missing.jpg"),
        ),
    ]

    harness = _Harness()
    result = await harness.generate_integrated_report(state)

    assert result == "integrated report"
    assert "kept_barplot" in harness.prompt
    assert "kept_reserved" in harness.prompt
    assert "pending_panel" not in harness.prompt
    assert "rejected_panel" not in harness.prompt
    assert "covered_panel" not in harness.prompt
    assert "approved_ghost" not in harness.prompt
    assert "declared type: barplot" in harness.prompt
    assert "bar-plot metadata" in harness.prompt
    assert "never describe a declared barplot as a UMAP" in harness.prompt
