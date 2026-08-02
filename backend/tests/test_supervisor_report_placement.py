"""Supervisor report generation wiring for round-bound figure placement."""
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from backend.agents.supervisor_agent import SupervisorAgent
from backend.models.schemas import AnalysisState, FigureReviewStatus, PublicationFigure
from backend.services.report.report_outline import build_report_outline

_T0 = datetime(2026, 1, 1, 12, 0, 0)


def _round(n, phase, direction=""):
    return SimpleNamespace(
        round_number=n, phase=phase, direction=direction,
        timestamp=_T0 + timedelta(minutes=n * 10),
    )


def _outline():
    return build_report_outline(SimpleNamespace(
        round_results=[
            _round(1, "exploration"),
            _round(2, "innovation", "direction A"),
            _round(3, "integration"),
        ],
        publication_figures=[],
    ))


def _agent():
    return SupervisorAgent.__new__(SupervisorAgent)


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    return path


def test_allocation_text_lists_figures_under_their_own_section(tmp_path):
    outline = _outline()
    figure_map = {"a.png": tmp_path / "a.png", "b.png": tmp_path / "b.png"}
    text = _agent()._build_figure_allocation_text(
        outline, figure_map, {"a.png": 1, "b.png": 2}
    )
    assert "[exploration_findings]" in text
    # Each figure appears under exactly the section owning its round.
    expl = text.index("[exploration_findings]")
    d1 = text.index("[direction_1]")
    assert expl < text.index("a.png") < d1 < text.index("b.png")


def test_allocation_marks_figureless_sections_explicitly(tmp_path):
    text = _agent()._build_figure_allocation_text(
        _outline(), {"a.png": tmp_path / "a.png"}, {"a.png": 1}
    )
    for sid in ("abstract", "methods", "conclusions"):
        line = next(l for l in text.splitlines() if l.startswith(f"[{sid}]"))
        assert "no figures" in line

    integration = next(s for s in _outline() if s.kind == "integration_final")
    line = next(l for l in text.splitlines() if l.startswith(f"[{integration.id}]"))
    assert "text-only" in line
    assert "no figure required" in line
    assert "no figures were produced" not in line


def test_allocation_flags_unattributed_figures(tmp_path):
    text = _agent()._build_figure_allocation_text(
        _outline(), {"mystery.png": tmp_path / "mystery.png"}, {}
    )
    # A figure the round cannot place is still offered to the model, told to
    # place it by content -- never silently dropped from the allocation.
    assert "mystery.png" in text
    assert "by content" in text


def test_a_round_several_sections_claim_allocates_to_none_of_them(tmp_path):
    """The legacy outline organises by analysis type, not by round.

    Its four findings sections each declare every round, so the round number
    carries no allocation. Handing the figures to the first claimant put the
    whole report in quality control and told the model the other three chapters
    produced nothing -- a statement it then wrote prose around.
    """
    outline = build_report_outline(SimpleNamespace(
        round_results=[_round(1, "exploration"), _round(2, "exploration")],
        publication_figures=[],
    ))
    shared = [sec.id for sec in outline if sec.rounds]
    assert len(shared) > 1, "expected the legacy outline's shared-round sections"

    text = _agent()._build_figure_allocation_text(
        outline, {"umap.png": tmp_path / "umap.png"}, {"umap.png": 1}
    )
    assert "no figures were produced" not in text
    assert "by content" in text
    # Offered once, to no single section.
    assert text.count("umap.png") == 1


def test_parse_uses_outline_titles_and_types():
    outline = _outline()
    agent = _agent()
    agent.session_id = "s"
    content = (
        "## [direction_1] Section Title\nbody one\n\n"
        "## [methods] Section Title\nbody two\n"
    )
    sections = agent._parse_report_sections(content, "en", outline=outline)
    by_id = {s["id"]: s for s in sections}
    # Placeholder titles are repaired from the outline, not the static table.
    assert by_id["direction_1"]["title"] == "Innovation Round 2"
    assert by_id["direction_1"]["analysis_type"] == "specific"
    assert by_id["methods"]["title"] == "Methods"
    assert by_id["methods"]["analysis_type"] == "general"


def test_parse_without_outline_keeps_legacy_behaviour():
    agent = _agent()
    agent.session_id = "s"
    sections = agent._parse_report_sections(
        "## [data_qc] Section Title\nbody\n", "en"
    )
    assert sections[0]["title"] == "Data Overview and Quality Control"


def test_html_embeds_a_repeated_figure_only_once(tmp_path):
    """A filename cited three times used to be inlined three times."""
    agent = _agent()
    agent.session_id = "s"
    fig = _png(tmp_path / "workspace" / "figures" / "dup.png")
    sections = [{
        "id": "direction_1",
        "title": "Direction",
        "analysis_type": "specific",
        "content": (
            "[FIGURE: dup.png]\n[FIGURE_CAPTION: CAPTIONALPHA]\n"
            "text\n[FIGURE: dup.png]\n[FIGURE_CAPTION: CAPTIONBETA]\n"
            "more\n[FIGURE: dup.png]\n"
        ),
        "figures": [], "tables": [],
    }]
    out = tmp_path / "reports"
    out.mkdir(parents=True, exist_ok=True)

    html_path = agent._generate_html_report(
        {"title": "T", "figures": [], "tables": []},
        sections, "abstract", "conclusions",
        [{"filename": "dup.png", "path": str(fig), "category": "other"}],
        {"dup.png": fig},
        {"total_steps_monitored": 0, "steps_with_technical_issues": 0,
         "pi_escalations": 0},
        "en", out, tmp_path / "workspace",
    )
    html = Path(html_path).read_text(encoding="utf-8")
    # One <img> for the figure, and the surviving caption is the first one.
    assert html.count('alt="dup.png"') == 1
    assert html.count("CAPTIONALPHA") == 1
    assert "CAPTIONBETA" not in html


def test_html_matches_registered_metadata_by_stem_when_record_path_is_empty(tmp_path):
    agent = _agent()
    agent.session_id = "s"
    fig = _png(tmp_path / "workspace" / "figures" / "planned_result.png")
    state = AnalysisState(id="s", user_id="u", name="T")
    state.publication_figures = [PublicationFigure(
        name="planned_result",
        description="Authoritative planned-result description.",
        figure_type="heatmap",
        png_path="",
        pdf_path="",
        review_status=FigureReviewStatus.APPROVED,
    )]
    sections = [{
        "id": "direction_1",
        "title": "Direction",
        "analysis_type": "specific",
        "content": "[FIGURE: planned_result.png]",
        "figures": [],
        "tables": [],
    }]
    out = tmp_path / "reports"
    out.mkdir(parents=True, exist_ok=True)

    html_path = agent._generate_html_report(
        {"title": "T", "figures": [], "tables": [], "_analysis_state": state},
        sections,
        "abstract",
        "conclusions",
        [{"filename": "planned_result.png", "path": str(fig), "category": "heatmap"}],
        {"planned_result.png": fig},
        {"total_steps_monitored": 0, "steps_with_technical_issues": 0,
         "pi_escalations": 0},
        "en",
        out,
        tmp_path / "workspace",
    )

    html = Path(html_path).read_text(encoding="utf-8")
    assert "Authoritative planned-result description." in html


def test_html_does_not_guess_same_stem_pdf_for_download_or_package(tmp_path):
    agent = _agent()
    agent.session_id = "s"
    fig = _png(tmp_path / "workspace" / "figures" / "result.png")
    sibling_pdf = fig.with_suffix(".pdf")
    sibling_pdf.write_bytes(b"%PDF-1.4\nunreviewed replacement")
    out = tmp_path / "reports"
    out.mkdir(parents=True)
    sections = [{
        "id": "results",
        "title": "Results",
        "analysis_type": "specific",
        "content": "[FIGURE: result.png]",
        "figures": [],
        "tables": [],
    }]

    html_path = agent._generate_html_report(
        {"title": "T", "figures": [], "tables": []},
        sections,
        "abstract",
        "conclusions",
        [{"filename": "result.png", "path": str(fig), "category": "other"}],
        {"result.png": fig},
        {"total_steps_monitored": 0, "steps_with_technical_issues": 0,
         "pi_escalations": 0},
        "en",
        out,
        tmp_path / "workspace",
    )

    html = Path(html_path).read_text(encoding="utf-8")
    assert "Download PDF" not in html
    assert not (out / "figures" / sibling_pdf.name).exists()


def test_html_uses_only_validated_report_figure_pdf_path(tmp_path):
    agent = _agent()
    agent.session_id = "s"
    fig = _png(tmp_path / "workspace" / "figures" / "result.png")
    validated_pdf = tmp_path / "validated" / "reviewed-result.pdf"
    validated_pdf.parent.mkdir()
    validated_pdf.write_bytes(b"%PDF-1.4\nvalidated")
    out = tmp_path / "reports"
    out.mkdir(parents=True)
    sections = [{
        "id": "results",
        "title": "Results",
        "analysis_type": "specific",
        "content": "[FIGURE: result.png]",
        "figures": [],
        "tables": [],
    }]

    html_path = agent._generate_html_report(
        {"title": "T", "figures": [], "tables": []},
        sections,
        "abstract",
        "conclusions",
        [{
            "filename": "result.png",
            "path": str(fig),
            "pdf_path": str(validated_pdf),
            "category": "other",
        }],
        {"result.png": fig},
        {"total_steps_monitored": 0, "steps_with_technical_issues": 0,
         "pi_escalations": 0},
        "en",
        out,
        tmp_path / "workspace",
    )

    html = Path(html_path).read_text(encoding="utf-8")
    assert 'figures/reviewed-result.pdf' in html
    assert (out / "figures" / "reviewed-result.pdf").read_bytes() == validated_pdf.read_bytes()
