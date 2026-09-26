"""Supervisor report generation wiring for round-bound figure placement."""
from pathlib import Path

import pytest

from backend.agents.supervisor_agent import SupervisorAgent


def _agent():
    return SupervisorAgent.__new__(SupervisorAgent)


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    return path


@pytest.mark.parametrize("validated", [False, True], ids=["guessed-pdf", "validated-pdf"])
def test_html_packages_only_explicitly_validated_figure_pdf(tmp_path, validated):
    agent = _agent()
    agent.session_id = "s"
    fig = _png(tmp_path / "workspace" / "figures" / "result.png")
    pdf = tmp_path / "validated" / "reviewed-result.pdf" if validated else fig.with_suffix(".pdf")
    pdf.parent.mkdir(exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4\nvalidated" if validated else b"%PDF-1.4\nunreviewed replacement")
    report_figure = {"filename": "result.png", "path": str(fig), "category": "other"}
    if validated:
        report_figure["pdf_path"] = str(pdf)
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
        [report_figure],
        {"result.png": fig},
        {"total_steps_monitored": 0, "steps_with_technical_issues": 0,
         "pi_escalations": 0},
        "en",
        out,
        tmp_path / "workspace",
    )

    html = Path(html_path).read_text(encoding="utf-8")
    if validated:
        assert 'figures/reviewed-result.pdf' in html
        assert (out / "figures" / pdf.name).read_bytes() == pdf.read_bytes()
    else:
        assert "Download PDF" not in html
        assert not (out / "figures" / pdf.name).exists()
