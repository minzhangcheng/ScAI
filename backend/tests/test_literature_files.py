from backend.services.report.literature_files import (
    build_literature_report_html,
    write_literature_report,
)


class _FakeReport:
    """Mimics models.schemas.LiteratureReport for tests (topic/report/references)."""
    def __init__(self, topic, report, references=None):
        self.topic = topic
        self.report = report
        self.references = references or []


def _sample():
    return [
        _FakeReport(
            topic="EMT in fibrosis",
            report="## Findings\nEMT is **upregulated** in fibrotic tissue.",
            references=[
                {"title": "EMT review", "url": "https://example.com/emt"},
                {"title": "Fibrosis 2020"},
            ],
        ),
        _FakeReport(topic="Macrophage polarization", report="M1/M2 balance shifts."),
    ]


def test_build_empty_returns_empty():
    assert build_literature_report_html([]) == ""


def test_build_contains_topics_body_and_refs():
    html = build_literature_report_html(_sample())
    assert "EMT in fibrosis" in html
    assert "Macrophage polarization" in html
    # markdown converted to HTML (bold -> <strong>)
    assert "<strong>upregulated</strong>" in html
    # references rendered
    assert "EMT review" in html
    assert 'href="https://example.com/emt"' in html
    # self-contained document
    assert "<html" in html.lower() and "</html>" in html.lower()


def test_build_escapes_topic():
    html = build_literature_report_html([_FakeReport("<x>&", "body")])
    assert "<x>&" not in html
    assert "&lt;x&gt;" in html


def test_write_empty_returns_none(tmp_path):
    assert write_literature_report([], tmp_path) is None


def test_write_creates_file(tmp_path):
    path = write_literature_report(_sample(), tmp_path)
    assert path is not None
    assert path.name == "literature_report.html"
    assert path.parent == tmp_path
    content = path.read_text(encoding="utf-8")
    assert "EMT in fibrosis" in content
