"""Scientific notation survives both report converters and offline export."""

from html.parser import HTMLParser
from pathlib import Path

import pytest

from backend.agents.supervisor.report_html import ReportHtmlMixin
from backend.utils.report_html import markdown_to_html


class MathElements(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.formulas = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-scai-math" in attrs:
            self.formulas.append((attrs["data-scai-math"], attrs["data-display"]))


@pytest.mark.parametrize("convert", [markdown_to_html, ReportHtmlMixin._markdown_to_html])
def test_code_escaped_dollars_and_unsafe_tex_remain_inert(convert):
    source = (
        '`$x$`\n\n```python\nvalue = "$y$"\n```\n\n'
        r"Price \$5 and \$10. Incomplete $formula. "
        '\n\n$\\text{<script>alert("x")</script>}$'
    )
    result = convert(source)
    assert MathElements(result).formulas == [
        ('\\text{<script>alert("x")</script>}', "false")
    ]
    assert '<script>' not in result
    assert '<code>$x$</code>' in result
    assert '$y$' in result


@pytest.mark.asyncio
async def test_generation_and_regeneration_preserve_tex_groups(tmp_path):
    import json
    from backend.services.report.generator import DetailedReportGenerator
    from backend.services.report.regenerator import generate_html_report

    tex = r"\begin{matrix}x_1 & y_2\\x_2 & y_1\end{matrix}"
    report = {"title": "Math", "abstract": f"$${tex}$$", "sections": []}
    (tmp_path / "report.json").write_text(json.dumps(report))
    regenerated = generate_html_report(tmp_path)
    generator = DetailedReportGenerator(object(), "pi", "bio", str(tmp_path / "generated"))
    generated = await generator._generate_html_report(report, "en")
    for path in [regenerated, generated]:
        document = Path(path).read_text()
        assert MathElements(document).formulas == [(tex, "true")]
        assert document.count('id="scai-report-math"') == 1
