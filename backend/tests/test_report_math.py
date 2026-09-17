"""Scientific notation survives both report converters and offline export."""

from html.parser import HTMLParser
from pathlib import Path

import pytest

from backend.agents.supervisor.report_html import ReportHtmlMixin
from backend.utils.report_html import markdown_to_html, render_report_html
from backend.utils.report_math import embed_report_math


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
def test_report_notation_survives_markdown(convert):
    source = (
        r"bulk RNA sequencing ($n = 3$ Control vs $n = 3$ DKO); "
        r"NADH/NAD$^+$; acid $\beta$-oxidation; "
        r"(*ZnT1$^{\text{flox/flox}}$ CTR1$^{\text{flox/flox}}$ Alb-Cre$^+$*) "
        r"and (*Alb-Cre$^-$*)."
        "\n\n| Estimate |\n|---|\n"
        r"| $x_i + y_j + \lvert z \rvert$ |"
        "\n\n$$\n\\frac{x_1}{y_2}\n$$"
    )
    result = convert(source)
    assert MathElements(result).formulas == [
        ("n = 3", "false"), ("n = 3", "false"), ("^+", "false"),
        (r"\beta", "false"), (r"^{\text{flox/flox}}", "false"),
        (r"^{\text{flox/flox}}", "false"), ("^+", "false"), ("^-", "false"),
        (r"x_i + y_j + \lvert z \rvert", "false"),
        ("\n\\frac{x_1}{y_2}\n", "true"),
    ]
    assert "<em>ZnT1<span" in result
    assert "<em>Alb-Cre<span" in result


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


def test_offline_renderer_is_embedded_once_only_when_needed():
    plain = "<html><body>No formulas</body></html>"
    assert embed_report_math(plain) == plain
    document = embed_report_math(plain.replace("No formulas", markdown_to_html(r"$\beta$")))
    assert document.count('id="scai-report-math"') == 1
    assert 'output: \'mathml\'' in document
    assert 'trust: false' in document
    assert '<script src=' not in document
    assert embed_report_math(document) == document
    assert document.index('id="scai-report-math"') > document.index('data-scai-math="')


def test_shared_report_template_embeds_math_renderer():
    import re
    from backend.utils.report_html import _HEAD_PRE_CSS, _HEAD_POST_CSS_BODY

    fields = re.findall(r"(?<!{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!})", _HEAD_PRE_CSS + _HEAD_POST_CSS_BODY)
    context = dict.fromkeys(fields, "")
    context["abstract"] = markdown_to_html(r"$n = 3$")
    assert 'id="scai-report-math"' in render_report_html(context)


def test_supervisor_export_embeds_math_renderer(tmp_path):
    from backend.tests.test_supervisor_report_shared import _make_supervisor
    from backend.models.schemas import AnalysisState

    state = AnalysisState(id="math-report", name="Math report", user_id="test")
    agent = _make_supervisor(state)
    path = agent._generate_html_report(
        {"title": "Math report", "appendix": {}, "_analysis_state": state},
        sections=[], abstract=r"$n = 3$", conclusions=r"$\beta$",
        report_figures=[], figure_map={}, stats={}, language="en",
        output_path=tmp_path, workspace_path=None,
    )
    document = Path(path).read_text()
    assert MathElements(document).formulas == [("n = 3", "false"), (r"\beta", "false")]
    assert document.count('id="scai-report-math"') == 1


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
