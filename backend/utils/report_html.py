"""
Shared HTML utilities for report generation.
Exports HTML_TEMPLATE, render_report_html, markdown_to_html, and figure helpers.
"""
import html as html_module
import re
from pathlib import Path
from typing import Dict, Optional

from .report_css import REPORT_CSS


def _build_figure_description_map(analysis_log) -> Dict[str, str]:
    """Build a mapping from figure filename stem to code_description from analysis_log.

    Provides rich, contextual captions for figures instead of just prettified filenames.
    """
    desc_map: Dict[str, str] = {}
    for step in analysis_log:
        if step.saved_figure_filenames and step.code_description:
            for fname in step.saved_figure_filenames:
                stem = Path(fname).stem
                if stem not in desc_map:
                    desc_map[stem] = step.code_description
    return desc_map


def _make_figure_block(src: str, caption: str, css_class: str = "step-figure") -> str:
    """Return an HTML figure block with image and caption."""
    return (
        f'<figure class="{css_class}">'
        f'<img src="{src}" alt="{caption}"/>'
        f'<figcaption class="{css_class}-caption">{caption}</figcaption>'
        f'</figure>'
    )


def render_table_html(
    headers: list[str],
    rows: list[list[str]],
    *,
    table_class: str = "direction-table",
    style: str = "font-size:0.9em",
) -> str:
    """Render an HTML table from headers and row data.

    Args:
        headers: List of column header strings.
        rows: List of rows, each row a list of HTML cell strings (already escaped).
        table_class: CSS class for the <table> element.
        style: Inline style for the <table> element.

    Returns:
        HTML string for the complete <table> element.
    """
    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    tbody_rows = []
    for row in rows:
        tbody_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    tbody = "<tbody>" + "".join(tbody_rows) + "</tbody>"
    return f'<table class="{table_class}" style="{style}">{thead}{tbody}</table>'


def _safe_parse_json(response: str) -> Optional[dict]:
    """Parse JSON from LLM response. Delegates to shared implementation."""
    from .formatting import safe_parse_json
    return safe_parse_json(response)


def markdown_to_html(text: str) -> str:
    """Convert basic markdown formatting to HTML"""
    if not text:
        return ""

    # Escape HTML entities first to prevent XSS
    text = html_module.escape(text)

    # Headers: must be processed before bold/italic (longest prefix first)
    text = re.sub(r'^####\s+(.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)

    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)

    # Italic: *text* or _text_
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)

    # Code: `text`
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)

    # Bullet lists
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(<li>.*?</li>\n?)+', r'<ul>\g<0></ul>', text)

    # Markdown tables — process BEFORE line-break substitution
    def _convert_table_block(m):
        lines = m.group(0).strip().split('\n')
        html = '<div class="table-wrapper"><table><thead>'
        in_body = False
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('|'):
                continue
            cells = [c.strip() for c in line.strip('|').split('|')]
            # Separator row (e.g. |---|---|)
            if all(re.match(r'^[-: ]+$', c) for c in cells if c.strip()):
                if not in_body:
                    html += '</thead><tbody>'
                    in_body = True
                continue
            if not in_body:
                html += '<tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr>'
            else:
                html += '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
        if not in_body:
            html += '</thead><tbody>'
        html += '</tbody></table></div>'
        return html

    text = re.sub(r'(?:^[ \t]*\|[^\n]*\n?)+', _convert_table_block, text, flags=re.MULTILINE)

    # Line breaks
    text = text.replace('\n\n', '</p><p>')
    text = text.replace('\n', '<br>')

    return text


# HTML Template fragments for the report.
# CSS is injected from report_css.REPORT_CSS at render time (single source of truth).

_HEAD_PRE_CSS = """\
<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
"""

_HEAD_POST_CSS_BODY = """\
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="meta">{generated_at}</div>
        </div>

        <div class="content">
            <!-- Table of Contents -->
            <div class="toc">
                <h3>{toc_title}</h3>
                <ul>
                    {toc_items}
                </ul>
            </div>

            <!-- Graphic Abstract -->
            {graphic_abstract_html}

            <!-- Abstract -->
            <div class="abstract">
                <h2>{abstract_title}</h2>
                <p>{abstract}</p>
            </div>

            <!-- Cluster Annotations -->
            {cluster_annotations_html}

            <!-- Sections -->
            {sections_html}

            <!-- Conclusions -->
            <div class="conclusions">
                <h2>{conclusions_title}</h2>
                <p>{conclusions}</p>
            </div>

            <!-- Deterministic Evidence Qualifications -->
            {report_limitations_html}

            <!-- Analysis Log -->
            {analysis_log_html}

            <!-- Score Definitions -->
            {score_definitions_html}

            <!-- Appendix: Analysis Settings & Records -->
            {appendix_html}
        </div>
    </div>
</body>
</html>
"""


def render_report_html(context: dict) -> str:
    """Render the report HTML by sandwiching REPORT_CSS between the
    head-pre-CSS and post-CSS body templates, then formatting placeholders.

    `context` must supply every {field} appearing in _HEAD_PRE_CSS and
    _HEAD_POST_CSS_BODY (e.g. language, title, generated_at, abstract,
    sections_html, ...).
    """
    context = dict(context)
    context.setdefault("report_limitations_html", "")
    head = _HEAD_PRE_CSS.format(**context)
    body = _HEAD_POST_CSS_BODY.format(**context)
    return head + REPORT_CSS + body


# Backwards-compat shim: old callers may still pass the full template to
# .format(...). Rebuild it on demand by stitching the three pieces back.
HTML_TEMPLATE = _HEAD_PRE_CSS + REPORT_CSS.replace("{", "{{").replace("}", "}}") + _HEAD_POST_CSS_BODY
