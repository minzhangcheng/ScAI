"""
Shared HTML utilities for report generation.
Exports HTML_TEMPLATE, markdown_to_html, and figure helpers.
"""
import html as html_module
import re
from pathlib import Path
from typing import Dict, Optional


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


# HTML Template for the report
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary-color: #4f46e5;
            --primary-light: #eef2ff;
            --text-color: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
            --bg-color: #f8fafc;
            --white: #ffffff;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background: var(--bg-color);
            padding: 20px;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: var(--white);
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
        }}

        .header .meta {{
            opacity: 0.9;
            font-size: 0.9rem;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 40px;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
        }}

        .section-header {{
            background: var(--primary-light);
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
        }}

        .section-header h2 {{
            font-size: 1.25rem;
            color: var(--primary-color);
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .section-header .badge {{
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 9999px;
            background: var(--primary-color);
            color: white;
            font-weight: 500;
        }}

        .section-content {{
            padding: 24px;
        }}

        .section-content p {{
            margin-bottom: 16px;
            text-align: justify;
        }}

        .section-content strong {{
            font-weight: 600;
            color: #1e40af;
        }}

        .section-content em {{
            font-style: italic;
            color: #6b7280;
        }}

        .section-content code {{
            background: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9em;
            color: #dc2626;
        }}

        .graphic-abstract {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px 24px;
            background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%);
            border-left: 4px solid #ec4899;
            border-radius: 0 8px 8px 0;
        }}

        .graphic-abstract h2 {{
            color: #be185d;
            margin-bottom: 16px;
        }}

        .graphic-abstract img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .abstract {{
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-left: 4px solid #0ea5e9;
            padding: 20px 24px;
            margin-bottom: 30px;
            border-radius: 0 8px 8px 0;
        }}

        .abstract h2 {{
            color: #0369a1;
            margin-bottom: 12px;
        }}

        .conclusions {{
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border-left: 4px solid #22c55e;
            padding: 20px 24px;
            margin-top: 30px;
            border-radius: 0 8px 8px 0;
        }}

        .conclusions h2 {{
            color: #15803d;
            margin-bottom: 12px;
        }}

        .figure-container {{
            margin: 24px 0;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            background: var(--bg-color);
        }}

        .figure-image {{
            text-align: center;
            padding: 16px;
            background: white;
        }}

        .figure-image img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}

        .figure-caption {{
            padding: 16px;
            border-top: 1px solid var(--border-color);
            background: var(--white);
        }}

        .figure-caption h4 {{
            font-size: 0.95rem;
            margin-bottom: 8px;
            color: var(--text-color);
        }}

        .figure-caption p {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }}

        .figure-interpretation {{
            margin-top: 12px;
            padding: 12px;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-radius: 8px;
            border-left: 3px solid #0ea5e9;
        }}

        .figure-interpretation strong {{
            color: #0369a1;
            font-size: 0.85rem;
        }}

        .figure-interpretation p {{
            font-size: 0.85rem;
            color: var(--text-color);
            margin-top: 8px;
            text-align: justify;
        }}

        .table-interpretation {{
            margin-top: 12px;
            padding: 12px;
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border-radius: 8px;
            border-left: 3px solid #22c55e;
        }}

        .table-interpretation strong {{
            color: #047857;
            font-size: 0.85rem;
        }}

        .table-interpretation p {{
            font-size: 0.85rem;
            color: var(--text-color);
            margin-top: 8px;
            text-align: justify;
        }}

        .figure-actions {{
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }}

        .download-btn {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 6px 12px;
            font-size: 0.8rem;
            border-radius: 6px;
            text-decoration: none;
            transition: all 0.2s;
        }}

        .download-btn.png {{
            background: #fce7f3;
            color: #be185d;
        }}

        .download-btn.png:hover {{
            background: #fbcfe8;
        }}

        .download-btn.pdf {{
            background: #fee2e2;
            color: #dc2626;
        }}

        .download-btn.pdf:hover {{
            background: #fecaca;
        }}

        .table-container {{
            margin: 24px 0;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }}

        .table-header {{
            background: #ecfdf5;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .table-header h4 {{
            color: #047857;
            font-size: 0.95rem;
        }}

        .download-btn.csv {{
            background: #d1fae5;
            color: #047857;
        }}

        .download-btn.csv:hover {{
            background: #a7f3d0;
        }}

        .table-wrapper {{
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}

        th {{
            background: #f1f5f9;
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid var(--border-color);
        }}

        td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
        }}

        tr:hover {{
            background: #f8fafc;
        }}

        .table-info {{
            padding: 12px 16px;
            background: #f8fafc;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}

        .toc {{
            background: var(--bg-color);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 30px;
        }}

        .toc h3 {{
            margin-bottom: 16px;
            color: var(--text-color);
        }}

        .toc ul {{
            list-style: none;
        }}

        .toc li {{
            padding: 8px 0;
            border-bottom: 1px dashed var(--border-color);
        }}

        .toc li:last-child {{
            border-bottom: none;
        }}

        .toc a {{
            color: var(--primary-color);
            text-decoration: none;
        }}

        .toc a:hover {{
            text-decoration: underline;
        }}

        .toc .section-type {{
            font-size: 0.75rem;
            padding: 2px 6px;
            border-radius: 4px;
            margin-left: 8px;
        }}

        .toc .general {{
            background: #dbeafe;
            color: #1d4ed8;
        }}

        .toc .specific {{
            background: #fef3c7;
            color: #b45309;
        }}

        .analysis-type-badge {{
            font-size: 0.7rem;
            padding: 3px 8px;
            border-radius: 4px;
            margin-left: 8px;
            font-weight: 500;
        }}

        .analysis-type-badge.general {{
            background: #dbeafe;
            color: #1d4ed8;
        }}

        .analysis-type-badge.specific {{
            background: #fef3c7;
            color: #b45309;
        }}

        .appendix {{
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border-left: 4px solid #f59e0b;
            padding: 20px 24px;
            margin-top: 30px;
            border-radius: 0 8px 8px 0;
        }}

        .appendix h2 {{
            color: #b45309;
            margin-bottom: 16px;
        }}

        .appendix h3 {{
            color: #92400e;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 1.05rem;
            border-bottom: 1px solid #fbbf24;
            padding-bottom: 6px;
        }}

        .appendix table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            margin: 10px 0 16px 0;
            background: white;
            border-radius: 6px;
            overflow: hidden;
        }}

        .appendix th {{
            background: #fef3c7;
            padding: 8px 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #fbbf24;
            color: #92400e;
        }}

        .appendix td {{
            padding: 8px 12px;
            border-bottom: 1px solid #fde68a;
        }}

        .appendix .human-input-item {{
            background: white;
            border-radius: 6px;
            padding: 12px 16px;
            margin: 8px 0;
            border: 1px solid #fde68a;
        }}

        .appendix .human-input-item .round-tag {{
            display: inline-block;
            background: #fbbf24;
            color: #78350f;
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
            margin-right: 8px;
        }}

        .appendix .human-input-item .timestamp {{
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}

        .appendix .human-input-item .message {{
            margin-top: 6px;
            white-space: pre-wrap;
        }}

        .appendix .human-input-item .files {{
            margin-top: 6px;
            font-size: 0.85rem;
            color: #b45309;
        }}

        .appendix ul {{
            margin: 6px 0 12px 20px;
            font-size: 0.9rem;
        }}

        .appendix li {{
            margin-bottom: 4px;
        }}

        /* Analysis Log Timeline */
        .analysis-log {{
            padding: 30px;
        }}
        .analysis-log h2 {{
            font-size: 1.6rem;
            margin-bottom: 10px;
            color: var(--text-color);
        }}
        .log-subtitle {{
            color: var(--text-secondary);
            margin-bottom: 20px;
        }}
        .log-step {{
            border-left: 3px solid var(--border-color);
            margin-left: 20px;
            padding: 12px 20px;
            margin-bottom: 8px;
            background: #fafafa;
            border-radius: 0 8px 8px 0;
        }}
        .log-step-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .log-step-number {{
            font-weight: bold;
            color: var(--primary-color);
            min-width: 35px;
            font-size: 0.9rem;
        }}
        .log-step-badge {{
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            white-space: nowrap;
        }}
        .log-step-time {{
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-left: auto;
        }}
        .log-reasoning-details {{
            margin: 6px 0;
        }}
        .log-reasoning {{
            background: #f0f4ff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #374151;
            line-height: 1.5;
            margin-top: 4px;
        }}
        .log-description {{
            font-weight: 500;
            margin-bottom: 6px;
            color: var(--text-color);
        }}
        .log-code {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            overflow-x: auto;
            margin-top: 4px;
        }}
        .log-output {{
            background: #f1f5f9;
            padding: 10px;
            border-radius: 4px;
            max-height: 300px;
            overflow-y: auto;
            font-size: 0.8rem;
            white-space: pre-wrap;
            word-break: break-word;
            margin-top: 4px;
        }}
        .log-error {{
            background: #fef2f2;
            color: #dc2626;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 6px 0;
            font-size: 0.85rem;
        }}
        .log-figure {{
            margin: 10px 0;
            text-align: center;
        }}
        .log-figure img {{
            max-width: 600px;
            width: 100%;
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }}
        .log-fig-name {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}
        .log-qa {{
            margin: 4px 0;
            font-size: 0.9rem;
            line-height: 1.5;
        }}
        .log-findings ul {{
            padding-left: 20px;
            margin-top: 4px;
        }}
        .log-findings li {{
            font-size: 0.85rem;
            margin-bottom: 2px;
        }}
        .log-text-response {{
            background: #f8fafc;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #475569;
        }}
        .analysis-log details summary {{
            cursor: pointer;
            color: var(--primary-color);
            font-size: 0.85rem;
            user-select: none;
        }}
        .analysis-log details summary:hover {{
            text-decoration: underline;
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
            .download-btn {{
                display: none;
            }}
            .analysis-log details,
            .analysis-log details > * {{
                display: block !important;
            }}
            .analysis-log details:not([open]) > :not(summary) {{
                display: block !important;
            }}
        }}
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

            <!-- Analysis Log -->
            {analysis_log_html}

            <!-- Score Definitions -->
            {score_definitions_html}

            <!-- Agent LLM Statistics -->
            {agent_stats_html}

            <!-- Appendix: Analysis Settings & Records -->
            {appendix_html}
        </div>
    </div>
</body>
</html>
"""
