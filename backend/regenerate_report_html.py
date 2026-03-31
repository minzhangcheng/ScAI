#!/usr/bin/env python3
"""
Regenerate HTML report with embedded base64 images.
This script takes an existing report folder and regenerates the HTML file
with all figures embedded as base64 for standalone viewing.
"""
import os
import io
import sys
import json
import base64
import uuid
import re
import html as html_module
from urllib.parse import quote as url_quote
from pathlib import Path
from datetime import datetime

import logging
from collections import defaultdict
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Import shared utilities — supports both package import and standalone script execution
try:
    from .utils.image_utils import compress_image_to_base64, categorize_figures
    from .config import get_session_logs
except ImportError:
    from utils.image_utils import compress_image_to_base64, categorize_figures
    from config import get_session_logs


def get_agent_llm_stats(session_id: str) -> Dict[str, Dict[str, Any]]:
    """Read LLM interaction logs and aggregate statistics per agent.

    Returns a dict mapping agent_name -> {
        "llm_server": str,      # raw base_url from user config
        "model_name": str,      # raw model string from user config
        "temperature": float,
        "call_count": int,
        "total_input_tokens": int,
        "total_output_tokens": int,
        "avg_response_time_ms": float,
    }
    """
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "llm_server": "",
        "model_name": "",
        "temperature": 0.0,
        "call_count": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_response_time_ms": 0.0,
    })

    log_path = get_session_logs(session_id) / "llm.jsonl"
    if not log_path.exists():
        return {}

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    interaction = json.loads(line)
                except json.JSONDecodeError:
                    continue

                agent = interaction.get("agent", "unknown")
                model = interaction.get("model", "")
                temperature = interaction.get("temperature", 0.0)
                usage = interaction.get("usage") or {}
                duration_ms = interaction.get("duration_ms")
                base_url = interaction.get("base_url") or ""

                s = stats[agent]
                # Capture first-seen values (stable per agent)
                if not s["llm_server"]:
                    s["llm_server"] = base_url
                if not s["model_name"]:
                    s["model_name"] = model
                if not s["temperature"]:
                    s["temperature"] = temperature
                s["call_count"] += 1
                s["total_input_tokens"] += usage.get("prompt_tokens", 0)
                s["total_output_tokens"] += usage.get("completion_tokens", 0)
                if duration_ms is not None:
                    s["total_response_time_ms"] += duration_ms

    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to read LLM logs from {log_path}: {e}")
        return {}

    # Compute averages
    for agent, s in stats.items():
        count = s["call_count"]
        if count > 0:
            s["avg_response_time_ms"] = s["total_response_time_ms"] / count
        else:
            s["avg_response_time_ms"] = 0.0
        del s["total_response_time_ms"]

    return dict(stats)


def markdown_to_html(text: str) -> str:
    """Convert basic markdown formatting to HTML"""
    if not text:
        return ""

    # Escape HTML entities first to prevent XSS
    text = html_module.escape(text)

    # Headers (longest prefix first)
    text = re.sub(r'^####\s+(.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)

    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)

    # Bullet lists
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(<li>.*?</li>\n?)+', r'<ul>\g<0></ul>', text)

    text = text.replace('\n\n', '</p><p>')
    text = text.replace('\n', '<br>')

    return text


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
            max-width: 1200px;
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

        .download-btn.pdf {{
            background: #fee2e2;
            color: #dc2626;
        }}

        .download-btn.csv {{
            background: #d1fae5;
            color: #047857;
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

        .section-type {{
            font-size: 0.75rem;
            padding: 2px 6px;
            border-radius: 4px;
            margin-left: 8px;
        }}

        .section-type.general {{
            background: #dbeafe;
            color: #1d4ed8;
        }}

        .section-type.specific {{
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

        /* Figure grid layout for multiple figures */
        .figure-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
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

        @media (max-width: 768px) {{
            .figure-grid {{
                grid-template-columns: 1fr;
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

            <!-- Abstract -->
            <div class="abstract">
                <h2>{abstract_title}</h2>
                <p>{abstract}</p>
            </div>

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


def _build_analysis_log_from_report(report: dict, language: str, figures_dir: Path) -> str:
    """Build analysis log HTML from report.json data. Mirrors DetailedReportGenerator._build_analysis_log_html()."""
    esc = html_module.escape

    log_data = report.get("analysis_log", [])
    if not log_data:
        return ""

    title = "Complete Analysis Log"
    subtitle = f"The following shows all {len(log_data)} analysis steps and their results in chronological order."

    type_badges = {
        "execute_code": ("Code", "#dbeafe", "#1d4ed8"),
        "consult_pi": ("PI Consultation", "#f3e8ff", "#7c3aed"),
        "search_literature": ("Literature Search", "#d1fae5", "#047857"),
        "annotate_clusters": ("Cluster Annotation", "#fef3c7", "#b45309"),
        "report_findings": ("Findings Report", "#dcfce7", "#15803d"),
        "generate_publication_figure": ("Publication Figure", "#fce7f3", "#be185d"),
        "text_response": ("Agent Response", "#f1f5f9", "#64748b"),
    }

    steps_html = ""
    for idx, entry in enumerate(log_data):
        step_num = entry.get("step_number", "?")
        step_type = entry.get("step_type", "unknown")
        timestamp = str(entry.get("timestamp", ""))[:19]
        reasoning = entry.get("agent_reasoning", "")

        badge_text, badge_bg, badge_color = type_badges.get(
            step_type, ("Step", "#f1f5f9", "#64748b")
        )

        step_html = f'''
        <div class="log-step" id="log-step-{idx}">
            <div class="log-step-header">
                <span class="log-step-number">#{step_num}</span>
                <span class="log-step-badge" style="background:{badge_bg};color:{badge_color}">{badge_text}</span>
                <span class="log-step-time">{esc(timestamp)}</span>
            </div>
        '''

        if reasoning:
            reasoning_label = "Agent Reasoning"
            step_html += f'<details class="log-reasoning-details"><summary>{reasoning_label}</summary><div class="log-reasoning">{esc(reasoning[:1000])}</div></details>'

        if step_type == "execute_code":
            desc = entry.get("code_description", "")
            code = entry.get("code", "")
            output = entry.get("code_output", "")
            error = entry.get("code_error", "")
            saved_figs = entry.get("saved_figure_filenames", [])

            if desc:
                step_html += f'<div class="log-description">{esc(desc)}</div>'
            if code:
                code_label = "View Code"
                step_html += f'<details><summary>{code_label}</summary><pre class="log-code"><code>{esc(code[:2500])}</code></pre></details>'
            if output:
                output_label = "Output"
                step_html += f'<details open><summary>{output_label}</summary><pre class="log-output">{esc(output[:2000])}</pre></details>'
            if error:
                error_label = "Error"
                step_html += f'<div class="log-error"><strong>{error_label}:</strong> {esc(error[:500])}</div>'

            if saved_figs and figures_dir:
                for fig_fname in saved_figs:
                    fig_path = figures_dir / os.path.basename(fig_fname)
                    if fig_path.exists():
                        img_src = compress_image_to_base64(str(fig_path), max_width=600, quality=80)
                        if img_src:
                            step_html += f'<div class="log-figure"><img src="{img_src}" alt="{esc(fig_fname)}" /><p class="log-fig-name">{esc(os.path.basename(fig_fname))}</p></div>'

        elif step_type == "consult_pi":
            q = entry.get("pi_question", "")
            r = entry.get("pi_response", "")
            q_label = "Question"
            r_label = "PI Response"
            if q:
                step_html += f'<div class="log-qa"><strong>{q_label}:</strong> {esc(q[:500])}</div>'
            if r:
                step_html += f'<div class="log-qa"><strong>{r_label}:</strong> {esc(r[:800])}</div>'

        elif step_type == "search_literature":
            topic = entry.get("literature_topic", "")
            result_text = entry.get("literature_result", "")
            topic_label = "Search Topic"
            result_label = "Results"
            if topic:
                step_html += f'<div class="log-qa"><strong>{topic_label}:</strong> {esc(topic)}</div>'
            if result_text:
                step_html += f'<details open><summary>{result_label}</summary><pre class="log-output">{esc(result_text[:1500])}</pre></details>'

        elif step_type == "annotate_clusters":
            markers = entry.get("cluster_markers_summary", "")
            tissue = entry.get("annotation_tissue", "")
            organism = entry.get("annotation_organism", "")
            ann_result = entry.get("annotation_result_summary", "")
            if markers:
                markers_label = "Cluster Markers"
                step_html += f'<div class="log-qa"><strong>{markers_label}:</strong> {esc(markers[:500])}</div>'
            if tissue or organism:
                ctx_label = "Tissue/Organism"
                step_html += f'<div class="log-qa"><strong>{ctx_label}:</strong> {esc(tissue)} / {esc(organism)}</div>'
            if ann_result:
                result_label = "Annotation Results"
                step_html += f'<details open><summary>{result_label}</summary><pre class="log-output">{esc(ann_result[:1500])}</pre></details>'

        elif step_type == "report_findings":
            findings_list = entry.get("findings", [])
            summary = entry.get("findings_summary", "")
            suggestions = entry.get("findings_suggestions", [])
            if summary:
                summary_label = "Summary"
                step_html += f'<div class="log-qa"><strong>{summary_label}:</strong> {esc(summary[:500])}</div>'
            if findings_list:
                findings_label = "Findings"
                items = "".join(f"<li>{esc(f[:300])}</li>" for f in findings_list[:20])
                step_html += f'<div class="log-findings"><strong>{findings_label}:</strong><ul>{items}</ul></div>'
            if suggestions:
                sugg_label = "Suggestions"
                items = "".join(f"<li>{esc(s[:200])}</li>" for s in suggestions[:10])
                step_html += f'<div class="log-findings"><strong>{sugg_label}:</strong><ul>{items}</ul></div>'

        elif step_type == "generate_publication_figure":
            fig_name = entry.get("pub_figure_name", "")
            fig_desc = entry.get("pub_figure_description", "")
            fig_code = entry.get("pub_figure_code", "")
            fig_result = entry.get("pub_figure_result", "")
            if fig_name:
                name_label = "Figure Name"
                step_html += f'<div class="log-qa"><strong>{name_label}:</strong> {esc(fig_name)}</div>'
            if fig_desc:
                desc_label = "Description"
                step_html += f'<div class="log-qa"><strong>{desc_label}:</strong> {esc(fig_desc[:300])}</div>'
            if fig_code:
                code_label = "View Code"
                step_html += f'<details><summary>{code_label}</summary><pre class="log-code"><code>{esc(fig_code[:2500])}</code></pre></details>'
            if fig_result:
                result_label = "Result"
                step_html += f'<details open><summary>{result_label}</summary><pre class="log-output">{esc(fig_result[:1000])}</pre></details>'

        elif step_type == "text_response":
            text = entry.get("text_content", "")
            if text:
                step_html += f'<div class="log-text-response">{esc(text[:1000])}</div>'

        step_html += '</div>'
        steps_html += step_html

    return f'''
    <div class="analysis-log" id="analysis_log">
        <h2>{title}</h2>
        <p class="log-subtitle">{subtitle}</p>
        {steps_html}
    </div>
    '''


def _build_score_definitions_from_report(report: dict, language: str) -> str:
    """Build score definitions HTML from report.json data."""
    score_defs = report.get("score_definitions", [])
    if not score_defs:
        return ""
    esc = html_module.escape
    score_rows = ""
    for sd in score_defs:
        gene_list = sd.get("gene_list", [])
        genes = esc(", ".join(gene_list[:10]))
        if len(gene_list) > 10:
            genes += f" ... ({len(gene_list)} total)"
        params = sd.get("parameters", {})
        params_str = esc(", ".join(f"{k}={v}" for k, v in params.items()) if params else "default")
        score_rows += f"""<tr>
            <td>{esc(sd.get("score_name", ""))}</td>
            <td><code>{esc(sd.get("method", ""))}</code></td>
            <td>{esc(sd.get("description", ""))}</td>
            <td style="font-size:0.85em">{params_str}</td>
            <td style="font-size:0.85em">{genes}</td>
            <td style="font-size:0.85em">{esc(sd.get("gene_list_source", "") or "N/A")}</td>
        </tr>"""
    return f"""
        <div class="section" id="score-definitions">
            <div class="section-header">
                <h2>Score Definitions</h2>
            </div>
            <div class="section-content">
                <p>The following scores were computed during the analysis. Each score's methodology is documented for transparency and reproducibility.</p>
                <table class="direction-table" style="font-size:0.9em">
                    <thead>
                        <tr>
                            <th>Score Name</th>
                            <th>Method</th>
                            <th>Description</th>
                            <th>Parameters</th>
                            <th>Gene List</th>
                            <th>Source</th>
                        </tr>
                    </thead>
                    <tbody>
                        {score_rows}
                    </tbody>
                </table>
            </div>
        </div>"""


def _build_appendix_from_report(report: dict, language: str) -> str:
    """Build appendix HTML from the appendix data stored in report.json."""
    appendix = report.get("appendix")
    if not appendix:
        return ""

    esc = html_module.escape

    appendix_title = "Appendix: Analysis Settings & Records"

    # 1. Project Information
    proj_title = "Project Information"
    lbl_name = "Project Name"
    lbl_desc = "Project Description"
    lbl_created = "Created At"
    lbl_report_time = "Report Generated At"

    project_html = f"""
    <h3>{proj_title}</h3>
    <table>
        <tbody>
            <tr><td><strong>{lbl_name}</strong></td><td>{esc(appendix.get('project_name', '-'))}</td></tr>
            <tr><td><strong>{lbl_desc}</strong></td><td>{esc((appendix.get('project_description') or '-')[:500])}</td></tr>
            <tr><td><strong>{lbl_created}</strong></td><td>{esc(appendix.get('created_at', '-'))}</td></tr>
            <tr><td><strong>{lbl_report_time}</strong></td><td>{esc(report.get('generated_at', '-'))}</td></tr>
        </tbody>
    </table>
    """

    # 2. Agent Configuration
    agent_title = "Agent Configuration"
    agent_rows = ""
    models = appendix.get("agent_models", {})
    temps = appendix.get("agent_temperatures", {})
    for key in ["pi", "pathology", "bioinformatics", "biochemistry", "literature", "painter", "supervisor", "technique"]:
        model = models.get(key, "-")
        temp = temps.get(key, "-")
        agent_rows += f"<tr><td>{esc(key.title())}</td><td>{esc(str(model))}</td><td>{temp}</td></tr>\n"

    agent_html = f"""
    <h3>{agent_title}</h3>
    <table>
        <thead><tr><th>Agent</th><th>Model</th><th>Temperature</th></tr></thead>
        <tbody>{agent_rows}</tbody>
    </table>
    """

    # 3. Uploaded Data
    data_title = "Uploaded Data"
    groups_html = ""
    groups = appendix.get("treatment_groups", [])
    if groups:
        lbl_group = "Treatment Groups"
        group_rows = ""
        for g in groups:
            group_rows += f"<tr><td>{esc(g.get('name', '-'))}</td><td>{esc(g.get('description', '-'))}</td><td>{g.get('sample_count', 0)}</td></tr>\n"
        groups_html = f"""
        <p><strong>{lbl_group}</strong></p>
        <table>
            <thead><tr><th>Group</th><th>Description</th><th>Samples</th></tr></thead>
            <tbody>{group_rows}</tbody>
        </table>
        """

    files_html = ""
    files = appendix.get("data_files", [])
    if files:
        lbl_files = "Data Files"
        file_rows = ""
        for f in files:
            file_rows += (
                f"<tr><td>{esc(f.get('filename', '-'))}</td><td>{esc(f.get('section', '-'))}</td>"
                f"<td>{esc(f.get('data_type', '-'))}</td><td>{esc(f.get('group') or '-')}</td></tr>\n"
            )
        files_html = f"""
        <p><strong>{lbl_files}</strong></p>
        <table>
            <thead><tr><th>Filename</th><th>Section</th><th>Type</th><th>Group</th></tr></thead>
            <tbody>{file_rows}</tbody>
        </table>
        """

    biochem_count = appendix.get("biochemistry_indicators_count", 0)
    biochem_html = ""
    if biochem_count:
        lbl_biochem = "Biochemistry Indicators"
        lbl_count = f"{biochem_count} manually entered indicator(s)"
        biochem_html = f"<p><strong>{lbl_biochem}:</strong> {lbl_count}</p>"

    data_html = f"<h3>{data_title}</h3>{groups_html}{files_html}{biochem_html}"

    # 4. Analysis Rounds
    rounds_title = "Analysis Rounds"
    round_results = appendix.get("round_results", [])
    if round_results:
        max_rounds = appendix.get("max_rounds", "?")
        lbl_total = f"Total: {len(round_results)} round(s) (max {max_rounds})"
        round_rows = ""
        for r in round_results:
            round_rows += (
                f"<tr><td>{r.get('round_number', '?')}</td><td>{esc(str(r.get('timestamp', '')))}</td>"
                f"<td>{r.get('pathology_findings_count', 0)}</td>"
                f"<td>{r.get('bioinformatics_findings_count', 0)}</td>"
                f"<td>{r.get('biochemistry_findings_count', 0)}</td>"
                f"<td>{r.get('literature_findings_count', 0)}</td></tr>\n"
            )
        rounds_html = f"""
        <h3>{rounds_title}</h3>
        <p>{lbl_total}</p>
        <table>
            <thead><tr>
                <th>Round</th><th>Time</th>
                <th>Pathology</th><th>Bioinformatics</th>
                <th>Biochemistry</th><th>Literature</th>
            </tr></thead>
            <tbody>{round_rows}</tbody>
        </table>
        """
    else:
        lbl_no_rounds = "No analysis round records"
        rounds_html = f"<h3>{rounds_title}</h3><p>{lbl_no_rounds}</p>"

    # 5. User Inputs Between Rounds
    human_title = "User Inputs Between Rounds"
    human_msgs = appendix.get("human_messages", [])
    if human_msgs:
        items = ""
        for hm in human_msgs:
            msg = hm.get("message", "")
            files_list = hm.get("files", [])
            ts = hm.get("timestamp", "")
            rn = hm.get("round_number", "?")
            round_label = f"Round {rn}"

            files_info = ""
            if files_list:
                file_names = ", ".join(esc(str(f)) for f in files_list)
                files_label = "Attached files"
                files_info = f'<div class="files">{files_label}: {file_names}</div>'

            items += f"""
            <div class="human-input-item">
                <span class="round-tag">{esc(round_label)}</span>
                <span class="timestamp">{esc(str(ts))}</span>
                <div class="message">{esc(msg)}</div>
                {files_info}
            </div>
            """
        human_html = f"<h3>{human_title}</h3>{items}"
    else:
        lbl_no_input = "No user inputs recorded"
        human_html = f"<h3>{human_title}</h3><p>{lbl_no_input}</p>"

    # 6. User Interaction History
    interaction_html = ""
    user_interactions = appendix.get("user_interactions", {})
    cluster_anns = user_interactions.get("cluster_annotations", [])
    dir_selections = user_interactions.get("direction_selection", [])

    if cluster_anns or dir_selections:
        interaction_title = "User Interaction History"
        interaction_html = f"<h3>{interaction_title}</h3>"

        if cluster_anns:
            ann_title = "Cell Annotation Decisions"
            ann_rows = ""
            for ca in cluster_anns:
                confirmed = "✓" if ca.get("confirmed") else "-"
                ann_rows += (
                    f"<tr><td>{esc(str(ca.get('cluster_id', '-')))}</td>"
                    f"<td>{esc(ca.get('cell_type', '-'))}</td>"
                    f"<td>{esc(ca.get('user_override', '-') or '-')}</td>"
                    f"<td>{confirmed}</td>"
                    f"<td>{esc(ca.get('confidence', '-'))}</td></tr>\n"
                )
            interaction_html += f"""
            <h4>{ann_title}</h4>
            <table>
                <thead><tr><th>Cluster</th><th>Cell Type</th>
                <th>User Override</th><th>Confirmed</th>
                <th>Confidence</th></tr></thead>
                <tbody>{ann_rows}</tbody>
            </table>
            """

        if dir_selections:
            dir_title = "Research Direction Selections"
            dir_rows = ""
            for d in dir_selections:
                dir_rows += f"<tr><td>{esc(d.get('direction', '-'))}</td><td>{esc(d.get('notes', '-') or '-')}</td></tr>\n"
            interaction_html += f"""
            <h4>{dir_title}</h4>
            <table>
                <thead><tr><th>Direction</th><th>User Notes</th></tr></thead>
                <tbody>{dir_rows}</tbody>
            </table>
            """

    # 7. Related Reports & Downloads
    downloads_html = ""
    related = appendix.get("related_reports", {})
    notebook = appendix.get("notebook_filename", "")
    if related or notebook:
        dl_title = "Related Reports & Downloads"
        dl_items = []
        if related.get("exploration_report"):
            label = "Exploration Report"
            dl_items.append(f'<li><a href="{esc(related["exploration_report"])}" download style="color:#4f46e5;">{label}</a></li>')
        if related.get("annotation_report"):
            label = "Annotation Context Report"
            dl_items.append(f'<li><a href="{esc(related["annotation_report"])}" download style="color:#4f46e5;">{label}</a></li>')
        if notebook:
            label = "Analysis Notebook (.ipynb)"
            dl_items.append(f'<li><a href="{esc(notebook)}" download style="color:#4f46e5;">{label}</a></li>')
        downloads_html = f'<h3>{dl_title}</h3><ul>{"".join(dl_items)}</ul>'

    return f"""
    <div class="appendix" id="appendix">
        <h2>{appendix_title}</h2>
        {project_html}
        {agent_html}
        {data_html}
        {rounds_html}
        {human_html}
        {interaction_html}
        {downloads_html}
    </div>
    """


def build_agent_stats_html(stats: Dict[str, Dict[str, Any]]) -> str:
    """Build agent LLM statistics HTML section."""
    if not stats:
        return ""

    esc = html_module.escape

    # Column headers
    headers = [
        "Agent",
        "LLM Server",
        "Model",
        "Temperature",
        "Calls",
        "Input Tokens",
        "Output Tokens",
        "Avg Response Time",
    ]

    header_html = "".join(f"<th>{esc(h)}</th>" for h in headers)

    rows = []
    for agent_name, s in sorted(stats.items()):
        avg_time = s.get("avg_response_time_ms", 0)
        avg_time_str = f"{avg_time:.0f}ms" if avg_time > 0 else "N/A"
        rows.append(f"""<tr>
            <td><strong>{esc(agent_name.replace("_", " ").title())}</strong></td>
            <td>{esc(s.get("llm_server", "-"))}</td>
            <td><code>{esc(s.get("model_name", "-"))}</code></td>
            <td>{s.get("temperature", 0):.1f}</td>
            <td>{s.get("call_count", 0)}</td>
            <td>{s.get("total_input_tokens", 0):,}</td>
            <td>{s.get("total_output_tokens", 0):,}</td>
            <td>{avg_time_str}</td>
        </tr>""")
    rows_html = "".join(rows)

    return f"""
    <div class="section" id="agent-llm-stats">
        <div class="section-header">
            <h2>Agent LLM Statistics</h2>
        </div>
        <div class="section-content">
            <p>The following table summarizes LLM usage for each agent during this analysis session.</p>
            <table>
                <thead><tr>{header_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </div>"""


def get_agent_stats_html(session_id: str) -> str:
    """Get agent LLM statistics HTML for a session, or empty string if unavailable."""
    stats = get_agent_llm_stats(session_id)
    return build_agent_stats_html(stats) if stats else ""


def generate_html_report(report_dir: Path, language: str = "en") -> Path:
    """
    Generate HTML report with embedded base64 images.
    """

    # Load report.json
    report_json_path = report_dir / "report.json"
    if not report_json_path.exists():
        raise FileNotFoundError(f"report.json not found in {report_dir}")

    with open(report_json_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"

    # Scan figures directory
    all_figures = categorize_figures(figures_dir)
    total_figs = sum(len(v) for v in all_figures.values())
    logger.info("Found %d figures in categories:", total_figs)
    for cat, figs in all_figures.items():
        if figs:
            logger.info("  - %s: %d", cat, len(figs))

    # Map section IDs to figure categories
    section_figure_map = {
        "qc_overview": ["qc"],
        "clustering_overview": ["umap", "clustering"],
        "marker_genes": ["markers", "heatmap"],
        "differential_expression": ["de", "violin"],
        "key_findings": ["de", "violin", "heatmap", "other"]
    }

    used_figures = set()

    # Prepare table of contents
    toc_items = []
    for section in report.get("sections", []):
        analysis_type = section.get("analysis_type", "general")
        type_class = "general" if analysis_type == "general" else "specific"
        type_label = "General" if analysis_type == "general" else "Specific"

        toc_items.append(
            f'<li><a href="#{html_module.escape(section["id"])}">{html_module.escape(section["title"])}</a>'
            f'<span class="section-type {type_class}">{type_label}</span></li>'
        )

    # Generate sections HTML
    sections_html = []
    for section in report.get("sections", []):
        analysis_type = section.get("analysis_type", "general")
        type_class = "general" if analysis_type == "general" else "specific"
        type_label = "General" if analysis_type == "general" else "Specific"

        content_html = markdown_to_html(section.get("content", ""))

        section_html = f'''
        <div class="section" id="{html_module.escape(section["id"])}">
            <div class="section-header">
                <h2>
                    {html_module.escape(section["title"])}
                    <span class="analysis-type-badge {type_class}">{type_label}</span>
                </h2>
            </div>
            <div class="section-content">
                <p>{content_html}</p>
        '''

        # Collect all figures for this section
        section_figures = list(section.get("figures", []))

        # Add figures from scanned directory based on section type
        section_id = section.get("id", "")
        categories_for_section = section_figure_map.get(section_id, [])

        for category in categories_for_section:
            for fig_info in all_figures.get(category, []):
                if fig_info["png_path"] in used_figures:
                    continue
                if any(f.get("png_path") == fig_info["png_path"] for f in section_figures):
                    continue

                section_figures.append({
                    "id": str(uuid.uuid4())[:8],
                    "name": fig_info["name"].replace("_", " ").title(),
                    "caption": "",
                    "png_path": fig_info["png_path"],
                    "pdf_path": fig_info["pdf_path"]
                })
                used_figures.add(fig_info["png_path"])

        # Add figures with base64 embedding
        if section_figures:
            section_html += '<div class="figure-grid">'

        fig_counter = 0
        for figure in section_figures:
            fig_counter += 1
            fig_id = figure.get("id", str(uuid.uuid4())[:8])
            fig_name = figure.get("name", "Figure")
            png_path = figure.get("png_path", "")
            pdf_path = figure.get("pdf_path", "")

            png_filename = url_quote(os.path.basename(png_path)) if png_path else ""
            pdf_filename = url_quote(os.path.basename(pdf_path)) if pdf_path else ""

            image_exists = png_path and os.path.exists(png_path)
            img_src = ""
            if image_exists:
                logger.info("  Embedding: %s", png_filename)
                img_src = compress_image_to_base64(png_path, max_width=900, quality=85)
                used_figures.add(png_path)

            fig_label = f"Fig. {section_id[:3].upper()}-{fig_counter}"
            fig_caption = figure.get("caption", "")
            fig_caption_html = markdown_to_html(fig_caption) if fig_caption else ""

            fig_interpretation = figure.get("interpretation", "")
            fig_interpretation_html = markdown_to_html(fig_interpretation) if fig_interpretation else ""

            img_tag = f"<img src='{img_src}' alt='{html_module.escape(fig_name)}' />" if img_src else "<p>Image not available</p>"

            interpretation_section = ""
            if fig_interpretation_html:
                interpretation_section = f'''
                            <div class="figure-interpretation">
                                <strong>Scientific Interpretation:</strong>
                                <p>{fig_interpretation_html}</p>
                            </div>'''

            section_html += f'''
                <div class="figure-container" id="fig-{fig_id}">
                    <div class="figure-image">
                        {img_tag}
                    </div>
                    <div class="figure-caption">
                        <h4>{html_module.escape(fig_label)}: {html_module.escape(fig_name)}</h4>
                        {"<p>" + fig_caption_html + "</p>" if fig_caption_html else ""}
                        {interpretation_section}
                        <div class="figure-actions">
                            <a href="figures/{png_filename}" download class="download-btn png">
                                "Download PNG"
                            </a>
                            {"<a href='figures/" + pdf_filename + "' download class='download-btn pdf'>" + "Download PDF" + "</a>" if pdf_path and os.path.exists(pdf_path) else ""}
                        </div>
                    </div>
                </div>
            '''

        if section_figures:
            section_html += '</div>'

        # Add tables
        for table in section.get("tables", []):
            tbl_name = table.get("name", "Table")
            csv_path = table.get("csv_path", "")
            columns = table.get("columns", [])
            preview_data = table.get("preview_data", [])
            tbl_interpretation = table.get("interpretation", "")

            csv_filename = url_quote(os.path.basename(csv_path)) if csv_path else ""
            tbl_interpretation_html = markdown_to_html(tbl_interpretation) if tbl_interpretation else ""

            tbl_interpretation_section = ""
            if tbl_interpretation_html:
                tbl_interpretation_section = f'''
                            <div class="table-interpretation">
                                <strong>Scientific Interpretation:</strong>
                                <p>{tbl_interpretation_html}</p>
                            </div>'''

            table_header = "".join([f"<th>{html_module.escape(str(col))}</th>" for col in columns])
            table_rows = ""
            for row in preview_data[:10]:
                row_cells = "".join([f"<td>{html_module.escape(str(row.get(col, '-')))}</td>" for col in columns])
                table_rows += f"<tr>{row_cells}</tr>"

            section_html += f'''
                <div class="table-container">
                    <div class="table-header">
                        <h4>{html_module.escape(tbl_name)}</h4>
                        <a href="tables/{csv_filename}" download class="download-btn csv">
                            "Download CSV"
                        </a>
                    </div>
                    <div class="table-wrapper">
                        <table>
                            <thead><tr>{table_header}</tr></thead>
                            <tbody>{table_rows}</tbody>
                        </table>
                    </div>
                    {tbl_interpretation_section}
                    {"<div class='table-info'>" + "Showing 10 of " + str(len(preview_data)) + " rows" + "</div>" if len(preview_data) > 10 else ""}
                </div>
            '''

        section_html += '''
            </div>
        </div>
        '''
        sections_html.append(section_html)

    # Add supplementary figures section
    remaining_figures = []
    for category, figs in all_figures.items():
        for fig_info in figs:
            if fig_info["png_path"] not in used_figures:
                remaining_figures.append(fig_info)

    if remaining_figures:
        logger.info("  Adding %d supplementary figures...", len(remaining_figures))
        supp_title = "Supplementary Figures"
        supp_badge = "Supplementary"
        toc_items.append(
            f'<li><a href="#supplementary_figures">{supp_title}</a>'
            f'<span class="section-type specific">{supp_badge}</span></li>'
        )

        supp_section_html = f'''
        <div class="section" id="supplementary_figures">
            <div class="section-header">
                <h2>
                    {supp_title}
                    <span class="analysis-type-badge specific">{supp_badge}</span>
                </h2>
            </div>
            <div class="section-content">
                <p>The following figures were generated during the analysis and are provided for reference.</p>
                <div class="figure-grid">
        '''

        for idx, fig_info in enumerate(remaining_figures, 1):
            fig_name = fig_info["name"].replace("_", " ").title()
            png_path = fig_info["png_path"]
            pdf_path = fig_info["pdf_path"]
            png_filename = url_quote(os.path.basename(png_path))
            pdf_filename = url_quote(os.path.basename(pdf_path)) if pdf_path else ""

            logger.info("  Embedding supplementary: %s", os.path.basename(png_path))
            img_src = compress_image_to_base64(png_path, max_width=900, quality=85) if os.path.exists(png_path) else ""
            img_tag = f"<img src='{img_src}' alt='{html_module.escape(fig_name)}' />" if img_src else "<p>Image not available</p>"

            supp_section_html += f'''
                <div class="figure-container" id="fig-supp-{idx}">
                    <div class="figure-image">
                        {img_tag}
                    </div>
                    <div class="figure-caption">
                        <h4>Supp. Fig. {idx}: {html_module.escape(fig_name)}</h4>
                        <div class="figure-actions">
                            <a href="figures/{png_filename}" download class="download-btn png">
                                "Download PNG"
                            </a>
                            {"<a href='figures/" + pdf_filename + "' download class='download-btn pdf'>" + "Download PDF" + "</a>" if pdf_path and os.path.exists(pdf_path) else ""}
                        </div>
                    </div>
                </div>
            '''

        supp_section_html += '''
                </div>
            </div>
        </div>
        '''
        sections_html.append(supp_section_html)

    # Convert markdown in abstract and conclusions
    abstract_html = markdown_to_html(report.get("abstract", ""))
    conclusions_html = markdown_to_html(report.get("conclusions", ""))

    # Escape curly braces in content to prevent str.format() crash on LLM output
    def _safe_format_value(v: str) -> str:
        return v.replace("{", "{{").replace("}", "}}")

    # Analysis log is no longer shown in the main report
    analysis_log_html = ""

    # Build score definitions HTML from report.json data if available
    score_definitions_html = _build_score_definitions_from_report(report, language) if report.get("score_definitions") else ""
    if score_definitions_html:
        score_toc_label = "Score Definitions"
        toc_items.append(
            f'<li><a href="#score-definitions">{score_toc_label}</a>'
            f'<span class="section-type specific">Methods</span></li>'
        )

    # Build appendix HTML from report.json data if available
    appendix_html = _build_appendix_from_report(report, language) if report.get("appendix") else ""
    if appendix_html:
        appendix_toc_label = "Appendix: Analysis Settings & Records"
        toc_items.append(
            f'<li><a href="#appendix">{appendix_toc_label}</a>'
            f'<span class="section-type specific">Appendix</span></li>'
        )

    # Build agent LLM statistics HTML
    agent_stats_html = ""
    session_id = report.get("session_id")
    if session_id:
        agent_stats_html = get_agent_stats_html(session_id)

    # Build full HTML
    html_content = HTML_TEMPLATE.format(
        language=language,
        title=_safe_format_value(report.get("title", "Analysis Report")),
        generated_at="Generated: " + report.get("generated_at", datetime.now().isoformat()),
        toc_title="Table of Contents",
        toc_items=_safe_format_value("\n".join(toc_items)),
        abstract_title="Abstract",
        abstract=_safe_format_value(abstract_html),
        sections_html=_safe_format_value("\n".join(sections_html)),
        conclusions_title="Conclusions",
        conclusions=_safe_format_value(conclusions_html),
        analysis_log_html=_safe_format_value(analysis_log_html),
        score_definitions_html=_safe_format_value(score_definitions_html),
        agent_stats_html=_safe_format_value(agent_stats_html),
        appendix_html=_safe_format_value(appendix_html)
    )

    # Save HTML file
    html_path = report_dir / "report_embedded.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("HTML report saved to: %s", html_path)
    return html_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) < 2:
        logger.error("Usage: python regenerate_report_html.py <report_directory>")
        logger.error("Example: python regenerate_report_html.py ../data/reports/427c8e3c-16da-4b09-a908-392e68ec9fb3/")
        sys.exit(1)

    report_dir = Path(sys.argv[1])
    if not report_dir.exists():
        logger.error("Error: Directory %s does not exist", report_dir)
        sys.exit(1)

    generate_html_report(report_dir)
