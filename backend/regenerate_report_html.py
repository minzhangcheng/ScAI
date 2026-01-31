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
from pathlib import Path
from datetime import datetime

from PIL import Image


def compress_image_to_base64(image_path: str, max_width: int = 800, quality: int = 85) -> str:
    """
    Compress an image and convert to base64 for HTML embedding.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize if too large
            width, height = img.size
            if width > max_width:
                ratio = max_width / width
                new_size = (max_width, int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Save to buffer as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)

            b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_data}"
    except Exception as e:
        print(f"Error compressing image {image_path}: {e}")
        return ""


def categorize_figures(figures_dir: Path) -> dict:
    """
    Scan figures directory and categorize images by their content type.
    """
    categories = {
        "qc": [],
        "umap": [],
        "clustering": [],
        "markers": [],
        "de": [],
        "heatmap": [],
        "violin": [],
        "other": []
    }

    if not figures_dir.exists():
        return categories

    for png_file in sorted(figures_dir.glob("*.png")):
        filename = png_file.stem.lower()
        pdf_file = png_file.with_suffix('.pdf')

        fig_info = {
            "name": png_file.stem,
            "png_path": str(png_file),
            "pdf_path": str(pdf_file) if pdf_file.exists() else None,
            "filename": png_file.name
        }

        if "qc" in filename or "quality" in filename:
            categories["qc"].append(fig_info)
        elif "umap" in filename:
            categories["umap"].append(fig_info)
        elif "cluster" in filename or "leiden" in filename:
            categories["clustering"].append(fig_info)
        elif "marker" in filename:
            categories["markers"].append(fig_info)
        elif "volcano" in filename or "_de_" in filename or "differential" in filename:
            categories["de"].append(fig_info)
        elif "heatmap" in filename:
            categories["heatmap"].append(fig_info)
        elif "violin" in filename:
            categories["violin"].append(fig_info)
        elif "dotplot" in filename:
            categories["markers"].append(fig_info)
        elif "stacked_bar" in filename or "composition" in filename or "fold_change" in filename:
            categories["clustering"].append(fig_info)
        elif "feature" in filename or "expression" in filename:
            categories["de"].append(fig_info)
        else:
            categories["other"].append(fig_info)

    return categories


def markdown_to_html(text: str) -> str:
    """Convert basic markdown formatting to HTML"""
    if not text:
        return ""

    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
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
        </div>
    </div>
</body>
</html>
"""


def generate_html_report(report_dir: Path, language: str = "en") -> Path:
    """
    Generate HTML report with embedded base64 images.
    """
    zh = language == "zh"

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
    print(f"Found {total_figs} figures in categories:")
    for cat, figs in all_figures.items():
        if figs:
            print(f"  - {cat}: {len(figs)}")

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
        type_label = ("General" if analysis_type == "general" else "Specific")

        toc_items.append(
            f'<li><a href="#{section["id"]}">{section["title"]}</a>'
            f'<span class="section-type {type_class}">{type_label}</span></li>'
        )

    # Generate sections HTML
    sections_html = []
    for section in report.get("sections", []):
        analysis_type = section.get("analysis_type", "general")
        type_class = "general" if analysis_type == "general" else "specific"
        type_label = ("General" if analysis_type == "general" else "Specific")

        content_html = markdown_to_html(section.get("content", ""))

        section_html = f'''
        <div class="section" id="{section["id"]}">
            <div class="section-header">
                <h2>
                    {section["title"]}
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

            png_filename = os.path.basename(png_path) if png_path else ""
            pdf_filename = os.path.basename(pdf_path) if pdf_path else ""

            image_exists = png_path and os.path.exists(png_path)
            img_src = ""
            if image_exists:
                print(f"  Embedding: {png_filename}")
                img_src = compress_image_to_base64(png_path, max_width=900, quality=85)
                used_figures.add(png_path)

            fig_label = f"Fig. {section_id[:3].upper()}-{fig_counter}"
            fig_caption = figure.get("caption", "")
            fig_caption_html = markdown_to_html(fig_caption) if fig_caption else ""

            img_tag = f"<img src='{img_src}' alt='{fig_name}' />" if img_src else "<p>Image not available</p>"

            section_html += f'''
                <div class="figure-container" id="fig-{fig_id}">
                    <div class="figure-image">
                        {img_tag}
                    </div>
                    <div class="figure-caption">
                        <h4>{fig_label}: {fig_name}</h4>
                        {"<p>" + fig_caption_html + "</p>" if fig_caption_html else ""}
                        <div class="figure-actions">
                            <a href="figures/{png_filename}" download class="download-btn png">
                                Download PNG
                            </a>
                            {"<a href='figures/" + pdf_filename + "' download class='download-btn pdf'>Download PDF</a>" if pdf_path and os.path.exists(pdf_path) else ""}
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

            csv_filename = os.path.basename(csv_path) if csv_path else ""

            table_header = "".join([f"<th>{col}</th>" for col in columns])
            table_rows = ""
            for row in preview_data[:10]:
                row_cells = "".join([f"<td>{row.get(col, '-')}</td>" for col in columns])
                table_rows += f"<tr>{row_cells}</tr>"

            section_html += f'''
                <div class="table-container">
                    <div class="table-header">
                        <h4>{tbl_name}</h4>
                        <a href="tables/{csv_filename}" download class="download-btn csv">
                            Download CSV
                        </a>
                    </div>
                    <div class="table-wrapper">
                        <table>
                            <thead><tr>{table_header}</tr></thead>
                            <tbody>{table_rows}</tbody>
                        </table>
                    </div>
                    {"<div class='table-info'>Showing 10 of " + str(len(preview_data)) + " rows</div>" if len(preview_data) > 10 else ""}
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
        print(f"  Adding {len(remaining_figures)} supplementary figures...")
        supp_title = "Supplementary Figures"
        toc_items.append(
            f'<li><a href="#supplementary_figures">{supp_title}</a>'
            f'<span class="section-type specific">Supplementary</span></li>'
        )

        supp_section_html = f'''
        <div class="section" id="supplementary_figures">
            <div class="section-header">
                <h2>
                    {supp_title}
                    <span class="analysis-type-badge specific">Supplementary</span>
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
            png_filename = os.path.basename(png_path)
            pdf_filename = os.path.basename(pdf_path) if pdf_path else ""

            print(f"  Embedding supplementary: {png_filename}")
            img_src = compress_image_to_base64(png_path, max_width=900, quality=85) if os.path.exists(png_path) else ""
            img_tag = f"<img src='{img_src}' alt='{fig_name}' />" if img_src else "<p>Image not available</p>"

            supp_section_html += f'''
                <div class="figure-container" id="fig-supp-{idx}">
                    <div class="figure-image">
                        {img_tag}
                    </div>
                    <div class="figure-caption">
                        <h4>Supp. Fig. {idx}: {fig_name}</h4>
                        <div class="figure-actions">
                            <a href="figures/{png_filename}" download class="download-btn png">
                                Download PNG
                            </a>
                            {"<a href='figures/" + pdf_filename + "' download class='download-btn pdf'>Download PDF</a>" if pdf_path and os.path.exists(pdf_path) else ""}
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

    # Build full HTML
    html_content = HTML_TEMPLATE.format(
        language="en",
        title=report.get("title", "Analysis Report"),
        generated_at="Generated: " + report.get("generated_at", datetime.now().isoformat()),
        toc_title="Table of Contents",
        toc_items="\n".join(toc_items),
        abstract_title="Abstract",
        abstract=abstract_html,
        sections_html="\n".join(sections_html),
        conclusions_title="Conclusions",
        conclusions=conclusions_html
    )

    # Save HTML file
    html_path = report_dir / "report_embedded.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nHTML report saved to: {html_path}")
    return html_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python regenerate_report_html.py <report_directory>")
        print("Example: python regenerate_report_html.py ../intesc_data/reports/427c8e3c-16da-4b09-a908-392e68ec9fb3/")
        sys.exit(1)

    report_dir = Path(sys.argv[1])
    if not report_dir.exists():
        print(f"Error: Directory {report_dir} does not exist")
        sys.exit(1)

    generate_html_report(report_dir)
