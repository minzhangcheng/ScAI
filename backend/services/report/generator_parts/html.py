"""HTML assembly for the detailed-report generator facade."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from backend.models.schemas import AnalysisState


async def generate_html_report(
    generator, report: Dict[str, Any], language: str,
    state: Optional[AnalysisState] = None, *, bindings,
) -> Path:
    """
        Generate HTML report with embedded images and download links.
        Images are embedded as base64 for standalone HTML display.
        """

    # Scan figures directory for all available images
    all_figures = bindings.categorize_figures(generator.figures_dir)
    bindings.logger.info(f"Found figures in categories: {', '.join([f'{k}: {len(v)}' for k, v in all_figures.items() if v])}")

    # Map section IDs to figure categories
    section_figure_map = {
        "qc_overview": ["qc"],
        "clustering_overview": ["umap", "clustering"],
        "marker_genes": ["markers", "heatmap"],
        "differential_expression": ["de", "violin"],
        "key_findings": ["de", "violin", "heatmap", "other"]
    }

    # Track which figures and tables have been used
    used_figures = set()
    used_tables = set()

    # Prepare table of contents
    toc_items = []
    for section in report.get("sections", []):
        analysis_type = section.get("analysis_type", "general")
        type_class = "general" if analysis_type == "general" else "specific"
        type_label = ("General") if analysis_type == "general" else ("Specific")

        toc_items.append(
            f'<li><a href="#{bindings.html_module.escape(section["id"])}">{bindings.html_module.escape(section["title"])}</a>'
            f'<span class="section-type {type_class}">{type_label}</span></li>'
        )

    sections_html = _render_primary_sections(
        report, all_figures=all_figures, section_figure_map=section_figure_map,
        used_figures=used_figures, used_tables=used_tables, bindings=bindings,
    )
    _append_supplementary_sections(
        generator, all_figures=all_figures, used_figures=used_figures,
        used_tables=used_tables, toc_items=toc_items,
        sections_html=sections_html, bindings=bindings,
    )
    _append_followups_section(
        report, toc_items=toc_items, sections_html=sections_html, bindings=bindings,
    )
    return _write_html_report(
        generator, report, language=language, state=state,
        toc_items=toc_items, sections_html=sections_html, bindings=bindings,
    )


def _render_primary_sections(
    report, *, all_figures, section_figure_map, used_figures, used_tables, bindings,
):
    # Generate sections HTML
    sections_html = []
    for section in report.get("sections", []):
        analysis_type = section.get("analysis_type", "general")
        type_class = "general" if analysis_type == "general" else "specific"
        type_label = ("General") if analysis_type == "general" else ("Specific")

        # Convert markdown in content to HTML
        content_html = bindings.markdown_to_html(section.get("content", ""))

        section_html = f'''
            <div class="section" id="{bindings.html_module.escape(section["id"])}">
                <div class="section-header">
                    <h2>
                        {bindings.html_module.escape(section["title"])}
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
                # Skip if already used
                if fig_info["png_path"] in used_figures:
                    continue
                # Skip if already in section figures
                if any(f.get("png_path") == fig_info["png_path"] for f in section_figures):
                    continue

                # Add to section figures
                section_figures.append({
                    "id": str(bindings.uuid.uuid4())[:8],
                    "name": fig_info["name"].replace("_", " ").title(),
                    "caption": "",
                    "png_path": fig_info["png_path"],
                    "pdf_path": fig_info["pdf_path"]
                })
                used_figures.add(fig_info["png_path"])

        # Add figures with base64 embedding
        fig_counter = 0
        for figure in section_figures:
            fig_counter += 1
            fig_id = figure.get("id", str(bindings.uuid.uuid4())[:8])
            fig_name = figure.get("name", "Figure")
            png_path = figure.get("png_path", "")
            pdf_path = figure.get("pdf_path", "")

            png_filename = bindings.url_quote(bindings.os.path.basename(png_path)) if png_path else ""
            pdf_filename = bindings.url_quote(bindings.os.path.basename(pdf_path)) if pdf_path else ""
            img_ext = bindings.os.path.splitext(png_path)[1].lstrip('.').upper() if png_path else "JPG"

            # Check if image file exists
            image_exists = png_path and bindings.os.path.exists(png_path)

            # Generate base64 embedded image
            img_src = ""
            if image_exists:
                img_src = bindings.compress_image_to_base64(png_path, max_width=900, quality=85)
                used_figures.add(png_path)

            # Generate figure ID for referencing
            fig_label = f"Figure {section_id[:3].upper()}-{fig_counter}"
            fig_caption = figure.get("caption", "")
            fig_caption_html = bindings.markdown_to_html(fig_caption) if fig_caption else ""

            # Get interpretation if available
            fig_interpretation = figure.get("interpretation", "")
            fig_interpretation_html = bindings.markdown_to_html(fig_interpretation) if fig_interpretation else ""
            reservation_html = bindings.render_figure_reservation_notice_html(
                figure.get("reservation_notice")
            )
            legacy_note = (
                '<p class="artifact-provenance legacy-unreviewed">'
                "Legacy unregistered artifact; no publication review record is available."
                "</p>"
                if figure.get("is_legacy") else ""
            )

            img_tag = f"<img src='{img_src}' alt='{bindings.html_module.escape(fig_name)}' />" if img_src else ("<p>" + ("Image not available") + "</p>")

            # Build interpretation section if available
            interpretation_section = ""
            if fig_interpretation_html:
                interpretation_section = f'''
                            <div class="figure-interpretation">
                                <strong>{"Scientific Interpretation"}:</strong>
                                <p>{fig_interpretation_html}</p>
                            </div>'''

            section_html += f'''
                    <div class="figure-container" id="fig-{fig_id}">
                        <div class="figure-image">
                            {img_tag}
                        </div>
                        <div class="figure-caption">
                            <h4>{bindings.html_module.escape(fig_label)}: {bindings.html_module.escape(fig_name)}</h4>
                            {legacy_note}
                            {"<p><em>" + fig_caption_html + "</em></p>" if fig_caption_html else ""}
                            {reservation_html}
                            {interpretation_section}
                            <div class="figure-actions">
                                <a href="figures/{png_filename}" download class="download-btn png">
                                    Download {img_ext}
                                </a>
                                {"<a href='figures/" + pdf_filename + "' download class='download-btn pdf'>" + ("Download PDF (High-Res)") + "</a>" if pdf_path and bindings.os.path.exists(pdf_path) else ""}
                            </div>
                        </div>
                    </div>
                '''

        # Add tables
        for table in section.get("tables", []):
            tbl_name = table.get("name", "Table")
            csv_path = table.get("csv_path", "")
            used_tables.add(bindings.os.path.basename(csv_path) if csv_path else "")
            columns = table.get("columns", [])
            preview_data = table.get("preview_data", [])
            total_rows = table.get("total_rows", len(preview_data))
            tbl_interpretation = table.get("interpretation", "")

            csv_filename = bindings.url_quote(bindings.os.path.basename(csv_path)) if csv_path else ""
            tbl_interpretation_html = bindings.markdown_to_html(tbl_interpretation) if tbl_interpretation else ""

            tbl_interpretation_section = ""
            if tbl_interpretation_html:
                tbl_interpretation_section = f'''
                            <div class="table-interpretation">
                                <strong>{"Scientific Interpretation"}:</strong>
                                <p>{tbl_interpretation_html}</p>
                            </div>'''

            # Build table HTML
            table_header = "".join([f"<th>{bindings.html_module.escape(str(col))}</th>" for col in columns])
            table_rows = ""
            for row in preview_data[:10]:
                row_cells = "".join([f"<td>{bindings.html_module.escape(str(row.get(col, '-')))}</td>" for col in columns])
                table_rows += f"<tr>{row_cells}</tr>"

            section_html += f'''
                    <div class="table-container">
                        <div class="table-header">
                            <h4>{bindings.html_module.escape(tbl_name)}</h4>
                            <a href="tables/{csv_filename}" download class="download-btn csv">
                                {"Download CSV"}
                            </a>
                        </div>
                        <div class="table-wrapper">
                            <table>
                                <thead><tr>{table_header}</tr></thead>
                                <tbody>{table_rows}</tbody>
                            </table>
                        </div>
                        {tbl_interpretation_section}
                        {"<div class='table-info'>" + (f"Showing 10 of {total_rows} rows") + "</div>" if total_rows > 10 else ""}
                    </div>
                '''

        section_html += '''
                </div>
            </div>
            '''
        sections_html.append(section_html)

    return sections_html


def _append_supplementary_sections(
    generator, *, all_figures, used_figures, used_tables, toc_items,
    sections_html, bindings,
):
    # Add supplementary figures section for any remaining figures
    remaining_figures = []
    for category, figs in all_figures.items():
        for fig_info in figs:
            if fig_info["png_path"] not in used_figures:
                remaining_figures.append(fig_info)

    if remaining_figures:
        supp_title = "Supplementary Figures"
        toc_items.append(
            f'<li><a href="#supplementary_figures">{supp_title}</a>'
            f'<span class="section-type specific">{"Supplementary"}</span></li>'
        )

        supp_section_html = f'''
            <div class="section" id="supplementary_figures">
                <div class="section-header">
                    <h2>
                        {supp_title}
                        <span class="analysis-type-badge specific">{"Supplementary"}</span>
                    </h2>
                </div>
                <div class="section-content">
                    <p>{"The following figures were generated during the analysis and are provided for reference."}</p>
            '''

        for idx, fig_info in enumerate(remaining_figures, 1):
            fig_name = fig_info["name"].replace("_", " ").title()
            png_path = fig_info["png_path"]
            pdf_path = fig_info["pdf_path"]
            png_filename = bindings.url_quote(bindings.os.path.basename(png_path))
            pdf_filename = bindings.url_quote(bindings.os.path.basename(pdf_path)) if pdf_path else ""
            img_ext = bindings.os.path.splitext(png_path)[1].lstrip('.').upper() if png_path else "JPG"

            img_src = bindings.compress_image_to_base64(png_path, max_width=900, quality=85) if bindings.os.path.exists(png_path) else ""
            img_tag = f"<img src='{img_src}' alt='{bindings.html_module.escape(fig_name)}' />" if img_src else ("<p>" + ("Image not available") + "</p>")

            supp_section_html += f'''
                    <div class="figure-container" id="fig-supp-{idx}">
                        <div class="figure-image">
                            {img_tag}
                        </div>
                        <div class="figure-caption">
                            <h4>{"Supp. Fig."} {idx}: {bindings.html_module.escape(fig_name)}</h4>
                            <div class="figure-actions">
                                <a href="figures/{png_filename}" download class="download-btn png">
                                    Download {img_ext}
                                </a>
                                {"<a href='figures/" + pdf_filename + "' download class='download-btn pdf'>" + ("Download PDF (High-Res)") + "</a>" if pdf_path and bindings.os.path.exists(pdf_path) else ""}
                            </div>
                        </div>
                    </div>
                '''

        supp_section_html += '''
                </div>
            </div>
            '''
        sections_html.append(supp_section_html)

    # Add supplementary data tables for any CSV files not assigned to sections
    remaining_table_files = []
    if generator.tables_dir.exists():
        for csv_file in sorted(generator.tables_dir.glob("*.csv")):
            if csv_file.name not in used_tables:
                columns, preview_data, total_rows = [], [], 0
                try:
                    df_preview = bindings.pd.read_csv(csv_file, nrows=10)
                    columns = list(df_preview.columns)
                    preview_data = df_preview.to_dict('records')
                    total_rows = bindings.pd.read_csv(csv_file, usecols=[0]).shape[0]
                except Exception:
                    bindings.logger.warning("Failed to read CSV preview for HTML: %s", csv_file, exc_info=True)
                remaining_table_files.append({
                    "name": csv_file.stem.replace("_", " ").title(),
                    "filename": csv_file.name,
                    "columns": columns,
                    "preview_data": preview_data,
                    "total_rows": total_rows,
                    "inferred_type": "differential_expression" if csv_file.stem.lower().startswith(("de_", "dea_", "deg_")) else "data"
                })

    if remaining_table_files:
        supp_tbl_title = "Supplementary Data Tables"
        toc_items.append(
            f'<li><a href="#supplementary_tables">{supp_tbl_title}</a>'
            f'<span class="section-type specific">{"Supplementary"}</span></li>'
        )
        supp_tbl_html = f'''
            <div class="section" id="supplementary_tables">
                <div class="section-header">
                    <h2>
                        {supp_tbl_title}
                        <span class="analysis-type-badge specific">{"Supplementary"}</span>
                    </h2>
                </div>
                <div class="section-content">
                    <p>{"The following tables were generated during analysis and are provided for download and reference."}</p>
            '''
        for tbl in remaining_table_files:
            tbl_name = tbl["name"]
            csv_filename = bindings.url_quote(tbl["filename"])
            columns = tbl["columns"]
            preview_data = tbl["preview_data"]
            total_rows = tbl.get("total_rows", len(preview_data))
            inferred_type = tbl["inferred_type"]

            type_label = ("Differential Expression") if inferred_type == "differential_expression" else ("Analysis Data")
            col_list = ", ".join(str(c) for c in columns[:8]) + ("..." if len(columns) > 8 else "")

            table_header = "".join([f"<th>{bindings.html_module.escape(str(col))}</th>" for col in columns])
            table_rows = ""
            for row in preview_data[:5]:
                row_cells = "".join([f"<td>{bindings.html_module.escape(str(row.get(col, '-')))}</td>" for col in columns])
                table_rows += f"<tr>{row_cells}</tr>"

            supp_tbl_html += f'''
                    <div class="table-container">
                        <div class="table-header">
                            <h4>{bindings.html_module.escape(tbl_name)}</h4>
                            <span class="table-type-badge">{type_label}</span>
                            <a href="tables/{csv_filename}" download class="download-btn csv">
                                {"Download Full CSV"}
                            </a>
                        </div>
                        <p class="table-columns-info">{"Columns: "}{bindings.html_module.escape(col_list)}</p>
                        <div class="table-wrapper">
                            <table>
                                <thead><tr>{table_header}</tr></thead>
                                <tbody>{table_rows}</tbody>
                            </table>
                        </div>
                        {"<div class='table-info'>" + (f"Showing 5 of {total_rows} rows — download CSV for full data") + "</div>" if total_rows > 5 else ""}
                    </div>
                '''
        supp_tbl_html += '''
                </div>
            </div>
            '''
        sections_html.append(supp_tbl_html)


def _append_followups_section(report, *, toc_items, sections_html, bindings):
    """Render readable PI follow-ups without exposing bookkeeping IDs."""
    followups = [item for item in (report.get("followups") or []) if isinstance(item, dict)]
    if not followups:
        return

    title = "Follow-ups"
    toc_items.append(
        f'<li><a href="#followups">{title}</a>'
        f'<span class="section-type specific">Next steps</span></li>'
    )
    cards = []
    disposition_labels = {
        "deepen_now": "Deepen now",
        "offer_in_chat": "Continue in Chat",
        "needs_new_data": "Needs new data",
        "closed": "Closed by PI",
    }
    for item in followups:
        esc = bindings.html_module.escape
        question = esc(str(item.get("question") or ""))
        why = str(item.get("why_it_matters") or "").strip()
        reason = str(item.get("reason") or "").strip()
        instruction = str(item.get("suggested_instruction") or "").strip()
        required_input = str(item.get("required_input") or "").strip()
        disposition = str(item.get("disposition") or "").strip().lower()
        source_round = item.get("source_round")
        direction = str(item.get("direction") or "").strip()
        source = f"Round {source_round}" if source_round else "Source round not recorded"
        if direction:
            source += f" · {esc(direction)}"
        body = [f"<h4>{question}</h4>"]
        if why:
            body.append(f"<p><strong>Why it matters:</strong> {bindings.markdown_to_html(why)}</p>")
        body.append(
            f"<p><strong>Disposition:</strong> {esc(disposition_labels.get(disposition, disposition or 'Recorded by PI'))}</p>"
        )
        if reason:
            body.append(f"<p><strong>PI reason:</strong> {bindings.markdown_to_html(reason)}</p>")
        if instruction:
            body.append(f"<p><strong>Suggested instruction:</strong> {bindings.markdown_to_html(instruction)}</p>")
        if required_input:
            body.append(f"<p><strong>Required input:</strong> {bindings.markdown_to_html(required_input)}</p>")
        outcome = str(item.get("execution_outcome") or "").strip()
        if outcome:
            body.append(f"<p><strong>Attempted outcome:</strong> {bindings.markdown_to_html(outcome)}</p>")
        body.append(f"<p class=\"followup-source\"><strong>Source:</strong> {source}</p>")
        cards.append('<div class="followup-card">' + "".join(body) + "</div>")

    sections_html.append(
        f'''<div class="section" id="followups">
            <div class="section-header"><h2>{title}
                <span class="analysis-type-badge specific">Next steps</span>
            </h2></div>
            <div class="section-content">{''.join(cards)}</div>
        </div>'''
    )



def _write_html_report(
    generator, report, *, language, state, toc_items, sections_html, bindings,
):
    # Convert markdown in abstract and conclusions
    abstract_html = bindings.markdown_to_html(report.get("abstract", ""))
    conclusions_html = bindings.markdown_to_html(report.get("conclusions", ""))

    # Build graphic abstract HTML
    ga_html = ""
    ga_path = report.get("graphic_abstract_path")
    if ga_path and bindings.os.path.exists(ga_path):
        try:
            ga_b64 = bindings.compress_image_to_base64(ga_path, max_width=1200, quality=90)
            if ga_b64:
                ga_title = "Graphic Abstract"
                ga_html = f'''
            <div class="graphic-abstract">
                <h2>{ga_title}</h2>
                <img src="{ga_b64}" alt="{ga_title}" />
            </div>
'''
            else:
                bindings.logger.warning(f"Graphic abstract image compression returned empty for: {ga_path}")
        except Exception as e:
            bindings.logger.exception("Failed to embed graphic abstract in HTML")

    # Cluster annotations table removed (n_cells not populated by pipeline)
    ca_html = ""

    # Analysis log is no longer shown in the main report
    analysis_log_html = ""

    # Build score definitions HTML
    score_definitions_html = ""
    if state is not None and state.score_definitions:
        score_definitions_html = bindings.build_score_definitions_section_html(state.score_definitions)
        score_toc_label = "Score Definitions"
        toc_items.append(
            f'<li><a href="#score-definitions">{score_toc_label}</a>'
            f'<span class="section-type specific">{"Methods"}</span></li>'
        )

    # Build appendix HTML (analysis settings & records)
    appendix_html = ""
    if state is not None:
        appendix_html = generator._build_appendix_html(state, report, language)
        appendix_toc_label = "Appendix: Analysis Settings & Records"
        toc_items.append(
            f'<li><a href="#appendix">{appendix_toc_label}</a>'
            f'<span class="section-type specific">{"Appendix"}</span></li>'
        )

    # Escape curly braces in content to prevent str.format() crash on LLM output
    def _safe_format_value(v: str) -> str:
        return v.replace("{", "{{").replace("}", "}}")

    # Build full HTML — protect against unescaped braces in dynamic content
    try:
        html_content = bindings.render_report_html({
            "language": "en",
            "title": _safe_format_value(report.get("title", "Analysis Report")),
            "generated_at": ("Generated: ") + report.get("generated_at", ""),
            "report_limitations_html": _safe_format_value(
                bindings.render_technical_review_notice_html(report)
                + bindings.render_report_limitations_html(report)
            ),
            "report_status_line_html": _safe_format_value(
                bindings.render_report_status_line_html(report)
            ),
            "toc_title": "Table of Contents",
            "toc_items": _safe_format_value("\n".join(toc_items)),
            "graphic_abstract_html": _safe_format_value(ga_html),
            "abstract_title": "Abstract",
            "abstract": _safe_format_value(abstract_html),
            "cluster_annotations_html": _safe_format_value(ca_html),
            "sections_html": _safe_format_value("\n".join(sections_html)),
            "conclusions_title": "Conclusions",
            "conclusions": _safe_format_value(conclusions_html),
            "analysis_log_html": _safe_format_value(analysis_log_html),
            "score_definitions_html": _safe_format_value(score_definitions_html),
            "appendix_html": _safe_format_value(appendix_html),
        })
    except (KeyError, ValueError, IndexError) as e:
        bindings.logger.error(f"HTML template formatting failed: {e}, falling back to plain report")
        # Fallback: produce a minimal valid HTML page with the report content
        plain_sections = "\n".join(sections_html) if sections_html else ""
        html_content = (
            f"<!DOCTYPE html><html><head><meta charset='UTF-8'>"
            f"<title>{bindings.html_module.escape(report.get('title', 'Analysis Report'))}</title></head>"
            f"<body><h1>{bindings.html_module.escape(report.get('title', 'Analysis Report'))}</h1>"
            f"<p>Report formatting failed ({bindings.html_module.escape(str(e))}). Raw content below:</p>"
            f"<div>{plain_sections}</div></body></html>"
        )

    # Save HTML file
    html_path = generator.output_dir / "report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    bindings.logger.info(f"HTML report saved to: {html_path}")

    # Save score definitions CSV into tables/ so it flows into download-all-tables
    if state is not None and state.score_definitions:
        csv_path = bindings.write_score_definitions_csv(state.score_definitions, generator.output_dir / "tables")
        if csv_path is not None:
            bindings.logger.info(f"Score definitions CSV saved to: {csv_path}")

    return html_path
