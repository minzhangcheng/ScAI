"""HTML rendering of the detailed report: markdown, figures, tables, appendix.

Moved verbatim out of backend/agents/supervisor_agent.py; composed into SupervisorAgent.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import os
import re
import io
import base64
import shutil
from pathlib import Path
from urllib.parse import quote as url_quote

from ...utils.files import resolve_figure_in_workspace

logger = logging.getLogger(__name__)


from ...models.schemas import AnalysisState
from ...services.report.artifacts import artifact_stems, eligible_artifacts


# ---------------------------------------------------------------------------
# Rich HTML report CSS (extracted to utils/report_css.py for separation of concerns)
# ---------------------------------------------------------------------------
from ...utils.report_css import REPORT_CSS as _REPORT_CSS  # noqa: E402

from .shared import (  # noqa: F401
    _SPECIFIC_SECTION_IDS,
    _SECTION_ID_TO_DEFAULT_TITLE,
    _PLACEHOLDER_SECTION_TITLES,
    SupervisorRecord,
    SupervisorEvalResult,
    _CRITICAL_OUTPUT_PATTERNS,
    _PERCENTAGE_PATTERN,
    _INF_NAN_PATTERN,
    _SUPERVISOR_SYSTEM_PROMPT,
    _serialize_record,
    persist_graphic_abstract,
)


class ReportHtmlMixin:
    """HTML rendering of the detailed report: markdown, figures, tables, appendix."""

    def _generate_html_report(
        self,
        report: Dict,
        sections: List[Dict],
        abstract: str,
        conclusions: str,
        report_figures: List[Dict],
        figure_map: Dict[str, Path],
        stats: Dict,
        language: str,
        output_path: Path,
        workspace_path: Optional[Path],
    ) -> Path:
        """Generate a standalone HTML report with embedded images and download links."""
        import html as html_module

        def _maybe_escape_html(text: str) -> str:
            """Return *text* as-is if it already contains HTML tags, otherwise escape it.

            Caption text from [FIGURE_CAPTION:]/[TABLE_CAPTION:] markers has been
            through _markdown_to_html() and may contain safe HTML.  Captions from
            pub_fig_descriptions are plain text and need escaping.
            """
            if '<' in text and '>' in text:
                return text
            return html_module.escape(text)

        # Build figure base64 cache for embedding
        figure_b64_cache: Dict[str, str] = {}
        for fig_name, fig_path in figure_map.items():
            try:
                with open(fig_path, "rb") as f:
                    raw = f.read()
                figure_b64_cache[fig_name] = base64.b64encode(raw).decode()
            except Exception as e:
                logger.warning(f"Could not read figure {fig_path}: {e}")

        # Build TOC
        toc_items = []
        for sec in sections:
            a_type = sec.get("analysis_type", "general")
            type_class = "general" if a_type == "general" else "specific"
            type_label = "General" if a_type == "general" else "Specific"
            toc_items.append(
                f'<li><a href="#{html_module.escape(sec["id"])}">'
                f'{html_module.escape(sec["title"])}</a>'
                f'<span class="section-type {type_class}">{type_label}</span></li>'
            )

        # Build publication figure description lookup
        pub_fig_descriptions: Dict[str, Dict[str, str]] = {}
        analysis_state = report.get("_analysis_state")
        eligible_figures = eligible_artifacts(
            analysis_state.publication_figures if analysis_state else []
        )
        if eligible_figures:
            for pf in eligible_figures:
                fname = os.path.basename(pf.png_path) if pf.png_path else ""
                if fname:
                    pub_fig_descriptions[fname] = {
                        "name": pf.name,
                        "description": pf.description or "",
                        "interpretation": pf.interpretation or "",
                    }

        def _alt_ext_name(name: str) -> str:
            """Return filename with alternate image extension (.png↔.jpg/.jpeg)."""
            stem, ext = os.path.splitext(name)
            lower_ext = ext.lower()
            if lower_ext in (".jpg", ".jpeg"):
                return stem + ".png"
            return stem + ".jpg"

        def _build_figure_html(fig_name: str, caption: str = "") -> str:
            """Build inline figure HTML block for a given figure filename."""
            fig_b64 = figure_b64_cache.get(fig_name, "")
            # Fallback: try alternate extension (.png ↔ .jpg/.jpeg) if exact name not found
            if not fig_b64:
                alt_name = _alt_ext_name(fig_name)
                fig_b64 = figure_b64_cache.get(alt_name, "")
                if fig_b64:
                    logger.info(f"Figure extension fallback: '{fig_name}' → '{alt_name}'")
                    fig_name = alt_name
            if not fig_b64:
                return ""
            # Determine MIME type from extension
            mime = "image/jpeg" if fig_name.lower().endswith((".jpg", ".jpeg")) else "image/png"

            # Figure description: prefer LLM caption, fallback to publication figures
            fig_desc = pub_fig_descriptions.get(fig_name, {})
            display_name = fig_desc.get("name", fig_name.replace("_", " ").rsplit(".", 1)[0].title())
            description = fig_desc.get("description", "")
            interpretation = caption or fig_desc.get("interpretation", "")

            # Description paragraph
            desc_html = ""
            if description:
                desc_html = f'<p><em>{html_module.escape(description)}</em></p>'

            # Interpretation block
            interp_html = ""
            if interpretation:
                interp_content = _maybe_escape_html(interpretation)
                interp_html = (
                    f'<div class="figure-interpretation">'
                    f'<strong>Interpretation</strong>'
                    f'<p>{interp_content}</p>'
                    f'</div>'
                )

            # Download buttons — label matches actual file extension
            fig_path = figure_map.get(fig_name) or figure_map.get(_alt_ext_name(fig_name))
            img_ext = os.path.splitext(fig_name)[1].lstrip('.').upper() or "PNG"
            img_btn = (
                f'<a href="figures/{html_module.escape(fig_name)}" '
                f'download class="download-btn png">Download {img_ext}</a>'
            )
            pdf_btn = ""
            if fig_path:
                pdf_p = fig_path.with_suffix(".pdf")
                if pdf_p.exists():
                    pdf_btn = (
                        f'<a href="figures/{html_module.escape(pdf_p.name)}" '
                        f'download class="download-btn pdf">Download PDF</a>'
                    )

            return f'''
                <div class="figure-container">
                    <div class="figure-image">
                        <img src="data:{mime};base64,{fig_b64}"
                             alt="{html_module.escape(fig_name)}"/>
                    </div>
                    <div class="figure-caption">
                        <h4>{html_module.escape(display_name)}</h4>
                        {desc_html}
                        {interp_html}
                        <div class="figure-actions">
                            {img_btn}
                            {pdf_btn}
                        </div>
                    </div>
                </div>
            '''

        # Track which figures have been placed
        placed_figures: set = set()

        # Build table lookup for inline embedding
        table_lookup: Dict[str, Dict] = {}
        for tbl in report.get("tables", []):
            table_lookup[tbl["filename"]] = tbl
        placed_tables: set = set()

        def _build_table_html(tbl_filename: str, caption: str = "") -> str:
            """Build inline table HTML block for a given CSV filename."""
            tbl_info = table_lookup.get(tbl_filename)
            if not tbl_info:
                return ""
            tbl_path = tbl_info.get("path", "")
            if not tbl_path or not Path(tbl_path).exists():
                return ""

            display_name = tbl_info.get("name", tbl_filename.replace("_", " ").rsplit(".", 1)[0].title())
            csv_url = url_quote(tbl_filename)

            # Read preview data (up to 10 rows) from CSV
            columns = tbl_info.get("columns", [])
            total_rows = tbl_info.get("total_rows", 0)
            preview_data = []
            try:
                import pandas as pd
                df_preview = pd.read_csv(tbl_path, nrows=10)
                columns = list(df_preview.columns)
                preview_data = df_preview.to_dict('records')
                if not total_rows:
                    total_rows = sum(1 for _ in open(tbl_path)) - 1
            except Exception:
                logger.debug("Failed to read CSV preview for HTML table: %s", tbl_path)

            if not columns:
                return ""

            table_header = "".join(
                f"<th>{html_module.escape(str(col))}</th>" for col in columns
            )
            table_rows = ""
            for row in preview_data[:10]:
                row_cells = "".join(
                    f"<td>{html_module.escape(str(row.get(col, '-')))}</td>" for col in columns
                )
                table_rows += f"<tr>{row_cells}</tr>"

            row_info = ""
            if total_rows > 10:
                row_info = f'<div class="table-info">Showing 10 of {total_rows} rows</div>'

            caption_html = ""
            if caption:
                caption_html = (
                    f'<div class="table-interpretation">'
                    f'<strong>Interpretation</strong>'
                    f'<p>{_maybe_escape_html(caption)}</p>'
                    f'</div>'
                )

            return f'''
                <div class="table-container">
                    <div class="table-header">
                        <h4>{html_module.escape(display_name)}</h4>
                        <a href="tables/{csv_url}" download class="download-btn csv">
                            Download CSV
                        </a>
                    </div>
                    <div class="table-wrapper">
                        <table>
                            <thead><tr>{table_header}</tr></thead>
                            <tbody>{table_rows}</tbody>
                        </table>
                    </div>
                    {row_info}
                    {caption_html}
                </div>
            '''

        # Build sections HTML with inline figure embedding
        sections_html = []
        for sec in sections:
            raw_content = sec.get("content", "")
            content_html = self._markdown_to_html(raw_content)

            # Replace [FIGURE: filename] + optional [FIGURE_CAPTION: ...] markers
            def _replace_figure_marker(m):
                fname = m.group(1).strip()
                caption = (m.group(2) or "").strip() if m.lastindex >= 2 else ""
                fig_html = _build_figure_html(fname, caption=caption)
                if fig_html:
                    placed_figures.add(fname)
                    return fig_html
                logger.info(f"Figure referenced in report but not found, skipping: {fname}")
                return ""

            # Match [FIGURE: name] optionally followed by [FIGURE_CAPTION: text]
            content_html = re.sub(
                r'\[FIGURE:\s*([^\]]+)\](?:\s*(?:<[^>]*>\s*)*\[FIGURE_CAPTION:\s*([^\]]+)\])?',
                _replace_figure_marker,
                content_html,
            )

            # Replace [TABLE: filename] + optional [TABLE_CAPTION: ...] markers
            def _replace_table_marker(m):
                fname = m.group(1).strip()
                caption = (m.group(2) or "").strip() if m.lastindex >= 2 else ""
                tbl_html = _build_table_html(fname, caption=caption)
                if tbl_html:
                    placed_tables.add(fname)
                    return tbl_html
                logger.info(f"Table referenced in report but not found, skipping: {fname}")
                return ""

            # Match [TABLE: name] optionally followed by [TABLE_CAPTION: text]
            content_html = re.sub(
                r'\[TABLE:\s*([^\]]+)\](?:\s*(?:<[^>]*>\s*)*\[TABLE_CAPTION:\s*([^\]]+)\])?',
                _replace_table_marker,
                content_html,
            )

            # Fallback: embed figures referenced by filename in text but without markers
            remaining_fig_parts = []
            for fig_name in figure_b64_cache:
                if fig_name in placed_figures:
                    continue
                if fig_name.lower() in raw_content.lower():
                    fig_html = _build_figure_html(fig_name)
                    if fig_html:
                        placed_figures.add(fig_name)
                        remaining_fig_parts.append(fig_html)

            # Fallback: embed tables referenced by filename in text but without markers
            remaining_tbl_parts = []
            for tbl_filename in table_lookup:
                if tbl_filename in placed_tables:
                    continue
                if tbl_filename.lower() in raw_content.lower():
                    tbl_html = _build_table_html(tbl_filename)
                    if tbl_html:
                        placed_tables.add(tbl_filename)
                        remaining_tbl_parts.append(tbl_html)

            a_type = sec.get("analysis_type", "general")
            type_class = "general" if a_type == "general" else "specific"
            type_label = "General" if a_type == "general" else "Specific"

            sec_html = f'''
            <div class="section" id="{html_module.escape(sec["id"])}">
                <div class="section-header">
                    <h2>
                        {html_module.escape(sec["title"])}
                        <span class="analysis-type-badge {type_class}">{type_label}</span>
                    </h2>
                </div>
                <div class="section-content">
                    {content_html}
                    {"".join(remaining_fig_parts)}
                    {"".join(remaining_tbl_parts)}
                </div>
            </div>
            '''
            sections_html.append(sec_html)

        # Append any unplaced figures as a supplementary section
        unplaced = [fn for fn in figure_b64_cache if fn not in placed_figures and not fn.startswith("graphic_abstract.")]
        if unplaced:
            supp_title = "Supplementary Figures"
            supp_figs = "".join(_build_figure_html(fn) for fn in unplaced)
            sections_html.append(f'''
            <div class="section" id="supplementary_figures">
                <div class="section-header">
                    <h2>{supp_title}</h2>
                </div>
                <div class="section-content">
                    {supp_figs}
                </div>
            </div>
            ''')

        # Append any unplaced tables as a supplementary section
        unplaced_tables = [fn for fn in table_lookup if fn not in placed_tables]
        if unplaced_tables:
            supp_tbl_title = "Supplementary Data Tables"
            supp_tbl_parts = []
            for fn in unplaced_tables:
                tbl_html = _build_table_html(fn)
                if tbl_html:
                    supp_tbl_parts.append(tbl_html)
            if supp_tbl_parts:
                sections_html.append(f'''
                <div class="section" id="supplementary_tables">
                    <div class="section-header">
                        <h2>{supp_tbl_title}</h2>
                    </div>
                    <div class="section-content">
                        <p>The following data tables were generated during analysis and are provided for download and reference.</p>
                        {"".join(supp_tbl_parts)}
                    </div>
                </div>
                ''')

        title = report.get("title", "Analysis Report")

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Graphic abstract HTML (derived from analysis_state, embedded as <img>)
        _ga_obj = getattr(analysis_state, "graphic_abstract", None) if analysis_state else None
        ga_b64 = (getattr(_ga_obj, "image_base64", "") or "") if _ga_obj else ""
        ga_html = ""
        if ga_b64:
            ga_html = (
                '<section id="graphic-abstract"><h2>Graphic Abstract</h2>'
                f'<figure><img src="data:image/jpeg;base64,{ga_b64}" '
                'alt="Graphic Abstract" style="max-width:100%"/></figure></section>'
            )

        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_module.escape(title)}</title>
<style>
{_REPORT_CSS}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>{html_module.escape(title)}</h1>
        <div class="meta">Generated: {generated_at}</div>
    </div>
    <div class="content">
        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>{"".join(toc_items)}</ul>
        </div>

        <div class="abstract">
            <h2>Abstract</h2>
            {self._markdown_to_html(abstract)}
        </div>

        {ga_html}
        {"".join(sections_html)}

        <div class="conclusions">
            <h2>Conclusions</h2>
            {self._markdown_to_html(conclusions)}
        </div>

        {self._build_score_definitions_html(report)}

        {self._build_appendix_html_section(report)}
    </div>
</div>
</body>
</html>'''

        html_file = output_path / "report.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Copy figures to output for download links
        figures_out = output_path / "figures"
        figures_out.mkdir(exist_ok=True)
        for fig_name, fig_path in figure_map.items():
            try:
                shutil.copy2(fig_path, figures_out / fig_name)
                # Copy PDF if exists
                pdf_path = fig_path.with_suffix(".pdf")
                if pdf_path.exists():
                    shutil.copy2(pdf_path, figures_out / pdf_path.name)
            except Exception as e:
                logger.warning(f"Could not copy figure {fig_name}: {e}")

        # Copy tables to output for download links
        tables_out = output_path / "tables"
        tables_out.mkdir(exist_ok=True)
        for tbl in report.get("tables", []):
            tbl_path = tbl.get("path", "")
            if tbl_path and Path(tbl_path).exists():
                try:
                    shutil.copy2(tbl_path, tables_out / os.path.basename(tbl_path))
                except Exception as e:
                    logger.warning(f"Could not copy table {tbl_path}: {e}")

        # Score definitions CSV — backs the "Download CSV" link in the section.
        try:
            from ...services.report.score_section import write_score_definitions_csv
            write_score_definitions_csv(report.get("score_definitions", []), tables_out)
        except Exception as e:
            logger.warning(f"Could not write score_definitions.csv: {e}")

        _analysis_state = report.get("_analysis_state")

        # Combined literature research report (linked from the appendix).
        if _analysis_state and _analysis_state.literature_reports:
            try:
                from ...services.report.literature_files import write_literature_report
                write_literature_report(_analysis_state.literature_reports, output_path)
            except Exception as e:
                logger.warning(f"Could not write literature report: {e}")

        # Copy notebook to output for download
        if _analysis_state:
            from ...config import get_session_states
            notebook_src = Path(get_session_states(_analysis_state.id)) / f"{_analysis_state.id}.ipynb"
            if notebook_src.exists():
                try:
                    shutil.copy2(notebook_src, output_path / f"{_analysis_state.id}.ipynb")
                except Exception as e:
                    logger.warning(f"Could not copy notebook: {e}")

        logger.info(f"HTML report generated: {html_file}")
        return html_file

    @staticmethod
    def _markdown_to_html(text: str) -> str:
        """Simple markdown to HTML conversion for report content."""
        if not text:
            return ""
        import html as html_module

        # Extract code blocks and inline code before escaping to avoid double-escaping
        code_blocks = []
        def _save_code_block(m):
            placeholder = f"\x00CODEBLOCK{len(code_blocks)}\x00"
            code_blocks.append(f'<pre><code>{html_module.escape(m.group(2))}</code></pre>')
            return placeholder
        text = re.sub(r'```(\w*)\n(.*?)```', _save_code_block, text, flags=re.DOTALL)

        inline_codes = []
        def _save_inline_code(m):
            placeholder = f"\x00INLINECODE{len(inline_codes)}\x00"
            inline_codes.append(f'<code>{html_module.escape(m.group(1))}</code>')
            return placeholder
        text = re.sub(r'`([^`]+)`', _save_inline_code, text)

        # Escape remaining (non-code) text
        text = html_module.escape(text)

        # Restore code blocks and inline code
        for i, block in enumerate(code_blocks):
            text = text.replace(f"\x00CODEBLOCK{i}\x00", block)
        for i, code in enumerate(inline_codes):
            text = text.replace(f"\x00INLINECODE{i}\x00", code)
        # Headers (longest prefix first)
        text = re.sub(r'^####\s+(.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
        text = re.sub(r'^###\s+(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^#\s+(.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        # Bullet lists
        text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
        text = re.sub(r'(<li>.*?</li>\n?)+', r'<ul>\g<0></ul>', text)
        # Paragraphs
        paragraphs = text.split('\n\n')
        result_parts = []
        for p in paragraphs:
            p = p.strip()
            if p and not p.startswith('<'):
                p = f'<p>{p}</p>'
            result_parts.append(p)
        text = '\n'.join(result_parts)
        # Line breaks
        text = text.replace('\n', '<br/>\n')
        # Clean up double <br/> in pre blocks
        text = re.sub(r'(<pre>.*?</pre>)', lambda m: m.group(0).replace('<br/>\n', '\n'), text, flags=re.DOTALL)
        return text

    def _build_appendix(self, state: AnalysisState, stats: Dict, record_files=None, reports_dir=None) -> Dict:
        """Build appendix metadata for the report."""
        # Build user interaction records
        cluster_annotations_data = []
        if state.annotation_state and state.annotation_state.cluster_annotations:
            for ca in state.annotation_state.cluster_annotations:
                cluster_annotations_data.append({
                    "cluster_id": ca.cluster_id,
                    "cell_type": ca.cell_type,
                    "user_override": ca.user_override or "",
                    "confirmed": ca.user_confirmed,
                    "confidence": ca.confidence,
                })

        from ...services.report.interaction_records import build_direction_selection_history
        direction_selection_data = build_direction_selection_history(state)

        agent_names = ("pi", "bioinformatics", "pathology", "biochemistry",
                       "literature", "painter", "supervisor")
        agent_models = {a: self._provider.get_model(a) for a in agent_names}
        agent_temperatures = {a: self._provider.get_temperature(a) for a in agent_names}
        # Show the model/mode actually used by the literature agent (Deep Research
        # vs the auxiliary PubMed LLM), not just the configured synthesis model.
        from ...services.report.sections import resolve_literature_model
        from ...services.report.literature_files import LITERATURE_REPORT_FILENAME
        agent_models["literature"] = resolve_literature_model(
            state, default=agent_models["literature"]
        )

        from ...services.report.download_section import build_related_reports_metadata
        related_reports = build_related_reports_metadata(
            state,
            record_files or {},
            reports_dir=reports_dir,
        )

        return {
            "project_name": state.name,
            "project_description": state.initial_instruction[:500],
            "created_at": datetime.now().isoformat(),
            "agent_models": agent_models,
            "agent_temperatures": agent_temperatures,
            "treatment_groups": [
                {"id": g.id, "name": g.name, "description": g.description, "sample_count": g.sample_count}
                for g in state.treatment_groups
            ],
            "data_files": [
                {"name": f.filename, "type": f.data_type, "section": f.section}
                for f in state.data_files
            ],
            "round_results": [
                {"round": rr.round_number, "findings_count": (
                    len(rr.bioinformatics_findings) + len(rr.pathology_findings) +
                    len(rr.biochemistry_findings) + len(rr.literature_findings)
                )}
                for rr in state.round_results
            ],
            "user_interactions": {
                "cluster_annotations": cluster_annotations_data,
                "direction_selection": direction_selection_data,
            },
            "related_reports": related_reports,
            "notebook_filename": f"{state.id}.ipynb",
            "literature_report": LITERATURE_REPORT_FILENAME if state.literature_reports else "",
        }

    def _build_appendix_html_section(self, report: Dict) -> str:
        """Build appendix HTML section from report data."""
        import html as html_module
        from ...services.report.regenerator import (
            get_agent_llm_stats,
            render_agent_config_table_html,
            AGENT_DISPLAY_NAMES,
        )
        appendix = report.get("appendix")
        if not appendix:
            return ""

        esc = html_module.escape

        parts = []
        parts.append('<div class="appendix" id="appendix">')
        parts.append('<h2>Appendix: Analysis Settings &amp; Records</h2>')

        # Project Information
        parts.append('<h3>Project Information</h3>')
        parts.append('<table>')
        parts.append(f'<tr><td><strong>Project Name</strong></td>'
                     f'<td>{esc(appendix.get("project_name", "-"))}</td></tr>')
        if appendix.get("created_at"):
            parts.append(f'<tr><td><strong>Created At</strong></td>'
                         f'<td>{esc(appendix.get("created_at", "-"))}</td></tr>')
        if report.get("generated_at"):
            parts.append(f'<tr><td><strong>Report Generated</strong></td>'
                         f'<td>{esc(report.get("generated_at", "-"))}</td></tr>')
        parts.append('</table>')

        models = appendix.get("agent_models", {})
        temps = appendix.get("agent_temperatures", {})
        if models:
            stats = get_agent_llm_stats(self.session_id) if self.session_id else {}
            config_rows = [
                (AGENT_DISPLAY_NAMES.get(key, key.title()), key, models[key], temps.get(key, "-"))
                for key in sorted(models.keys())
            ]
            parts.append(render_agent_config_table_html(config_rows, stats))

        # Data Files
        data_files = appendix.get("data_files", [])
        if data_files:
            parts.append('<h3>Data Files</h3>')
            parts.append('<table>')
            parts.append('<thead><tr><th>Filename</th>'
                         '<th>Type</th>'
                         '<th>Section</th></tr></thead><tbody>')
            for f in data_files:
                parts.append(f'<tr><td><code>{esc(f.get("name", "-"))}</code></td>'
                             f'<td>{esc(f.get("type", "-"))}</td>'
                             f'<td>{esc(f.get("section", "-"))}</td></tr>')
            parts.append('</tbody></table>')

        # Treatment Groups
        groups = appendix.get("treatment_groups", [])
        if groups:
            parts.append('<h3>Treatment Groups</h3>')
            parts.append('<table>')
            parts.append('<thead><tr><th>Group</th>'
                         '<th>Description</th>'
                         '<th>Samples</th></tr></thead><tbody>')
            for g in groups:
                parts.append(f'<tr><td>{esc(g.get("name", "-"))}</td>'
                             f'<td>{esc(g.get("description", "-"))}</td>'
                             f'<td>{g.get("sample_count", 0)}</td></tr>')
            parts.append('</tbody></table>')

        # Analysis Rounds
        round_results = appendix.get("round_results", [])
        if round_results:
            parts.append('<h3>Analysis Rounds</h3>')
            parts.append('<table>')
            parts.append('<thead><tr><th>Round</th>'
                         '<th>Findings</th></tr></thead><tbody>')
            for rr in round_results:
                parts.append(f'<tr><td>{rr.get("round", "-")}</td>'
                             f'<td>{rr.get("findings_count", 0)}</td></tr>')
            parts.append('</tbody></table>')

        # User Interaction History
        user_interactions = appendix.get("user_interactions", {})
        cluster_anns = user_interactions.get("cluster_annotations", [])
        dir_selections = user_interactions.get("direction_selection", [])

        if cluster_anns or dir_selections:
            parts.append('<h3>User Interaction History</h3>')

            if cluster_anns:
                parts.append('<h4>Cell Annotation Decisions</h4>')
                parts.append('<table>')
                parts.append('<thead><tr>'
                             '<th>Cluster</th>'
                             '<th>Cell Type</th>'
                             '<th>User Override</th>'
                             '<th>Confirmed</th>'
                             '<th>Confidence</th>'
                             '</tr></thead><tbody>')
                for ca in cluster_anns:
                    confirmed_icon = "✓" if ca.get("confirmed") else "-"
                    parts.append(
                        f'<tr><td>{esc(str(ca.get("cluster_id", "-")))}</td>'
                        f'<td>{esc(ca.get("cell_type", "-"))}</td>'
                        f'<td>{esc(ca.get("user_override", "-") or "-")}</td>'
                        f'<td>{confirmed_icon}</td>'
                        f'<td>{esc(ca.get("confidence", "-"))}</td></tr>'
                    )
                parts.append('</tbody></table>')

            if dir_selections:
                from ...services.report.interaction_records import (
                    render_direction_selection_history_html,
                )
                parts.append(render_direction_selection_history_html(dir_selections))

        # Related Reports & Downloads
        from ...services.report.download_section import render_related_reports_downloads_html
        downloads_html = render_related_reports_downloads_html(
            appendix.get("related_reports", {}),
            notebook_filename=appendix.get("notebook_filename", ""),
            literature_report=appendix.get("literature_report", ""),
        )
        if downloads_html:
            parts.append(downloads_html)

        parts.append('</div>')
        return "\n".join(parts)

    @staticmethod
    def _build_score_definitions_html(report: Dict) -> str:
        """Build the enhanced 'Score Definitions' section (shared with the
        services/report path so the two stay in sync)."""
        from ...services.report.score_section import build_score_definitions_section_html
        return build_score_definitions_section_html(report.get("score_definitions", []))

    @staticmethod
    def _categorize_figure(filename: str) -> str:
        """Categorize a figure by its filename."""
        name = filename.lower()
        if "umap" in name:
            return "umap"
        if "qc" in name or "quality" in name or "violin_qc" in name:
            return "qc"
        if "cluster" in name:
            return "clustering"
        if "marker" in name or "dotplot" in name:
            return "markers"
        if "heatmap" in name:
            return "heatmap"
        if "de_" in name or "volcano" in name or "differential" in name:
            return "de"
        if "violin" in name:
            return "violin"
        if "trajectory" in name or "paga" in name:
            return "trajectory"
        return "other"

    def _collect_figure_paths(
        self, records: List[Dict], workspace_path: Optional[Path],
        state: Optional[AnalysisState] = None,
    ) -> Dict[str, Path]:
        """Collect all figure file paths from supervision records, publication figures,
        and workspace directory scan.

        The workspace scan fallback is essential for report-only generation after
        session restore, where a fresh SupervisorAgent has empty supervision logs.
        """
        figure_map: Dict[str, Path] = {}
        seen_stems: set = set()
        if not workspace_path:
            return figure_map

        eligible_figures = eligible_artifacts(
            state.publication_figures if state else []
        )
        excluded_stems = (
            artifact_stems(state.publication_figures if state else [])
            - artifact_stems(eligible_figures)
        )
        if state:
            excluded_stems.update(state.invalidated_artifact_stems or [])

        for rec in records:
            for fname in rec.get("saved_figure_filenames", []):
                if Path(fname).stem in excluded_stems:
                    continue
                if fname in figure_map:
                    continue
                resolved = resolve_figure_in_workspace(fname, workspace_path, seen_stems)
                if resolved is not None:
                    seen_stems.add(resolved.stem)
                    figure_map[fname] = resolved

        # Also include publication figures (PI-reviewed, may not be in supervision records)
        if eligible_figures:
            for pf in eligible_figures:
                if not pf.png_path:
                    continue
                fname = os.path.basename(pf.png_path)
                if fname in figure_map:
                    continue
                fig_path = Path(pf.png_path)
                if not fig_path.exists():
                    fig_path = workspace_path / "figures" / fname
                if fig_path.exists():
                    figure_map[fname] = fig_path

        # Fallback: scan workspace directory for any figures not already collected.
        scan_dirs = [workspace_path]
        figures_subdir = workspace_path / "figures"
        if figures_subdir.exists():
            scan_dirs.append(figures_subdir)

        seen_stems = {Path(f).stem for f in figure_map}
        for scan_dir in scan_dirs:
            for img_file in sorted(scan_dir.glob("*.jpg")) + sorted(scan_dir.glob("*.png")):
                if img_file.stem in excluded_stems:
                    continue
                if img_file.stem in seen_stems:
                    continue
                seen_stems.add(img_file.stem)
                figure_map[img_file.name] = img_file

        # Deduplicate versioned figures: when both base and _v2/_v3/_final/_revised
        # exist, keep only the highest version and remove the base figure.
        _version_re = re.compile(r'^(.+?)_(v(\d+)|final|revised)$')
        base_to_versioned: Dict[str, List[str]] = {}
        for stem in list(seen_stems):
            m = _version_re.match(stem)
            if m:
                base_to_versioned.setdefault(m.group(1), []).append(stem)
        for base, versions in base_to_versioned.items():
            if base in seen_stems:
                # Remove the base version from figure_map, keep the versioned one(s)
                for fname in list(figure_map.keys()):
                    if Path(fname).stem == base:
                        del figure_map[fname]
                        logger.debug(f"Removed base figure {fname} in favor of versioned: {versions}")
                        break

        logger.info(f"Collected {len(figure_map)} figures for report "
                     f"({len(records)} supervision records, workspace={workspace_path})")

        return figure_map

    def _collect_table_paths(
        self, workspace_path: Optional[Path],
        invalidated_stems: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Scan workspace for CSV tables generated during analysis."""
        tables = []
        if not workspace_path:
            return tables

        tables_subdir = workspace_path / "tables"
        csv_search_dir = tables_subdir if tables_subdir.exists() else workspace_path

        for csv_file in sorted(csv_search_dir.glob("*.csv")):
            if csv_file.stem in (invalidated_stems or set()):
                continue
            table_info: Dict[str, Any] = {
                "filename": csv_file.name,
                "path": str(csv_file),
                "name": csv_file.stem.replace("_", " ").title(),
            }
            # Read preview
            try:
                import pandas as pd
                df_preview = pd.read_csv(csv_file, nrows=5)
                table_info["columns"] = list(df_preview.columns)
                table_info["preview_rows"] = len(df_preview)
                table_info["total_rows"] = sum(1 for _ in open(csv_file)) - 1
            except Exception:
                logger.debug("Failed to read CSV preview for table collection: %s", csv_file)
            tables.append(table_info)

        if tables:
            logger.info(f"Collected {len(tables)} tables from workspace")
        return tables

    @staticmethod
    def _compress_base64_figure(b64_png: str, max_width: int = 512) -> str:
        """Compress a base64 PNG to a smaller JPEG base64 for vision model review."""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(base64.b64decode(b64_png)))
            if img.width > max_width:
                ratio = max_width / img.width
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS
                img = img.resize((max_width, int(img.height * ratio)), resample)
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.warning(f"Image compression failed: {e}")
            return b64_png  # Return original if compression fails

    @staticmethod
    def _sanitize(obj):
        """Sanitize NaN/Infinity values for JSON serialization."""
        import math
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: ReportHtmlMixin._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ReportHtmlMixin._sanitize(v) for v in obj]
        return obj
