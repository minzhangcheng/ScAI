"""HTML rendering of the detailed report: markdown, figures, tables, appendix.

Moved verbatim out of backend/agents/supervisor_agent.py; composed into SupervisorAgent.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
import logging
import os
import re
import io
import base64
import shutil
from pathlib import Path
from urllib.parse import quote as url_quote


import sys
from . import report_presentation

logger = logging.getLogger(__name__)


from ...models.schemas import AnalysisState
from ...services.report.artifacts import (
    artifact_stems,
    eligible_artifacts,
    excluded_artifact_stems,
    resolve_archival_render,
)
from ...services.report.figure_discovery import (
    discover_figures_in_workspace,
    ensure_legacy_artifact_inventory,
    prepare_discovered_figures,
)
from ...services.report.quality import (
    render_quality_disclosures_html,
    render_report_status_banner_html,
)


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
        self, report, sections, abstract, conclusions, report_figures,
        figure_map, stats, language, output_path, workspace_path,
    ) -> Path:
        report_presentation.bind_facade(sys.modules[__name__])
        ctx = report_presentation.prepare_html_render(
            self, report, sections, abstract, conclusions, report_figures,
            figure_map, stats, language, output_path, workspace_path,
        )
        report_presentation.render_html_sections(self, ctx)
        return report_presentation.persist_html_render(self, ctx)

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

        if state is not None:
            ensure_legacy_artifact_inventory(state, workspace_path)

        eligible_figures = eligible_artifacts(
            state.publication_figures if state else []
        )
        report_target_lineage = str(
            ((getattr(state, "sandbox_state", None) or {}).get(
                "_report_target_lineage_id", "main"
            ))
            or "main"
        )
        strict_lineage = report_target_lineage != "main"
        eligible_figure_stems = artifact_stems(eligible_figures)
        eligible_records_by_stem: Dict[str, Any] = {}
        for figure in eligible_figures:
            for stem in artifact_stems([figure]):
                eligible_records_by_stem.setdefault(stem, figure)
        included_record_ids: set[int] = set()
        excluded_stems = excluded_artifact_stems(
            state.publication_figures if state else [],
            eligible_figures,
            invalidated_stems=(
                getattr(state, "invalidated_artifact_stems", None) if state else None
            ),
        )
        admitted_candidates = prepare_discovered_figures(
            discover_figures_in_workspace(workspace_path),
            state.publication_figures if state else [],
            invalidated_stems=(
                getattr(state, "invalidated_artifact_stems", None) if state else None
            ),
            report_only=True,
            allow_unregistered_diagnostics=False,
            registry_version=(
                getattr(state, "artifact_registry_version", None) if state else None
            ),
            legacy_artifacts=(
                getattr(state, "legacy_artifacts", None) if state else None
            ),
        )
        admitted_by_stem = {
            Path(str(candidate.get("path") or "")).stem: candidate
            for candidate in admitted_candidates
            if candidate.get("path")
        }

        for rec in records:
            for fname in rec.get("saved_figure_filenames", []):
                requested_stem = Path(fname).stem
                if requested_stem in excluded_stems:
                    continue
                if fname in figure_map:
                    continue
                registered = eligible_records_by_stem.get(requested_stem)
                if registered is not None:
                    # A supervision record proves that a filename was emitted,
                    # not that bytes currently stored under that name are the
                    # reviewed bytes. Resolve through the registered artifact so
                    # a later same-name save cannot bypass its render hash.
                    resolved = resolve_archival_render(registered, workspace_path)
                else:
                    if strict_lineage:
                        # AnalysisStep/SupervisorRecord entries persisted before
                        # lineage tracking carry no ownership. Main reports keep
                        # that legacy fallback; a branch may only use a figure
                        # backed by one of its filtered PublicationFigure records.
                        continue
                    admitted = admitted_by_stem.get(requested_stem)
                    resolved = (
                        Path(str(admitted["path"]))
                        if admitted is not None else None
                    )
                if resolved is not None:
                    try:
                        rel_dirs = resolved.relative_to(workspace_path).parts[:-1]
                    except ValueError:
                        rel_dirs = ()
                    if (
                        "annotation_validation" in rel_dirs
                        and resolved.stem not in eligible_figure_stems
                    ):
                        continue
                    if (
                        registered is None
                        and any("supplement" in part.lower() for part in rel_dirs)
                    ):
                        # Historical restricted supplementary executors could
                        # leave images on disk and in saved_figure_filenames.
                        # A log entry is not publication review; registered
                        # user-requested supplementary figures remain allowed.
                        continue
                    seen_stems.add(resolved.stem)
                    if registered is not None:
                        included_record_ids.add(id(registered))
                        seen_stems.update(artifact_stems([registered]))
                        figure_map[resolved.name] = resolved
                    else:
                        figure_map[fname] = resolved

        # Also include publication figures (PI-reviewed, may not be in supervision
        # records). The excluded_stems check applies here too, and all three tiers
        # read the same set -- ``excluded_artifact_stems`` is where the two staleness
        # mechanisms are reconciled, so a stem can no longer overrule a record that
        # says the figure is current.
        if eligible_figures:
            for pf in eligible_figures:
                if id(pf) in included_record_ids:
                    continue
                declared_path = str(getattr(pf, "png_path", "") or "")
                fname = os.path.basename(declared_path) or f"{pf.name}.png"
                if Path(fname).stem in excluded_stems:
                    continue
                if fname in figure_map:
                    continue
                fig_path = resolve_archival_render(pf, workspace_path)
                if fig_path is not None:
                    fname = fig_path.name
                    figure_map[fname] = fig_path

        # Fallback: use the shared recursive discovery. This reaches
        # nested agent output consistently, including eligible registered
        # annotation-validation figures, without letting a stale/rejected record
        # return through its on-disk file. Unregistered validation diagnostics
        # remain annotation-context-only rather than entering this archival report.
        discovered = admitted_candidates
        seen_stems = {Path(f).stem for f in figure_map}
        for candidate in discovered:
            if strict_lineage and not candidate.get("is_publication", False):
                continue
            img_file = Path(candidate["path"])
            if img_file.stem in excluded_stems or img_file.stem in seen_stems:
                continue
            seen_stems.add(img_file.stem)
            figure_map[img_file.name] = img_file

        # Two names can hold the same file. The analysis saves a figure with
        # savefig and later re-registers it as a publication figure under a
        # different name; the workspace-scan tier then picks up both. Every dedupe
        # above compares filename strings, which cannot see that. Measured across
        # the real sessions on this host, three such byte-identical pairs exist.
        #
        # Scope, stated because the neighbouring case looks identical and is not
        # caught: this compares the raster the report embeds, so it collapses two
        # names over one file, not one drawing re-encoded twice. A run that saved a
        # panel as PNG and re-rendered it as JPG produces two different byte strings
        # of the same picture, and that pair still reaches the report twice --
        # session c31c0b42 shipped exactly that, its two PDFs differing only in the
        # three bytes of their creation timestamp. Deciding *pictures* are equal
        # needs a perceptual comparison, which is a different tool with a different
        # false-positive profile; this one is exact and cannot be wrong.
        registered_stems = artifact_stems(eligible_figures)
        by_digest: Dict[str, str] = {}
        for fname in list(figure_map):
            try:
                digest = hashlib.sha256(figure_map[fname].read_bytes()).hexdigest()
            except Exception:
                continue  # unreadable here is not fatal; embedding reports it
            kept = by_digest.get(digest)
            if kept is None:
                by_digest[digest] = fname
                continue
            # Keep whichever copy carries a publication record: that is the one
            # with a round number, a review status and a description, and losing
            # it would strip the surviving image of its provenance.
            loser = fname
            if (
                Path(fname).stem in registered_stems
                and Path(kept).stem not in registered_stems
            ):
                loser, by_digest[digest] = kept, fname
            figure_map.pop(loser, None)
            logger.info(
                f"Duplicate figure content: dropping {loser}, kept "
                f"{by_digest[digest]}"
            )

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
    def _compress_base64_figure(b64_png: str, max_width: int = 1024) -> str:
        """Compress a base64 PNG to a smaller JPEG base64 for vision model review.

        A publication figure is typically 2000-2500 px wide, so the previous
        512 px ceiling was a 4-5x linear reduction before the reviewer ever saw
        it — tick labels and legend entries on a multi-panel figure land at a
        few pixels of glyph height. 1024 px roughly quadruples the image tokens
        per figure while keeping small text resolvable, which is the whole point
        of showing the model the image.
        """
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
