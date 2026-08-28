"""Sections mixin for DetailedReportGenerator."""
import html as html_module
import json

from backend.config import get_session_workspace
from backend.services.report.literature_files import render_literature_download_li
from backend.services.report.interaction_records import (
    build_direction_selection_history,
    render_direction_selection_history_html,
)
from backend.services.reproducibility import KEY_PACKAGES, REPRODUCIBILITY_DIR_NAME
from backend.services.report.quality import build_quality_disclosures
from typing import Dict, Any, List, Optional


def resolve_literature_model(state, default: str) -> str:
    """Literature row model for the report: prefer the actually-used label
    (DR vs PubMed) recorded during analysis, else the configured default."""
    actual = getattr(state, "literature_actual_model", None)
    return actual if actual else default


class SectionsMixin:
    """Provides abstract, conclusions, and appendix methods for report generation."""

    async def _generate_abstract(
        self,
        state: "AnalysisState",
        sections: List[Dict],
        findings: Dict[str, Any],
        language: str,
        analysis_summary: Optional[Dict[str, Any]] = None,
        force_fallback: bool = False,
    ) -> str:
        """PI Agent generates the report abstract"""
        import re

        analysis_summary = analysis_summary or {}
        section_summaries = "\n".join([
            f"- {s['title']}: {len(s.get('figures', []))} figures, {len(s.get('tables', []))} tables"
            for s in sections
        ])

        # Include methodology from analysis review
        methodology = analysis_summary.get("methodology_overview", "")
        key_results = analysis_summary.get("key_results", "")
        methodology_section = ""
        if methodology or key_results:
            methodology_section = f"""
## Analysis Methodology
{methodology[:1000] if methodology else ""}

## Key Quantitative Results
{key_results[:1000] if key_results else ""}
"""

        abstract_instruction = (
            "Write abstract in English, 150-250 words, including background, methods "
            "(specific analysis methods and parameters), main findings and conclusions. "
            "IMPORTANT: Output the abstract text directly. Do NOT start with a title like "
            "'**Abstract**' or '## Abstract' - the section heading is already provided by the report template."
        )

        prompt = f"""Write an abstract for this research report.

## Research Project
Title: {state.name}
Objective: {state.initial_instruction}

## Report Sections:
{section_summaries}
{methodology_section}

## Key Findings:
{json.dumps(findings['bioinformatics'][:15], ensure_ascii=False)}

## Final Summary:
{findings['summary'][:1500] if findings['summary'] else 'Not available'}

{abstract_instruction}
"""

        if force_fallback:
            summary = str(findings.get("summary") or "").strip()
            retained = summary[:1200] if summary else (
                "The retained analysis state contains the section-level results "
                "and artifacts listed below."
            )
            return (
                "This report was assembled deterministically because the "
                "report-writing model was unavailable. The analysis objective was: "
                f"{state.initial_instruction}. {retained}"
            )

        abstract = await self._pi_chat(prompt, temperature=0.3)
        # Strip redundant leading title that LLM may have added
        abstract = re.sub(r'^\*{0,2}Abstract\*{0,2}\s*\n*', '', abstract, flags=re.IGNORECASE).strip()
        abstract = re.sub(r'^#{1,4}\s+.*?Abstract.*?\n+', '', abstract, flags=re.IGNORECASE).strip()
        return abstract

    async def _generate_conclusions(
        self,
        state: "AnalysisState",
        sections: List[Dict],
        findings: Dict[str, Any],
        language: str,
        analysis_summary: Optional[Dict[str, Any]] = None,
        force_fallback: bool = False,
    ) -> str:
        """PI Agent generates the final conclusions"""
        import re

        analysis_summary = analysis_summary or {}
        quality = build_quality_disclosures(state)
        limitation_items = []
        for item in [
            *(quality.get("blocking_issues") or []),
            *(quality.get("cautions") or []),
        ]:
            text = str(item or "").strip()
            if text and text not in limitation_items:
                limitation_items.append(text[:800])
        limitations_text = (
            "\n".join(f"- {item}" for item in limitation_items[:40])
            or "- No unresolved machine-recorded evidence qualification was retained."
        )

        # Include key results from analysis review
        key_results = analysis_summary.get("key_results", "")
        results_section = ""
        if key_results:
            results_section = f"""
## Key Quantitative Results (from analysis review)
{key_results[:1500]}
"""

        conclusions_instruction = (
            "Write conclusions in English including: 1) Summary of main findings (cite specific numbers) "
            "2) Biological significance 3) Future directions, followed by exactly one "
            "**Limitations** subtopic that consolidates the supplied evidence qualifications. "
            "Do not repeat workflow-status labels elsewhere. 250-400 words. "
            "IMPORTANT: Output the conclusions text directly. Do NOT start with a header like "
            "'### Conclusions' - the section heading is already provided by the report template. "
            "Use **bold text** for subtopic labels (e.g., '**Summary of Findings**'), followed by paragraph text."
        )

        prompt = f"""Write the final conclusions for this research report.

## Research Objective:
{state.initial_instruction}

## Report Sections Generated:
{json.dumps([s['title'] for s in sections], ensure_ascii=False)}

## All Key Findings:
Bioinformatics: {json.dumps(findings['bioinformatics'][:10], ensure_ascii=False)}
Pathology: {json.dumps(findings['pathology'][:5], ensure_ascii=False)}
Biochemistry: {json.dumps(findings['biochemistry'][:5], ensure_ascii=False)}
{results_section}

## Final Summary from Analysis:
{findings['summary'][:2000] if findings['summary'] else 'Not available'}

## Evidence Qualifications to Consolidate:
{limitations_text}

{conclusions_instruction}
"""

        if force_fallback:
            summary = str(findings.get("summary") or "").strip()
            body = summary[:1800] if summary else (
                "No synthesized final summary was retained. Consult the report "
                "sections, figures, tables, and analysis-log appendix."
            )
            return (
                "**Degraded report.** The report-writing model was unavailable; "
                "these conclusions reproduce retained workflow evidence without "
                f"new model interpretation. {body}\n\n"
                "**Limitations** The report-writing model was unavailable, so this "
                "section contains no new PI interpretation."
            )

        conclusions = await self._pi_chat(prompt, temperature=0.3)
        # Strip redundant leading title that LLM may have added
        conclusions = re.sub(r'^#{1,4}\s+[\d.]*\s*Conclusions?.*?\n+', '', conclusions, flags=re.IGNORECASE).strip()
        return conclusions

    def _build_reproducibility_section_html(
        self,
        state: "AnalysisState",
    ) -> str:
        """Build the Reproducibility appendix section from on-disk bundle."""
        esc = html_module.escape
        workspace = get_session_workspace(state.id)
        repro_dir = workspace / REPRODUCIBILITY_DIR_NAME

        if not repro_dir.exists():
            return ""

        # Environment summary from system_info.json
        env_html = ""
        sysinfo_path = repro_dir / "system_info.json"
        if sysinfo_path.exists():
            try:
                info = json.loads(sysinfo_path.read_text(encoding="utf-8"))
                env_html = f"""
                <p><strong>Environment Summary</strong></p>
                <table>
                    <tbody>
                        <tr><td>Python</td><td><code>{esc(str(info.get('python_version', 'N/A'))[:120])}</code></td></tr>
                        <tr><td>Platform</td><td>{esc(info.get('platform', 'N/A'))}</td></tr>
                        <tr><td>ScAI Version</td><td>{esc(info.get('scai_version', 'N/A'))}</td></tr>
                        <tr><td>CPU Cores</td><td>{info.get('cpu_count', 'N/A')}</td></tr>
                        <tr><td>Memory</td><td>{info.get('memory_gb', 'N/A')} GB</td></tr>
                    </tbody>
                </table>
                """
            except Exception:
                pass

        # Key packages from frozen requirements
        key_pkgs_html = ""
        env_txt_path = repro_dir / "environment_frozen.txt"
        if env_txt_path.exists():
            try:
                pkg_lines = env_txt_path.read_text(encoding="utf-8").strip().split("\n")
                key_pkgs = [
                    ln for ln in pkg_lines
                    if any(pkg in ln.lower() for pkg in KEY_PACKAGES)
                ]
                if key_pkgs:
                    items = "".join(
                        f"<li><code>{esc(pkg)}</code></li>" for pkg in key_pkgs[:20]
                    )
                    key_pkgs_html = f"<p><strong>Key Packages</strong></p><ul>{items}</ul>"
            except Exception:
                pass

        # Checksum summary (first 8 entries)
        checksum_html = ""
        checksum_path = repro_dir / "checksums.sha256"
        if checksum_path.exists():
            try:
                chk_lines = [
                    ln.strip() for ln in checksum_path.read_text(encoding="utf-8").split("\n")
                    if ln.strip()
                ]
                if chk_lines:
                    rows = ""
                    for line in chk_lines[:8]:
                        parts = line.split("  ", 1)
                        sha_short = parts[0][:16] + "..." if len(parts) == 2 else line[:16] + "..."
                        fname = parts[1] if len(parts) == 2 else line
                        rows += f"<tr><td><code>{esc(sha_short)}</code></td><td>{esc(fname)}</td></tr>"
                    suffix = ""
                    if len(chk_lines) > 8:
                        suffix = f"<p>... and {len(chk_lines) - 8} more files</p>"
                    checksum_html = f"""
                    <p><strong>File Integrity</strong> ({len(chk_lines)} files verified)</p>
                    <table>
                        <thead><tr><th>SHA256 (first 16 chars)</th><th>File</th></tr></thead>
                        <tbody>{rows}</tbody>
                    </table>
                    {suffix}
                    """
            except Exception:
                pass

        # Download button
        dl_url = f"/api/session/{state.id}/report/download-reproducibility"
        download_btn = f"""
        <p>
            <a href="{esc(dl_url)}" download
               style="display:inline-block;padding:8px 16px;background:#4f46e5;color:#fff;
                      border-radius:6px;text-decoration:none;font-weight:500;">
                &#x1F5DC; Download Reproducibility Bundle (.zip)
            </a>
        </p>
        """

        return f"""
        <h3>Reproducibility</h3>
        <p>The following information ensures this analysis can be independently reproduced:</p>
        {download_btn}
        {env_html}
        {key_pkgs_html}
        {checksum_html}
        <p style="font-size:0.85em;color:#888;">
            The downloaded bundle contains the complete environment specification,
            SHA256 checksums for all files, executable Python commands, and agent
            decision logs for full computational reproducibility.
        </p>
        """

    def _build_appendix_html(
        self,
        state: "AnalysisState",
        report: Dict[str, Any],
        language: str,
    ) -> str:
        """
        Build the Appendix HTML section containing analysis settings and records.
        Pure data extraction — no LLM calls.
        """
        from backend.config import ModelConfig, AgentTemperatures
        from .regenerator import get_agent_llm_stats, render_agent_config_table_html

        esc = html_module.escape
        default_models = ModelConfig()
        default_temps = AgentTemperatures()

        appendix_title = "Appendix: Analysis Settings & Records"

        # ── 1. Project Information ──────────────────────────────────
        proj_title = "Project Information"
        lbl_name = "Project Name"
        lbl_desc = "Project Description"
        lbl_created = "Created At"
        lbl_report_time = "Report Generated At"

        created_str = state.created_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(state.created_at, 'strftime') else str(state.created_at)
        report_time = report.get("generated_at", "")

        project_html = f"""
        <h3>{proj_title}</h3>
        <table>
            <tbody>
                <tr><td><strong>{lbl_name}</strong></td><td>{esc(state.name)}</td></tr>
                <tr><td><strong>{lbl_desc}</strong></td><td>{esc(state.initial_instruction[:500]) if state.initial_instruction else '-'}</td></tr>
                <tr><td><strong>{lbl_created}</strong></td><td>{esc(created_str)}</td></tr>
                <tr><td><strong>{lbl_report_time}</strong></td><td>{esc(report_time)}</td></tr>
            </tbody>
        </table>
        """

        agent_keys = [
            ("PI", "pi", "pi_model"),
            ("Pathology", "pathology", "pathology_model"),
            ("Bioinformatics", "bioinformatics", "bioinformatics_model"),
            ("Biochemistry", "biochemistry", "biochemistry_model"),
            ("Literature", "literature", "literature_model"),
            ("Painter", "painter", "painter_model"),
        ]
        def _model_for(key, model_attr):
            base = state.custom_models.get(key, getattr(default_models, model_attr, "-"))
            if key == "literature":
                return resolve_literature_model(state, default=base)
            return base

        config_rows = [
            (
                display_name,
                key,
                _model_for(key, model_attr),
                state.custom_temperatures.get(key, getattr(default_temps, key, "-")),
            )
            for display_name, key, model_attr in agent_keys
        ]
        agent_html = render_agent_config_table_html(config_rows, get_agent_llm_stats(state.id))

        # ── 3. Uploaded Data ────────────────────────────────────────
        data_title = "Uploaded Data"

        # Treatment groups
        groups_html = ""
        if state.treatment_groups:
            lbl_group = "Treatment Groups"
            lbl_gname = "Group"
            lbl_gdesc = "Description"
            lbl_samples = "Samples"

            group_rows = ""
            for g in sorted(state.treatment_groups, key=lambda x: x.order):
                group_rows += f"<tr><td>{esc(g.name)}</td><td>{esc(g.description or '-')}</td><td>{g.sample_count}</td></tr>\n"
            groups_html = f"""
            <p><strong>{lbl_group}</strong></p>
            <table>
                <thead><tr><th>{lbl_gname}</th><th>{lbl_gdesc}</th><th>{lbl_samples}</th></tr></thead>
                <tbody>{group_rows}</tbody>
            </table>
            """

        # Data files
        files_html = ""
        if state.data_files:
            lbl_files = "Data Files"
            lbl_filename = "Filename"
            lbl_section = "Section"
            lbl_type = "Type"
            lbl_fgroup = "Group"

            file_rows = ""
            for f in state.data_files:
                group = state.get_group_by_id(f.group_id)
                group_name = group.name if group else ("-" if not f.group_id else f.group_id)
                file_rows += (
                    f"<tr><td>{esc(f.filename)}</td><td>{esc(f.section)}</td>"
                    f"<td>{esc(f.data_type)}</td><td>{esc(group_name)}</td></tr>\n"
                )
            files_html = f"""
            <p><strong>{lbl_files}</strong></p>
            <table>
                <thead><tr><th>{lbl_filename}</th><th>{lbl_section}</th><th>{lbl_type}</th><th>{lbl_fgroup}</th></tr></thead>
                <tbody>{file_rows}</tbody>
            </table>
            """

        # Biochemistry indicators
        biochem_html = ""
        if state.biochemistry_indicators:
            count = len(state.biochemistry_indicators)
            lbl_biochem = "Biochemistry Indicators"
            lbl_count = f"{count} manually entered indicator(s)"
            biochem_html = f"<p><strong>{lbl_biochem}:</strong> {lbl_count}</p>"

        data_html = f"""
        <h3>{data_title}</h3>
        {groups_html}
        {files_html}
        {biochem_html}
        """

        # ── 4. Analysis Rounds ──────────────────────────────────────
        rounds_title = "Analysis Rounds"

        rounds_html = ""
        if state.round_results:
            lbl_round = "Round"
            lbl_time = "Time"
            lbl_patho = "Pathology"
            lbl_bio = "Bioinformatics"
            lbl_chem = "Biochemistry"
            lbl_lit = "Literature"
            lbl_total_rounds = (
                f"Total: {len(state.round_results)} round(s) (max {state.max_rounds})"
            )

            round_rows = ""
            for r in state.round_results:
                ts = r.timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(r.timestamp, 'strftime') else str(r.timestamp)
                round_rows += (
                    f"<tr><td>{r.round_number}</td><td>{ts}</td>"
                    f"<td>{len(r.pathology_findings)}</td>"
                    f"<td>{len(r.bioinformatics_findings)}</td>"
                    f"<td>{len(r.biochemistry_findings)}</td>"
                    f"<td>{len(r.literature_findings)}</td></tr>\n"
                )

            rounds_html = f"""
            <h3>{rounds_title}</h3>
            <p>{lbl_total_rounds}</p>
            <table>
                <thead>
                    <tr>
                        <th>{lbl_round}</th><th>{lbl_time}</th>
                        <th>{lbl_patho}</th><th>{lbl_bio}</th><th>{lbl_chem}</th><th>{lbl_lit}</th>
                    </tr>
                </thead>
                <tbody>{round_rows}</tbody>
            </table>
            """
        else:
            lbl_no_rounds = "No analysis round records"
            rounds_html = f"<h3>{rounds_title}</h3><p>{lbl_no_rounds}</p>"

        # ── 5. User Inputs Between Rounds ───────────────────────────
        human_title = "User Inputs Between Rounds"

        human_html = ""
        if state.human_messages:
            items = ""
            for hm in state.human_messages:
                msg = hm.get("message", "")
                files = hm.get("files", [])
                ts = hm.get("timestamp", "")
                rn = hm.get("round_number", "?")
                round_label = f"Round {rn}"
                ts_str = ts if isinstance(ts, str) else (ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, 'strftime') else str(ts))

                files_info = ""
                if files:
                    file_names = ", ".join(
                        esc(f.get("filename", f.get("name", "file"))) for f in files
                    )
                    files_label = "Attached files"
                    files_info = f'<div class="files">{files_label}: {file_names}</div>'

                items += f"""
                <div class="human-input-item">
                    <span class="round-tag">{esc(round_label)}</span>
                    <span class="timestamp">{esc(ts_str)}</span>
                    <div class="message">{esc(msg)}</div>
                    {files_info}
                </div>
                """
            human_html = f"<h3>{human_title}</h3>{items}"
        else:
            lbl_no_input = "No user inputs recorded"
            human_html = f"<h3>{human_title}</h3><p>{lbl_no_input}</p>"

        # ── 6. User Interaction History ─────────────────────────────
        interaction_html = ""
        annotation_items = ""
        if state.annotation_state and state.annotation_state.cluster_annotations:
            ann_title = "Cell Annotation Decisions"
            lbl_cluster = "Cluster"
            lbl_ct = "Cell Type"
            lbl_override = "User Override"
            lbl_confirmed = "Confirmed"
            lbl_conf = "Confidence"
            ann_rows = ""
            for ca in state.annotation_state.cluster_annotations:
                override = esc(ca.user_override or "-") if ca.user_override else "-"
                confirmed = "✓" if ca.user_confirmed else "-"
                ann_rows += (
                    f"<tr><td>{esc(ca.cluster_id)}</td><td>{esc(ca.cell_type)}</td>"
                    f"<td>{override}</td><td>{confirmed}</td><td>{esc(ca.confidence)}</td></tr>\n"
                )
            annotation_items = f"""
            <h4>{ann_title}</h4>
            <table>
                <thead><tr><th>{lbl_cluster}</th><th>{lbl_ct}</th><th>{lbl_override}</th><th>{lbl_confirmed}</th><th>{lbl_conf}</th></tr></thead>
                <tbody>{ann_rows}</tbody>
            </table>
            """

        direction_items = render_direction_selection_history_html(
            build_direction_selection_history(state)
        )

        if annotation_items or direction_items:
            interaction_title = "User Interaction History"
            interaction_html = f"<h3>{interaction_title}</h3>{annotation_items}{direction_items}"

        # ── 7. Related Reports & Downloads ─────────────────────────
        from .interaction_records import direction_record_entries
        from .download_section import render_related_reports_downloads_html
        related_reports = {
            "annotation_record": (
                "annotation_record.html"
                if getattr(state, "annotation_record", None) is not None else None
            ),
            "direction_records": direction_record_entries(state),
            "integration_report": (
                "integration_report.html"
                if getattr(state, "integration_report_html_path", None) else None
            ),
        }
        downloads_html = render_related_reports_downloads_html(
            related_reports,
            notebook_filename=f"{state.id}.ipynb",
            literature_report="literature_report.html" if state.literature_reports else "",
        )

        # ── 8. Reproducibility ────────────────────────────────────────
        repro_html = self._build_reproducibility_section_html(state)

        # ── Assemble ────────────────────────────────────────────────
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
            {repro_html}
        </div>
        """
